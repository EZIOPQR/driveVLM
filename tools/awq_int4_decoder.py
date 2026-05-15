#!/usr/bin/env python3
"""
AWQ-style int4 weight quantization for DriveVLM Phi4 decoder.

Scope:
- Quantize decoder linear modules only:
  - self_attn.qkv_proj
  - self_attn.o_proj
  - mlp.gate_up_proj
  - mlp.down_proj
- Keep activations in FP16 (W4A16).
- Skip first/last decoder layers by count.
- Skip vision/audio towers naturally (decoder-only traversal).
- Optional: merge LoRA into base linear first (default), then quantize merged weights.

This script stores a reusable quantized delta (.pt) instead of trying to export a
fully standalone HF checkpoint, because custom int4 modules are injected at
runtime and need the same replacement logic during load.
"""

import argparse
import json
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


TARGET_LINEAR_NAME_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("self_attn.qkv_proj",),
    ("self_attn.o_proj", "self_attn.wo"),
    ("mlp.gate_up_proj",),
    ("mlp.down_proj",),
)


class AWQInt4LinearW4A16(nn.Module):
    """Simple int4 packed linear with on-the-fly dequantization (W4A16)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int,
        has_bias: bool,
    ) -> None:
        super().__init__()
        if group_size <= 0:
            group_size = in_features
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by group_size ({group_size})."
            )
        if in_features % 2 != 0:
            raise ValueError("in_features must be even for uint8 pack-two-int4 layout.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.groups = self.in_features // self.group_size

        # qweight layout: [out_features, in_features // 2], low4|high4
        self.register_buffer(
            "qweight",
            torch.zeros((self.out_features, self.in_features // 2), dtype=torch.uint8),
            persistent=True,
        )
        # Per-output-channel per-group quantization params.
        self.register_buffer(
            "scales",
            torch.zeros((self.out_features, self.groups), dtype=torch.float16),
            persistent=True,
        )
        self.register_buffer(
            "zeros",
            torch.zeros((self.out_features, self.groups), dtype=torch.uint8),
            persistent=True,
        )

        if has_bias:
            self.register_buffer(
                "bias",
                torch.zeros((self.out_features,), dtype=torch.float16),
                persistent=True,
            )
        else:
            self.bias = None

    @staticmethod
    def _pseudo_quantize_tensor_int4(
        weight: torch.Tensor, group_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        AWQ-style per-group affine quantization (zero_point=True), n_bit=4.
        Returns (qweight_int, scales, zeros) with shapes:
          qweight_int: [out_features, in_features] in [0, 15]
          scales:      [out_features, groups]
          zeros:       [out_features, groups]
        """
        if weight.dim() != 2:
            raise ValueError("Expected 2D weight tensor [out_features, in_features].")
        out_features, in_features = weight.shape
        if group_size <= 0:
            group_size = in_features
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) not divisible by group_size ({group_size})."
            )

        groups = in_features // group_size
        w = weight.reshape(out_features * groups, group_size)
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)

        max_int = 15.0
        min_int = 0.0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp(min_int, max_int)

        q = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int).to(torch.uint8)
        q = q.reshape(out_features, in_features)
        scales = scales.reshape(out_features, groups)
        zeros = zeros.reshape(out_features, groups).to(torch.uint8)
        return q, scales, zeros

    @staticmethod
    def _pack_int4(qweight_int: torch.Tensor) -> torch.Tensor:
        if qweight_int.dtype != torch.uint8:
            raise ValueError("qweight_int must be uint8.")
        if qweight_int.shape[1] % 2 != 0:
            raise ValueError("in_features must be even to pack int4 pairs.")
        low = qweight_int[:, 0::2] & 0x0F
        high = (qweight_int[:, 1::2] & 0x0F) << 4
        return (low | high).contiguous()

    @staticmethod
    def _unpack_int4(qweight_packed: torch.Tensor) -> torch.Tensor:
        low = qweight_packed & 0x0F
        high = (qweight_packed >> 4) & 0x0F
        out = torch.empty(
            (qweight_packed.shape[0], qweight_packed.shape[1] * 2),
            dtype=torch.uint8,
            device=qweight_packed.device,
        )
        out[:, 0::2] = low
        out[:, 1::2] = high
        return out

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128) -> "AWQInt4LinearW4A16":
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            has_bias=linear.bias is not None,
        )
        w_fp32 = linear.weight.detach().to(torch.float32)
        q_int, scales, zeros = cls._pseudo_quantize_tensor_int4(w_fp32, module.group_size)
        module.qweight = cls._pack_int4(q_int).to(linear.weight.device)
        module.scales = scales.to(dtype=torch.float16, device=linear.weight.device)
        module.zeros = zeros.to(device=linear.weight.device)
        if linear.bias is not None:
            module.bias = linear.bias.detach().to(dtype=torch.float16, device=linear.weight.device)
        return module

    def _dequant_weight(self, dtype: torch.dtype) -> torch.Tensor:
        q = self._unpack_int4(self.qweight).to(dtype=torch.float32)
        scales = self.scales.to(dtype=torch.float32)
        zeros = self.zeros.to(dtype=torch.float32)
        # Expand [out, groups] -> [out, in_features] by repeating each group.
        scales_full = scales.repeat_interleave(self.group_size, dim=1)
        zeros_full = zeros.repeat_interleave(self.group_size, dim=1)
        w = (q - zeros_full) * scales_full
        return w.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequant_weight(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def _set_submodule(root: nn.Module, path: str, new_module: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _resolve_target_name(layer: nn.Module, candidates: Sequence[str], layer_name: str) -> str:
    found: List[str] = []
    for name in candidates:
        try:
            layer.get_submodule(name)
            found.append(name)
        except AttributeError:
            continue
    if len(found) == 1:
        return found[0]
    if not found:
        raise KeyError(
            f"{layer_name}: none of expected target names exists: {list(candidates)}. "
            "Check model architecture or target config."
        )
    raise RuntimeError(
        f"{layer_name}: multiple target names matched {found} from candidates {list(candidates)}; "
        "target mapping is ambiguous."
    )


def _resolve_target_name_non_strict(layer: nn.Module, candidates: Sequence[str]) -> Tuple[Optional[str], List[str]]:
    found: List[str] = []
    for name in candidates:
        try:
            layer.get_submodule(name)
            found.append(name)
        except AttributeError:
            continue
    if len(found) == 1:
        return found[0], found
    return None, found


def _quantize_one_target(layer: nn.Module, target_name: str, group_size: int) -> Tuple[bool, str]:
    target = layer.get_submodule(target_name)
    # Policy: skip LoRA wrappers entirely to avoid changing adapter training/inference behavior.
    if hasattr(target, "base_layer"):
        return False, f"{target_name}: LoRA wrapper detected and skipped by policy"
    if not isinstance(target, nn.Linear):
        raise TypeError(
            f"{target_name}: expected nn.Linear, got {type(target).__name__}. "
            "Refusing silent fallback."
        )
  
    q_module = AWQInt4LinearW4A16.from_linear(target, group_size=group_size)
    _set_submodule(layer, target_name, q_module.to(next(layer.parameters()).device))
    return True, f"{target_name}: quantized nn.Linear"


def merge_lora_into_base_linear(
    model: nn.Module,
    adapter_name: str,
    target_name_groups: Tuple[Tuple[str, ...], ...] = TARGET_LINEAR_NAME_GROUPS,
) -> int:
    """
    Merge LoRA adapter weights into base linear and replace wrappers with base layers.
    Returns number of merged target modules.
    """
    if hasattr(model, "set_lora_adapter"):
        model.set_lora_adapter(adapter_name)

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Model does not expose model.layers decoder stack.")

    merged_count = 0
    for layer_idx, layer in enumerate(model.model.layers):
        layer_name = f"model.layers.{layer_idx}"
        for target_candidates in target_name_groups:
            target_name = _resolve_target_name(layer, target_candidates, layer_name)
            module_name = f"{layer_name}.{target_name}"
            module = layer.get_submodule(target_name)
            if isinstance(module, nn.Linear):
                continue
            if not hasattr(module, "base_layer"):
                raise TypeError(
                    f"{module_name}: expected nn.Linear or LoRA wrapper, got {type(module).__name__}"
                )
            lora_a = getattr(module, "lora_A", None)
            if lora_a is None:
                raise TypeError(f"{module_name}: LoRA wrapper missing lora_A mapping.")
            if adapter_name not in lora_a:
                continue
            merge_fn = getattr(module, "merge", None)
            if not callable(merge_fn):
                raise TypeError(
                    f"{module_name}: LoRA wrapper has no callable merge() method; "
                    f"cannot merge adapter '{adapter_name}'."
                )
            merge_fn()
            base_layer = getattr(module, "base_layer", None)
            if not isinstance(base_layer, nn.Linear):
                raise TypeError(
                    f"{module_name}: expected merged base_layer to be nn.Linear, got {type(base_layer).__name__}"
                )
            _set_submodule(layer, target_name, base_layer)
            merged_count += 1
    return merged_count


def apply_decoder_awq_int4(
    model: nn.Module,
    group_size: int,
    skip_first_n: int,
    skip_last_n: int,
    target_name_groups: Tuple[Tuple[str, ...], ...] = TARGET_LINEAR_NAME_GROUPS,
) -> Dict[str, List[str]]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Model does not expose model.layers decoder stack.")
    layers = model.model.layers
    total_layers = len(layers)
    if skip_first_n < 0 or skip_last_n < 0 or (skip_first_n + skip_last_n) >= total_layers:
        raise ValueError(
            f"Invalid skip range: total={total_layers}, skip_first={skip_first_n}, skip_last={skip_last_n}"
        )

    skip_indices = set(range(skip_first_n)) | set(range(total_layers - skip_last_n, total_layers))
    quantized_modules: List[str] = []
    skipped_logs: List[str] = []

    for layer_idx, layer in enumerate(layers):
        if layer_idx in skip_indices:
            skipped_logs.append(f"layer {layer_idx}: skipped by boundary rule")
            continue
        layer_name = f"model.layers.{layer_idx}"
        for target_candidates in target_name_groups:
            target_name = _resolve_target_name(layer, target_candidates, layer_name)
            ok, log = _quantize_one_target(layer, target_name, group_size=group_size)
            full_name = f"model.layers.{layer_idx}.{target_name}"
            if ok:
                quantized_modules.append(full_name)
            else:
                skipped_logs.append(f"{full_name} -> {log}")

    if not quantized_modules:
        raise RuntimeError("No decoder linear modules were quantized. Aborting.")
    return {"quantized_modules": quantized_modules, "skipped": skipped_logs}


def apply_awq_delta_to_model(model: nn.Module, delta_path: str, map_location: str = "cpu") -> None:
    """Apply a saved AWQ int4 delta to a freshly loaded base model."""
    delta = torch.load(delta_path, map_location=map_location)
    if "meta" not in delta or "module_states" not in delta:
        raise KeyError(f"Invalid AWQ delta format: missing required keys in {delta_path}")
    meta = delta["meta"]
    required_meta = ("group_size", "skip_first_n", "skip_last_n", "target_name_groups")
    missing = [k for k in required_meta if k not in meta]
    if missing:
        raise KeyError(f"Invalid AWQ delta meta: missing keys {missing}")
    target_name_groups = tuple(tuple(v) for v in meta["target_name_groups"])
    lora_policy = str(meta.get("lora_policy", "skip_wrapper"))
    if lora_policy.startswith("merge_adapter:"):
        adapter_name = lora_policy.split(":", 1)[1].strip()
        if not adapter_name:
            raise ValueError(f"Invalid lora_policy in delta meta: {lora_policy}")
        merged_count = merge_lora_into_base_linear(
            model=model,
            adapter_name=adapter_name,
            target_name_groups=target_name_groups,
        )
        if merged_count <= 0:
            raise RuntimeError(
                f"Delta expects merged LoRA adapter '{adapter_name}' but no target module was merged. "
                "Please ensure base_model_path points to the same LoRA architecture/checkpoint family "
                "used during AWQ export."
            )
    apply_decoder_awq_int4(
        model=model,
        group_size=int(meta["group_size"]),
        skip_first_n=int(meta["skip_first_n"]),
        skip_last_n=int(meta["skip_last_n"]),
        target_name_groups=target_name_groups,
    )
    module_states: Dict[str, Dict[str, torch.Tensor]] = delta["module_states"]
    for module_name, state in module_states.items():
        module = model.get_submodule(module_name)
        module.load_state_dict(state, strict=True)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AWQ int4 quantization for DriveVLM decoder linear layers")
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/epoch-4", help="Input checkpoint directory")
    parser.add_argument(
        "--output",
        type=str,
        default="/root/autodl-tmp/epoch-4-awq/decoder_awq_int4_delta.pt",
        help="Output delta path (.pt)",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default="",
        help=(
            "Optional export folder for quantized checkpoint package. "
            "Will contain awq delta + copied tokenizer/config files."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for quantization")
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Per-group quantization size; must divide in_features. Use -1 for per-channel full row.",
    )
    parser.add_argument("--skip_first_n", type=int, default=1, help="Skip first N decoder layers")
    parser.add_argument("--skip_last_n", type=int, default=1, help="Skip last N decoder layers")
    parser.add_argument(
        "--merge_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to merge LoRA adapter into base linear before quantization (default: true).",
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        default="vision",
        help="Adapter name to merge when --merge_lora is enabled.",
    )
    parser.add_argument(
        "--layer_report",
        type=str,
        default="",
        help=(
            "Optional output path for hierarchical layer quantization report JSON. "
            "Default: <output_basename>.layers.json"
        ),
    )
    return parser


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if not parent:
        raise ValueError(f"Output path has no parent directory: {path}")
    os.makedirs(parent, exist_ok=True)


def _infer_target_status(module: nn.Module) -> Tuple[str, str]:
    if isinstance(module, AWQInt4LinearW4A16):
        return "quantized", type(module).__name__
    if isinstance(module, nn.Linear):
        return "not_quantized_linear", type(module).__name__
    if hasattr(module, "base_layer"):
        return "lora_wrapper_unquantized", type(module).__name__
    return "not_quantized_other", type(module).__name__


def export_layer_quantization_report(
    model: nn.Module,
    report_path: str,
    quantized_modules: List[str],
    skip_first_n: int,
    skip_last_n: int,
    target_name_groups: Tuple[Tuple[str, ...], ...] = TARGET_LINEAR_NAME_GROUPS,
) -> None:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Model does not expose model.layers decoder stack.")
    _ensure_parent_dir(report_path)

    quantized_set = set(quantized_modules)
    layers = model.model.layers
    total_layers = len(layers)
    skip_indices = set(range(skip_first_n)) | set(range(total_layers - skip_last_n, total_layers))
    layer_items: List[Dict[str, Any]] = []

    for layer_idx, layer in enumerate(layers):
        layer_name = f"model.layers.{layer_idx}"
        layer_entry: Dict[str, Any] = {
            "layer_name": layer_name,
            "participates_quantization": layer_idx not in skip_indices,
            "skip_reason": "boundary_rule" if layer_idx in skip_indices else "",
            "targets": [],
        }

        for candidates in target_name_groups:
            resolved_name, matched_names = _resolve_target_name_non_strict(layer, candidates)
            target_entry: Dict[str, Any] = {
                "candidates": list(candidates),
                "resolved_name": resolved_name or "",
                "matched_names": matched_names,
                "module_path": "",
                "participates_quantization": False,
                "status": "not_found" if not matched_names else "ambiguous_target_mapping",
                "module_type": "",
            }
            if resolved_name is not None:
                module_path = f"{layer_name}.{resolved_name}"
                module = layer.get_submodule(resolved_name)
                status, module_type = _infer_target_status(module)
                target_entry.update(
                    {
                        "module_path": module_path,
                        "participates_quantization": module_path in quantized_set,
                        "status": status,
                        "module_type": module_type,
                    }
                )
            layer_entry["targets"].append(target_entry)
        layer_items.append(layer_entry)

    report = {
        "format": "drivevlm_awq_layer_report_v1",
        "summary": {
            "total_decoder_layers": total_layers,
            "skip_first_n": skip_first_n,
            "skip_last_n": skip_last_n,
            "target_name_groups": [list(x) for x in target_name_groups],
            "quantized_module_count": len(quantized_modules),
            "quantized_modules": quantized_modules,
        },
        "model_tree": {
            "decoder": {"layers": layer_items},
            "vision_tower": {
                "participates_quantization": False,
                "reason": "decoder-only quantization scope",
            },
            "audio_tower": {
                "participates_quantization": False,
                "reason": "decoder-only quantization scope",
            },
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _copy_base_runtime_files(base_model_dir: str, export_dir: str) -> None:
    """Copy minimal runtime files for loading base model + AWQ delta."""
    os.makedirs(export_dir, exist_ok=True)
    required_files = [
        "config.json",
        "configuration_phi4mm.py",
        "modeling_phi4mm.py",
        "processing_phi4mm.py",
        "vision_siglip_navit.py",
    ]
    optional_files = [
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.json",
        # Audio path is not required for current project workflow, keep optional.
        "speech_conformer_encoder.py",
    ]
    index_file = "model.safetensors.index.json"
    if os.path.exists(os.path.join(base_model_dir, index_file)):
        optional_files.append(index_file)

    missing_required = [fn for fn in required_files if not os.path.exists(os.path.join(base_model_dir, fn))]
    if missing_required:
        raise FileNotFoundError(
            f"Missing required runtime files in base model dir {base_model_dir}: {missing_required}"
        )

    for fn in required_files + optional_files:
        src = os.path.join(base_model_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(export_dir, fn))

    # Keep model weights out of export package to avoid huge duplication.
    # Runtime should load base weights from `base_model_path` in manifest and then
    # apply `awq_delta`.


def main() -> None:
    args = _build_argparser().parse_args()
    _ensure_parent_dir(args.output)

    print(f"[awq] loading model from: {args.model}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",
        )
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        msg = str(exc)
        if "flash_attn_2_cuda" in msg and "undefined symbol" in msg:
            raise RuntimeError(
                "Detected incompatible flash_attn build in current environment. "
                "Install a wheel matching torch/cuda/abi before running quantization."
            ) from exc
        raise
    model.to(args.device)
    model.eval()

    if args.merge_lora:
        merged_count = merge_lora_into_base_linear(
            model,
            adapter_name=args.lora_adapter,
            target_name_groups=TARGET_LINEAR_NAME_GROUPS,
        )
        if merged_count <= 0:
            raise RuntimeError(
                f"--merge_lora is enabled but no LoRA module matched adapter '{args.lora_adapter}'. "
                "Disable with --no-merge_lora if this checkpoint is already merged or has no LoRA."
            )
        print(f"[awq] merged LoRA adapter '{args.lora_adapter}' into {merged_count} modules.")
    else:
        print("[awq] merge_lora disabled: LoRA wrappers will be skipped during quantization.")

    print("[awq] applying int4 quantization on decoder target linear layers ...")
    result = apply_decoder_awq_int4(
        model=model,
        group_size=args.group_size,
        skip_first_n=args.skip_first_n,
        skip_last_n=args.skip_last_n,
    )

    module_states = {
        module_name: model.get_submodule(module_name).state_dict()
        for module_name in result["quantized_modules"]
    }

    delta_state = {
        "meta": {
            "algorithm": "awq_style_int4_weight_only",
            "activation_dtype": "float16",
            "weight_bits": 4,
            "group_size": args.group_size,
            "skip_first_n": args.skip_first_n,
            "skip_last_n": args.skip_last_n,
            "target_name_groups": [list(x) for x in TARGET_LINEAR_NAME_GROUPS],
            "model_path": args.model,
            "scope": "decoder_only_linear_qkv_wo_ffn",
            "lora_policy": (
                f"merge_adapter:{args.lora_adapter}" if args.merge_lora else "skip_wrapper"
            ),
        },
        "quantized_modules": result["quantized_modules"],
        "module_states": module_states,
        "skipped": result["skipped"],
    }
    torch.save(delta_state, args.output)

    meta_path = os.path.splitext(args.output)[0] + ".json"
    layer_report_path = args.layer_report or (os.path.splitext(args.output)[0] + ".layers.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": delta_state["meta"],
                "quantized_count": len(result["quantized_modules"]),
                "quantized_modules": result["quantized_modules"],
                "skipped_count": len(result["skipped"]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    export_layer_quantization_report(
        model=model,
        report_path=layer_report_path,
        quantized_modules=result["quantized_modules"],
        skip_first_n=args.skip_first_n,
        skip_last_n=args.skip_last_n,
        target_name_groups=TARGET_LINEAR_NAME_GROUPS,
    )

    if args.export_dir:
        _copy_base_runtime_files(args.model, args.export_dir)
        export_delta = os.path.join(args.export_dir, "awq_int4_delta.pt")
        shutil.copy2(args.output, export_delta)
        manifest = {
            "format": "drivevlm_awq_int4_decoder_package_v1",
            "base_model_path": args.model,
            "awq_delta": "awq_int4_delta.pt",
            "meta_json": os.path.basename(meta_path),
            "layer_report_json": os.path.basename(layer_report_path),
            "notes": [
                "Load base model from base_model_path with trust_remote_code=True.",
                "Call apply_awq_delta_to_model(model, awq_delta_path).",
                "This package is decoder-weight-only int4 AWQ style; activations stay fp16.",
            ],
        }
        with open(os.path.join(args.export_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        export_meta = os.path.join(args.export_dir, os.path.basename(meta_path))
        if os.path.abspath(meta_path) != os.path.abspath(export_meta):
            shutil.copy2(meta_path, export_meta)
        export_report = os.path.join(args.export_dir, os.path.basename(layer_report_path))
        if os.path.abspath(layer_report_path) != os.path.abspath(export_report):
            shutil.copy2(layer_report_path, export_report)

    print(f"[awq] done. quantized modules: {len(result['quantized_modules'])}")
    print(f"[awq] delta saved to: {args.output}")
    print(f"[awq] meta  saved to: {meta_path}")
    print(f"[awq] layer report saved to: {layer_report_path}")
    if args.export_dir:
        print(f"[awq] export package: {args.export_dir}")


if __name__ == "__main__":
    main()
