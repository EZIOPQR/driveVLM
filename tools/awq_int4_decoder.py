#!/usr/bin/env python3
"""
AWQ-style int4 weight quantization for DriveVLM Phi4 decoder.

Scope:
- Quantize decoder linear modules only:
  - self_attn.qkv_proj
  - self_attn.o_proj
  - mlp.gate_up_proj
  - mlp.down_proj
- Keep activations in BF16.
- Skip first/last decoder layers by count.
- Skip vision/audio towers naturally (decoder-only traversal).
- Keep LoRA adapters unquantized by replacing only LoRA base linear when present.

This script stores a reusable quantized delta (.pt) instead of trying to export a
fully standalone HF checkpoint, because custom int4 modules are injected at
runtime and need the same replacement logic during load.
"""

import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


TARGET_LINEAR_NAMES: Tuple[str, ...] = (
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "mlp.gate_up_proj",
    "mlp.down_proj",
)


class AWQInt4LinearBF16(nn.Module):
    """Simple int4 packed linear with on-the-fly dequantization to activation dtype."""

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
            torch.zeros((self.out_features, self.groups), dtype=torch.bfloat16),
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
                torch.zeros((self.out_features,), dtype=torch.bfloat16),
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
    def from_linear(cls, linear: nn.Linear, group_size: int = 128) -> "AWQInt4LinearBF16":
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            has_bias=linear.bias is not None,
        )
        w_fp32 = linear.weight.detach().to(torch.float32)
        q_int, scales, zeros = cls._pseudo_quantize_tensor_int4(w_fp32, module.group_size)
        module.qweight = cls._pack_int4(q_int).to(linear.weight.device)
        module.scales = scales.to(dtype=torch.bfloat16, device=linear.weight.device)
        module.zeros = zeros.to(device=linear.weight.device)
        if linear.bias is not None:
            module.bias = linear.bias.detach().to(dtype=torch.bfloat16, device=linear.weight.device)
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


def _quantize_one_target(
    layer: nn.Module, target_name: str, group_size: int
) -> Tuple[bool, str]:
    target = layer.get_submodule(target_name)
    # Keep LoRA wrapper untouched, quantize only its base linear.
    if hasattr(target, "base_layer"):
        base = getattr(target, "base_layer")
        if not isinstance(base, nn.Linear):
            return False, f"{target_name}: base_layer is not nn.Linear, skipped"
        q_module = AWQInt4LinearBF16.from_linear(base, group_size=group_size)
        target.base_layer = q_module.to(next(layer.parameters()).device)
        return True, f"{target_name}: quantized LoRA base_layer"
    if isinstance(target, nn.Linear):
        q_module = AWQInt4LinearBF16.from_linear(target, group_size=group_size)
        _set_submodule(layer, target_name, q_module.to(next(layer.parameters()).device))
        return True, f"{target_name}: quantized nn.Linear"
    return False, f"{target_name}: unsupported module type {type(target).__name__}, skipped"


def apply_decoder_awq_int4(
    model: nn.Module,
    group_size: int,
    skip_first_n: int,
    skip_last_n: int,
    target_names: Tuple[str, ...] = TARGET_LINEAR_NAMES,
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
        for target_name in target_names:
            ok, log = _quantize_one_target(layer, target_name, group_size=group_size)
            full_name = f"model.layers.{layer_idx}.{target_name}"
            if ok:
                quantized_modules.append(full_name)
            else:
                skipped_logs.append(f"{full_name} -> {log}")

    return {"quantized_modules": quantized_modules, "skipped": skipped_logs}


def apply_awq_delta_to_model(model: nn.Module, delta_path: str, map_location: str = "cpu") -> None:
    """Apply a saved AWQ int4 delta to a freshly loaded base model."""
    delta = torch.load(delta_path, map_location=map_location)
    meta = delta["meta"]
    apply_decoder_awq_int4(
        model=model,
        group_size=int(meta["group_size"]),
        skip_first_n=int(meta["skip_first_n"]),
        skip_last_n=int(meta["skip_last_n"]),
        target_names=tuple(meta["target_names"]),
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
    return parser


def _copy_base_runtime_files(base_model_dir: str, export_dir: str) -> None:
    """Copy minimal runtime files for loading base model + AWQ delta."""
    os.makedirs(export_dir, exist_ok=True)
    must_copy = [
        "config.json",
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
        "configuration_phi4mm.py",
        "modeling_phi4mm.py",
        "processing_phi4mm.py",
        "speech_conformer_encoder.py",
        "vision_siglip_navit.py",
    ]
    index_file = "model.safetensors.index.json"
    if os.path.exists(os.path.join(base_model_dir, index_file)):
        must_copy.append(index_file)

    for fn in must_copy:
        src = os.path.join(base_model_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(export_dir, fn))

    # Keep model weights out of export package to avoid huge duplication.
    # Runtime should load base weights from `base_model_path` in manifest and then
    # apply `awq_delta`.


def main() -> None:
    args = _build_argparser().parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"[awq] loading model from: {args.model}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
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
            "activation_dtype": "bfloat16",
            "weight_bits": 4,
            "group_size": args.group_size,
            "skip_first_n": args.skip_first_n,
            "skip_last_n": args.skip_last_n,
            "target_names": list(TARGET_LINEAR_NAMES),
            "model_path": args.model,
        },
        "quantized_modules": result["quantized_modules"],
        "module_states": module_states,
        "skipped": result["skipped"],
    }
    torch.save(delta_state, args.output)

    meta_path = os.path.splitext(args.output)[0] + ".json"
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

    if args.export_dir:
        _copy_base_runtime_files(args.model, args.export_dir)
        export_delta = os.path.join(args.export_dir, "awq_int4_delta.pt")
        shutil.copy2(args.output, export_delta)
        manifest = {
            "format": "drivevlm_awq_int4_decoder_package_v1",
            "base_model_path": args.model,
            "awq_delta": "awq_int4_delta.pt",
            "meta_json": os.path.basename(meta_path),
            "notes": [
                "Load base model from base_model_path with trust_remote_code=True.",
                "Call apply_awq_delta_to_model(model, awq_delta_path).",
                "This package is decoder-weight-only int4 AWQ style; activations stay bf16.",
            ],
        }
        with open(os.path.join(args.export_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        export_meta = os.path.join(args.export_dir, os.path.basename(meta_path))
        if os.path.abspath(meta_path) != os.path.abspath(export_meta):
            shutil.copy2(meta_path, export_meta)

    print(f"[awq] done. quantized modules: {len(result['quantized_modules'])}")
    print(f"[awq] delta saved to: {args.output}")
    print(f"[awq] meta  saved to: {meta_path}")
    if args.export_dir:
        print(f"[awq] export package: {args.export_dir}")


if __name__ == "__main__":
    main()
