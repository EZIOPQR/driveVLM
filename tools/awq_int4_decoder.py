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
import math
import sys
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from drivevlms.build import build_collate_fn
from tools.quant_calib_utils import select_balanced_calibration_indices


TARGET_LINEAR_NAME_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("self_attn.qkv_proj",),
    ("self_attn.o_proj", "self_attn.wo"),
    ("mlp.gate_up_proj",),
    ("mlp.down_proj",),
)


class AWQInt4LinearW4A16(nn.Module):
    """Int4 weight-only linear backed by fused PyTorch int4 GEMM kernel."""

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
        self.register_buffer(
            "qweight_fused",
            torch.empty(0, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "q_scale_and_zeros",
            torch.empty(0, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "bias_fused",
            torch.empty(0, dtype=torch.bfloat16),
            persistent=False,
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
        module.rebuild_fused_params()
        return module

    def rebuild_fused_params(self) -> None:
        qweight = (((self.qweight & 0x0F) << 4) | ((self.qweight >> 4) & 0x0F)).contiguous()
        scales = self.scales.transpose(0, 1).to(dtype=torch.bfloat16).contiguous()
        zeros = self.zeros.transpose(0, 1).to(dtype=torch.bfloat16).contiguous()
        self.qweight_fused = torch.ops.aten._convert_weight_to_int4pack(qweight, 8)
        self.q_scale_and_zeros = torch.stack((scales, scales * (8.0 - zeros)), dim=-1).contiguous()
        if self.bias is not None:
            self.bias_fused = self.bias.to(dtype=torch.bfloat16).contiguous()
        else:
            self.bias_fused = torch.empty(0, dtype=torch.bfloat16, device=self.qweight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("AWQ fused int4 kernel requires CUDA tensor input.")
        if x.dtype != torch.bfloat16:
            raise RuntimeError("AWQ fused int4 kernel requires bf16 activations.")
        if self.qweight_fused.numel() == 0:
            self.rebuild_fused_params()
        x2d = x.reshape(-1, self.in_features)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        out = torch.ops.aten._weight_int4pack_mm(
            x2d,
            self.qweight_fused,
            self.group_size,
            self.q_scale_and_zeros,
        )
        if self.bias_fused.numel() != 0:
            out = out + self.bias_fused
        return out.reshape(*x.shape[:-1], self.out_features)


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


@torch.no_grad()
def collect_hessian_diag(
    model: nn.Module,
    module_names: Sequence[str],
    dataloader: DataLoader,
    device: str,
    max_batches: int,
) -> Dict[str, torch.Tensor]:
    if max_batches <= 0:
        raise ValueError("max_batches must be > 0 for calibration.")
    if not module_names:
        raise ValueError("No target modules provided for calibration.")

    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    hooks = []
    for module_name in module_names:
        module = model.get_submodule(module_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"Calibration expects nn.Linear at {module_name}, got {type(module).__name__}."
            )
        stats[module_name] = {
            "sum_sq": torch.zeros(module.in_features, dtype=torch.float64),
            "count": torch.zeros((), dtype=torch.long),
        }

        def _pre_hook(mod, inputs, name=module_name):
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"{name}: expected tensor input, got {type(x).__name__}")
            x2d = x.detach().reshape(-1, x.shape[-1]).to(dtype=torch.float32, device="cpu")
            if x2d.shape[-1] != stats[name]["sum_sq"].shape[0]:
                raise ValueError(f"{name}: input hidden dim mismatch.")
            stats[name]["sum_sq"] += (x2d * x2d).sum(dim=0, dtype=torch.float64)
            stats[name]["count"] += x2d.shape[0]

        hooks.append(module.register_forward_pre_hook(_pre_hook))

    model.eval()
    seen_batches = 0
    progress = tqdm(
        dataloader,
        total=max_batches,
        desc="[awq] calibration",
        dynamic_ncols=True,
    )
    for batch in progress:
        if seen_batches >= max_batches:
            break
        inputs, _, _ = batch
        inputs = inputs.to(device)
        try:
            model(**inputs, use_cache=False)
        except TypeError:
            model(**inputs)
        seen_batches += 1

    for h in hooks:
        h.remove()
    if seen_batches == 0:
        raise RuntimeError("Calibration dataloader produced 0 batches.")

    out: Dict[str, torch.Tensor] = {}
    for module_name, s in stats.items():
        count = int(s["count"].item())
        if count <= 0:
            raise RuntimeError(f"No calibration activations collected for {module_name}.")
        h = (s["sum_sq"] / float(count)).to(dtype=torch.float32)
        out[module_name] = h.clamp(min=1e-8)
    return out


def _awq_quantize_with_group_activation(
    linear: nn.Linear,
    h_diag: torch.Tensor,
    group_size: int,
    alpha: float = 0.5,
) -> AWQInt4LinearW4A16:
    if group_size <= 0:
        group_size = linear.in_features
    if linear.in_features % group_size != 0:
        raise ValueError(
            f"in_features ({linear.in_features}) must be divisible by group_size ({group_size})."
        )
    if h_diag.numel() != linear.in_features:
        raise ValueError(f"h_diag length mismatch: {h_diag.numel()} vs {linear.in_features}")

    q_module = AWQInt4LinearW4A16(
        in_features=linear.in_features,
        out_features=linear.out_features,
        group_size=group_size,
        has_bias=linear.bias is not None,
    )
    w = linear.weight.detach().to(torch.float32)
    groups = linear.in_features // group_size
    h = h_diag.to(dtype=torch.float32, device=w.device).reshape(groups, group_size)
    group_importance = h.mean(dim=1).clamp(min=1e-8)
    group_scale = group_importance.pow(alpha)
    group_scale = group_scale / group_scale.mean().clamp(min=1e-8)

    w_scaled = w.clone()
    for g in range(groups):
        c0 = g * group_size
        c1 = (g + 1) * group_size
        w_scaled[:, c0:c1] *= group_scale[g]

    q_int, scales, zeros = AWQInt4LinearW4A16._pseudo_quantize_tensor_int4(w_scaled, group_size)
    scales = scales / group_scale.unsqueeze(0)
    q_module.qweight = AWQInt4LinearW4A16._pack_int4(q_int).to(linear.weight.device)
    q_module.scales = scales.to(dtype=torch.float16, device=linear.weight.device)
    q_module.zeros = zeros.to(device=linear.weight.device)
    if linear.bias is not None:
        q_module.bias = linear.bias.detach().to(dtype=torch.float16, device=linear.weight.device)
    q_module.rebuild_fused_params()
    return q_module


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
    h_diag_map: Dict[str, torch.Tensor],
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
            full_name = f"model.layers.{layer_idx}.{target_name}"
            target = layer.get_submodule(target_name)
            if hasattr(target, "base_layer"):
                skipped_logs.append(f"{full_name} -> LoRA wrapper detected and skipped by policy")
                continue
            if not isinstance(target, nn.Linear):
                raise TypeError(f"{full_name}: expected nn.Linear, got {type(target).__name__}")
            if full_name not in h_diag_map:
                raise KeyError(f"Missing hessian diagonal stats for {full_name}")
            q_module = _awq_quantize_with_group_activation(
                linear=target,
                h_diag=h_diag_map[full_name],
                group_size=group_size,
                alpha=0.5,
            )
            _set_submodule(layer, target_name, q_module.to(next(layer.parameters()).device))
            quantized_modules.append(full_name)

    if not quantized_modules:
        raise RuntimeError("No decoder linear modules were quantized. Aborting.")
    return {"quantized_modules": quantized_modules, "skipped": skipped_logs}


def _prepare_awq_module_scaffold(
    model: nn.Module,
    group_size: int,
    skip_first_n: int,
    skip_last_n: int,
    target_name_groups: Tuple[Tuple[str, ...], ...] = TARGET_LINEAR_NAME_GROUPS,
) -> List[str]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Model does not expose model.layers decoder stack.")
    layers = model.model.layers
    total_layers = len(layers)
    if skip_first_n < 0 or skip_last_n < 0 or (skip_first_n + skip_last_n) >= total_layers:
        raise ValueError(
            f"Invalid skip range: total={total_layers}, skip_first={skip_first_n}, skip_last={skip_last_n}"
        )
    skip_indices = set(range(skip_first_n)) | set(range(total_layers - skip_last_n, total_layers))
    module_names: List[str] = []
    for layer_idx, layer in enumerate(layers):
        if layer_idx in skip_indices:
            continue
        layer_name = f"model.layers.{layer_idx}"
        for candidates in target_name_groups:
            target_name = _resolve_target_name(layer, candidates, layer_name)
            module_name = f"{layer_name}.{target_name}"
            target = layer.get_submodule(target_name)
            if hasattr(target, "base_layer"):
                continue
            if not isinstance(target, nn.Linear):
                raise TypeError(f"{module_name}: expected nn.Linear, got {type(target).__name__}")
            awq_module = AWQInt4LinearW4A16(
                in_features=target.in_features,
                out_features=target.out_features,
                group_size=group_size,
                has_bias=target.bias is not None,
            )
            _set_submodule(layer, target_name, awq_module.to(target.weight.device))
            module_names.append(module_name)
    if not module_names:
        raise RuntimeError("No decoder linear modules were prepared for AWQ delta load.")
    return module_names


def _collect_target_module_names(
    model: nn.Module,
    skip_first_n: int,
    skip_last_n: int,
    target_name_groups: Tuple[Tuple[str, ...], ...] = TARGET_LINEAR_NAME_GROUPS,
) -> List[str]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Model does not expose model.layers decoder stack.")
    layers = model.model.layers
    total_layers = len(layers)
    if skip_first_n < 0 or skip_last_n < 0 or (skip_first_n + skip_last_n) >= total_layers:
        raise ValueError(
            f"Invalid skip range: total={total_layers}, skip_first={skip_first_n}, skip_last={skip_last_n}"
        )
    skip_indices = set(range(skip_first_n)) | set(range(total_layers - skip_last_n, total_layers))
    names: List[str] = []
    for layer_idx, layer in enumerate(layers):
        if layer_idx in skip_indices:
            continue
        layer_name = f"model.layers.{layer_idx}"
        for candidates in target_name_groups:
            target_name = _resolve_target_name(layer, candidates, layer_name)
            full_name = f"{layer_name}.{target_name}"
            target = layer.get_submodule(target_name)
            if hasattr(target, "base_layer"):
                continue
            if not isinstance(target, nn.Linear):
                raise TypeError(f"{full_name}: expected nn.Linear, got {type(target).__name__}")
            names.append(full_name)
    if not names:
        raise RuntimeError("No target decoder linear modules found for calibration.")
    return names


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
    _prepare_awq_module_scaffold(
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
        module.rebuild_fused_params()


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
    parser.add_argument("--calib_data", type=str, default="data/DriveLM_nuScenes/split_448/val")
    parser.add_argument("--calib_samples", type=int, default=256)
    parser.add_argument("--calib_batch_size", type=int, default=1)
    parser.add_argument("--calib_num_workers", type=int, default=4)
    parser.add_argument(
        "--calib_tag_ref",
        type=str,
        default="data/DriveLM_nuScenes/refs/val_cot.json",
        help="QA tag reference JSON used to build balanced calibration subset.",
    )
    parser.add_argument(
        "--calib_coord_tag",
        type=int,
        default=3,
        help="Coordinate-sensitive tag id to boost during calibration sampling.",
    )
    parser.add_argument(
        "--calib_coord_ratio",
        type=float,
        default=0.4,
        help="Target ratio for coordinate-sensitive tag in calibration subset.",
    )
    parser.add_argument("--calib_seed", type=int, default=42)
    parser.add_argument(
        "--collate_fn",
        type=str,
        default="drivelm_nus_phi4_collate_fn_val",
        help="Collate function used to build calibration batches.",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default=None,
        help="Tokenizer/processor dir for tokenizer overlay. Defaults to --model if tokenizer exists.",
    )
    parser.add_argument(
        "--processor_base",
        type=str,
        default="/root/autodl-tmp/phi-4-multimodal-finetuned/",
        help="Base processor path used to initialize multimodal processor.",
    )
    parser.add_argument("--use_optical_flow", action="store_true")
    parser.add_argument("--flow_root", type=str, default="")
    parser.add_argument("--flow_scale_u", type=float, default=8.778)
    parser.add_argument("--flow_scale_v", type=float, default=2.888)
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
    if args.calib_samples <= 0:
        raise ValueError("--calib_samples must be > 0.")
    if args.calib_batch_size <= 0:
        raise ValueError("--calib_batch_size must be > 0.")

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

    if not os.path.isdir(args.processor_base):
        raise FileNotFoundError(f"processor_base directory not found: {args.processor_base}")
    if args.processor:
        tokenizer_src = args.processor
    elif os.path.isfile(os.path.join(args.model, "tokenizer.json")) or os.path.isfile(
        os.path.join(args.model, "added_tokens.json")
    ):
        tokenizer_src = args.model
    else:
        tokenizer_src = args.processor_base
    processor = AutoProcessor.from_pretrained(args.processor_base, trust_remote_code=True)
    if tokenizer_src != args.processor_base:
        from transformers import AutoTokenizer
        processor.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)

    collate_fn = build_collate_fn(args.collate_fn)
    calib_collate = partial(
        collate_fn,
        processor=processor,
        dtype=torch.bfloat16,
        use_optical_flow=args.use_optical_flow,
        flow_root=args.flow_root or "",
        flow_scale_u=args.flow_scale_u,
        flow_scale_v=args.flow_scale_v,
    )
    calib_dataset = load_from_disk(args.calib_data)
    n = min(args.calib_samples, len(calib_dataset))
    calib_indices, calib_dist = select_balanced_calibration_indices(
        dataset=calib_dataset,
        tag_ref_json=args.calib_tag_ref,
        calib_samples=n,
        coord_tag=args.calib_coord_tag,
        coord_ratio=args.calib_coord_ratio,
        seed=args.calib_seed,
    )
    calib_dataset = Subset(calib_dataset, calib_indices)
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=args.calib_batch_size,
        collate_fn=calib_collate,
        num_workers=args.calib_num_workers,
        shuffle=False,
    )
    max_batches = int(math.ceil(n / args.calib_batch_size))
    target_modules = _collect_target_module_names(
        model=model,
        skip_first_n=args.skip_first_n,
        skip_last_n=args.skip_last_n,
        target_name_groups=TARGET_LINEAR_NAME_GROUPS,
    )
    print(
        f"[awq] collecting calibration stats on {n} samples, "
        f"tag_dist={calib_dist}, target_modules={len(target_modules)}"
    )
    h_diag_map = collect_hessian_diag(
        model=model,
        module_names=target_modules,
        dataloader=calib_loader,
        device=args.device,
        max_batches=max_batches,
    )

    print("[awq] applying int4 quantization on decoder target linear layers ...")
    result = apply_decoder_awq_int4(
        model=model,
        h_diag_map=h_diag_map,
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
            "algorithm": "awq_style_activation_aware_int4_weight_only",
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
            "calibration": {
                "calib_data": args.calib_data,
                "calib_samples": n,
                "calib_batch_size": args.calib_batch_size,
                "calib_tag_ref": args.calib_tag_ref,
                "calib_coord_tag": args.calib_coord_tag,
                "calib_coord_ratio": args.calib_coord_ratio,
                "calib_seed": args.calib_seed,
                "selected_tag_distribution": calib_dist,
            },
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
                "This package is decoder-weight-only int4 AWQ style with calibration; activations stay fp16.",
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
