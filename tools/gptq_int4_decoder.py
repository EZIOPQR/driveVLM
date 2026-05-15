#!/usr/bin/env python3
"""
GPTQ-style int4 weight quantization for DriveVLM Phi4 decoder.

Notes:
- Scope is decoder linear only (qkv/wo/ffn), same as AWQ path.
- Activations stay FP16 (W4A16 runtime).
- Uses Hessian-diagonal weighted quantization (calibration-based), i.e. a
  practical GPTQ-style approximation with explicit, reproducible metadata.
"""

import argparse
import json
import math
import os
import shutil
import sys
from functools import partial
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from drivevlms.build import build_collate_fn
from tools.awq_int4_decoder import (
    AWQInt4LinearW4A16,
    TARGET_LINEAR_NAME_GROUPS,
    _copy_base_runtime_files,
    _resolve_target_name,
    _set_submodule,
    merge_lora_into_base_linear,
)
from tools.quant_calib_utils import select_balanced_calibration_indices


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if not parent:
        raise ValueError(f"Output path has no parent directory: {path}")
    os.makedirs(parent, exist_ok=True)


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
    module_names: List[str] = []
    for layer_idx, layer in enumerate(layers):
        if layer_idx in skip_indices:
            continue
        layer_name = f"model.layers.{layer_idx}"
        for candidates in target_name_groups:
            target_name = _resolve_target_name(layer, candidates, layer_name)
            module_names.append(f"{layer_name}.{target_name}")
    return module_names


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
        raise ValueError("No target modules provided for GPTQ calibration.")

    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    hooks = []
    for module_name in module_names:
        module = model.get_submodule(module_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"Calibration expects nn.Linear at {module_name}, got {type(module).__name__}. "
                "If this is a LoRA-wrapped checkpoint, enable merge_lora first."
            )
        stats[module_name] = {
            "sum_sq": torch.zeros(module.in_features, dtype=torch.float64),
            "count": torch.zeros((), dtype=torch.long),
        }

        def _pre_hook(mod, inputs, name=module_name):
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"{name}: expected tensor input, got {type(x).__name__}")
            if x.shape[-1] != stats[name]["sum_sq"].shape[0]:
                raise ValueError(
                    f"{name}: input hidden dim mismatch, got {x.shape[-1]}, "
                    f"expected {stats[name]['sum_sq'].shape[0]}"
                )
            x2d = x.detach().reshape(-1, x.shape[-1]).to(dtype=torch.float32, device="cpu")
            stats[name]["sum_sq"] += (x2d * x2d).sum(dim=0, dtype=torch.float64)
            stats[name]["count"] += x2d.shape[0]

        hooks.append(module.register_forward_pre_hook(_pre_hook))

    model.eval()
    seen_batches = 0
    progress = tqdm(
        dataloader,
        total=max_batches,
        desc="[gptq] calibration",
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

    h_diag: Dict[str, torch.Tensor] = {}
    for module_name, s in stats.items():
        count = int(s["count"].item())
        if count <= 0:
            raise RuntimeError(f"No calibration activations collected for {module_name}.")
        h = (s["sum_sq"] / float(count)).to(dtype=torch.float32)
        h_diag[module_name] = h.clamp(min=1e-8)
    return h_diag


def _gptq_quantize_linear_diag(
    linear: nn.Linear,
    h_diag: torch.Tensor,
    group_size: int,
    shrink_candidates: Tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.8),
) -> AWQInt4LinearW4A16:
    in_features = linear.in_features
    out_features = linear.out_features
    if group_size <= 0:
        group_size = in_features
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})."
        )
    if h_diag.numel() != in_features:
        raise ValueError(f"h_diag length mismatch: {h_diag.numel()} vs {in_features}")

    q_module = AWQInt4LinearW4A16(
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        has_bias=linear.bias is not None,
    )

    w = linear.weight.detach().to(dtype=torch.float32, device="cpu")
    h = h_diag.to(dtype=torch.float32, device="cpu")
    h = h / h.mean().clamp(min=1e-8)

    groups = in_features // group_size
    q_int = torch.empty_like(w, dtype=torch.uint8)
    scales = torch.empty((out_features, groups), dtype=torch.float32)
    zeros = torch.full((out_features, groups), 8, dtype=torch.uint8)

    for g in range(groups):
        c0 = g * group_size
        c1 = (g + 1) * group_size
        w_g = w[:, c0:c1]  # [out, group]
        h_g = h[c0:c1].unsqueeze(0)  # [1, group]

        absmax = w_g.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        best_err = None
        best_q = None
        best_scale = None
        for shrink in shrink_candidates:
            bound = (absmax * float(shrink)).clamp(min=1e-6)
            scale = (bound / 7.0).clamp(min=1e-8)
            q_signed = torch.round(w_g / scale).clamp(-8, 7)
            deq = q_signed * scale
            err = ((w_g - deq).pow(2) * h_g).sum(dim=1)
            if best_err is None:
                best_err = err
                best_q = q_signed
                best_scale = scale
            else:
                mask = err < best_err
                best_err = torch.where(mask, err, best_err)
                best_q = torch.where(mask.unsqueeze(1), q_signed, best_q)
                best_scale = torch.where(mask.unsqueeze(1), scale, best_scale)

        q_int[:, c0:c1] = (best_q + 8).to(torch.uint8)
        scales[:, g] = best_scale.squeeze(1)

    q_module.qweight = AWQInt4LinearW4A16._pack_int4(q_int).to(linear.weight.device)
    q_module.scales = scales.to(dtype=torch.float16, device=linear.weight.device)
    q_module.zeros = zeros.to(device=linear.weight.device)
    if linear.bias is not None:
        q_module.bias = linear.bias.detach().to(dtype=torch.float16, device=linear.weight.device)
    return q_module


def apply_decoder_gptq_int4(
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
    skipped: List[str] = []

    for layer_idx, layer in enumerate(layers):
        if layer_idx in skip_indices:
            skipped.append(f"layer {layer_idx}: skipped by boundary rule")
            continue
        layer_name = f"model.layers.{layer_idx}"
        for candidates in target_name_groups:
            target_name = _resolve_target_name(layer, candidates, layer_name)
            module_name = f"{layer_name}.{target_name}"
            target = layer.get_submodule(target_name)
            if hasattr(target, "base_layer"):
                skipped.append(f"{module_name}: LoRA wrapper detected and skipped by policy")
                continue
            if not isinstance(target, nn.Linear):
                raise TypeError(
                    f"{module_name}: expected nn.Linear, got {type(target).__name__}. "
                    "Refusing silent fallback."
                )
            if module_name not in h_diag_map:
                raise KeyError(f"Missing hessian diagonal stats for {module_name}")
            q_module = _gptq_quantize_linear_diag(
                linear=target,
                h_diag=h_diag_map[module_name],
                group_size=group_size,
            )
            _set_submodule(layer, target_name, q_module.to(next(layer.parameters()).device))
            quantized_modules.append(module_name)

    if not quantized_modules:
        raise RuntimeError("No decoder linear modules were quantized. Aborting.")
    return {"quantized_modules": quantized_modules, "skipped": skipped}


def apply_gptq_delta_to_model(model: nn.Module, delta_path: str, map_location: str = "cpu") -> None:
    delta = torch.load(delta_path, map_location=map_location)
    if "meta" not in delta or "module_states" not in delta:
        raise KeyError(f"Invalid GPTQ delta format: missing required keys in {delta_path}")
    meta = delta["meta"]
    required_meta = ("group_size", "quantized_modules")
    missing = [k for k in required_meta if k not in meta]
    if missing:
        raise KeyError(f"Invalid GPTQ delta meta: missing keys {missing}")

    group_size = int(meta["group_size"])
    lora_policy = str(meta.get("lora_policy", "skip_wrapper"))
    if lora_policy.startswith("merge_adapter:"):
        adapter_name = lora_policy.split(":", 1)[1].strip()
        if not adapter_name:
            raise ValueError(f"Invalid lora_policy in delta meta: {lora_policy}")
        merged_count = merge_lora_into_base_linear(
            model=model,
            adapter_name=adapter_name,
            target_name_groups=TARGET_LINEAR_NAME_GROUPS,
        )
        if merged_count <= 0:
            raise RuntimeError(
                f"Delta expects merged LoRA adapter '{adapter_name}' but no target module was merged."
            )

    module_states: Dict[str, Dict[str, torch.Tensor]] = delta["module_states"]
    for module_name in meta["quantized_modules"]:
        if module_name not in module_states:
            raise KeyError(f"module_states missing entry for {module_name}")
        module = model.get_submodule(module_name)
        if not isinstance(module, AWQInt4LinearW4A16):
            if hasattr(module, "base_layer"):
                raise TypeError(
                    f"{module_name}: target is LoRA wrapper during load. "
                    "Merge LoRA before applying GPTQ delta."
                )
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{module_name}: expected nn.Linear, got {type(module).__name__}")
            q_module = AWQInt4LinearW4A16(
                in_features=module.in_features,
                out_features=module.out_features,
                group_size=group_size,
                has_bias=module.bias is not None,
            )
            _set_submodule(model, module_name, q_module.to(module.weight.device))
            module = model.get_submodule(module_name)
        module.load_state_dict(module_states[module_name], strict=True)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPTQ-style int4 quantization for DriveVLM decoder linear layers")
    p.add_argument("--model", type=str, default="/root/autodl-tmp/epoch-4", help="Input checkpoint directory")
    p.add_argument(
        "--output",
        type=str,
        default="/root/autodl-tmp/epoch-4-gptq/decoder_gptq_int4_delta.pt",
        help="Output delta path (.pt)",
    )
    p.add_argument("--export_dir", type=str, default="", help="Optional export folder for package")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--skip_first_n", type=int, default=1)
    p.add_argument("--skip_last_n", type=int, default=1)
    p.add_argument(
        "--merge_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge LoRA adapter into base linear before quantization.",
    )
    p.add_argument("--lora_adapter", type=str, default="vision")
    p.add_argument("--calib_data", type=str, default="data/DriveLM_nuScenes/split_448/val")
    p.add_argument("--calib_samples", type=int, default=256)
    p.add_argument("--calib_batch_size", type=int, default=1)
    p.add_argument("--calib_num_workers", type=int, default=4)
    p.add_argument(
        "--calib_tag_ref",
        type=str,
        default="data/DriveLM_nuScenes/refs/val_cot.json",
        help="QA tag reference JSON used to build balanced calibration subset.",
    )
    p.add_argument(
        "--calib_coord_tag",
        type=int,
        default=3,
        help="Coordinate-sensitive tag id to boost during calibration sampling.",
    )
    p.add_argument(
        "--calib_coord_ratio",
        type=float,
        default=0.4,
        help="Target ratio for coordinate-sensitive tag in calibration subset.",
    )
    p.add_argument("--calib_seed", type=int, default=42)
    p.add_argument("--collate_fn", type=str, default="drivelm_nus_phi4_collate_fn_val")
    p.add_argument(
        "--processor",
        type=str,
        default=None,
        help="Tokenizer/processor dir for tokenizer overlay. Defaults to --model if tokenizer exists.",
    )
    p.add_argument("--processor_base", type=str, default="/root/autodl-tmp/phi-4-multimodal-finetuned/")
    p.add_argument("--use_optical_flow", action="store_true")
    p.add_argument("--flow_root", type=str, default="")
    p.add_argument("--flow_scale_u", type=float, default=8.778)
    p.add_argument("--flow_scale_v", type=float, default=2.888)
    return p


@torch.no_grad()
def main() -> None:
    args = _build_argparser().parse_args()
    _ensure_parent_dir(args.output)
    if args.calib_samples <= 0:
        raise ValueError("--calib_samples must be > 0.")
    if args.calib_batch_size <= 0:
        raise ValueError("--calib_batch_size must be > 0.")

    print(f"[gptq] loading model from: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2",
    )
    model.to(args.device)
    model.eval()

    if args.merge_lora:
        merged_count = merge_lora_into_base_linear(
            model=model,
            adapter_name=args.lora_adapter,
            target_name_groups=TARGET_LINEAR_NAME_GROUPS,
        )
        if merged_count <= 0:
            raise RuntimeError(
                f"--merge_lora is enabled but no LoRA module matched adapter '{args.lora_adapter}'. "
                "Disable with --no-merge_lora if this checkpoint is already merged or has no LoRA."
            )
        print(f"[gptq] merged LoRA adapter '{args.lora_adapter}' into {merged_count} modules.")

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
        f"[gptq] collecting calibration stats on {n} samples, "
        f"tag_dist={calib_dist}, target_modules={len(target_modules)}"
    )
    h_diag_map = collect_hessian_diag(
        model=model,
        module_names=target_modules,
        dataloader=calib_loader,
        device=args.device,
        max_batches=max_batches,
    )

    print("[gptq] applying int4 quantization with Hessian-diagonal weighting ...")
    result = apply_decoder_gptq_int4(
        model=model,
        h_diag_map=h_diag_map,
        group_size=args.group_size,
        skip_first_n=args.skip_first_n,
        skip_last_n=args.skip_last_n,
        target_name_groups=TARGET_LINEAR_NAME_GROUPS,
    )

    module_states = {
        module_name: model.get_submodule(module_name).state_dict()
        for module_name in result["quantized_modules"]
    }
    delta_state = {
        "meta": {
            "algorithm": "gptq_style_diag_hessian_int4_weight_only",
            "activation_dtype": "float16",
            "weight_bits": 4,
            "group_size": args.group_size,
            "skip_first_n": args.skip_first_n,
            "skip_last_n": args.skip_last_n,
            "target_name_groups": [list(x) for x in TARGET_LINEAR_NAME_GROUPS],
            "quantized_modules": result["quantized_modules"],
            "model_path": args.model,
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
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": delta_state["meta"],
                "quantized_count": len(result["quantized_modules"]),
                "skipped_count": len(result["skipped"]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if args.export_dir:
        _copy_base_runtime_files(args.model, args.export_dir)
        export_delta = os.path.join(args.export_dir, "gptq_int4_delta.pt")
        shutil.copy2(args.output, export_delta)
        manifest = {
            "format": "drivevlm_gptq_int4_decoder_package_v1",
            "base_model_path": args.model,
            "gptq_delta": "gptq_int4_delta.pt",
            "meta_json": os.path.basename(meta_path),
            "notes": [
                "Load base model from base_model_path with trust_remote_code=True.",
                "Call apply_gptq_delta_to_model(model, gptq_delta_path).",
                "This package is decoder-weight-only int4 GPTQ-style (diag-Hessian) with FP16 activations.",
            ],
        }
        with open(os.path.join(args.export_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        export_meta = os.path.join(args.export_dir, os.path.basename(meta_path))
        if os.path.abspath(meta_path) != os.path.abspath(export_meta):
            shutil.copy2(meta_path, export_meta)

    print(f"[gptq] done. quantized modules: {len(result['quantized_modules'])}")
    print(f"[gptq] delta saved to: {args.output}")
    print(f"[gptq] meta  saved to: {meta_path}")
    if args.export_dir:
        print(f"[gptq] export package: {args.export_dir}")


if __name__ == "__main__":
    main()

