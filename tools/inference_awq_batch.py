#!/usr/bin/env python3
"""
Batched inference for DriveLM using exported AWQ package.

Expected AWQ package layout:
- manifest.json
- awq_int4_delta.pt (or manifest["awq_delta"])
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from functools import partial
from typing import Any, Dict, Tuple

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, GenerationConfig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from drivevlms.build import build_collate_fn
from tools.awq_int4_decoder import apply_awq_delta_to_model


_LOC_RE = re.compile(r"<loc_(\d+)>")
_CTRL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")


def _postprocess_generated(text: str, stride: int = 4) -> str:
    text = _LOC_RE.sub(lambda m: f"{int(m.group(1)) * stride:.2f}", text)
    text = _CTRL_TOKEN_RE.sub("", text)
    return text.strip()


def _gpu_name(device: str, cuda_enabled: bool) -> str:
    if not cuda_enabled:
        return ""
    if ":" in str(device):
        idx = int(str(device).split(":")[1])
    else:
        idx = torch.cuda.current_device()
    return torch.cuda.get_device_name(idx)


class StageProfiler:
    """Collect latency and CUDA memory stats for vision/prefill/decode."""

    def __init__(self, model: torch.nn.Module, device: str) -> None:
        self.device = device
        self.cuda_enabled = torch.cuda.is_available() and str(device).startswith("cuda")
        self._hooks = []
        self._active: Dict[str, Dict[str, Any]] = {}
        self._backbone_call_count = 0
        self._last_vision_ms = 0.0
        self._last_backbone_mem: Tuple[int, int, int] = (0, 0, 0)
        self.stats: Dict[str, Dict[str, float]] = {
            "vision_tower": self._empty_stage(),
            "prefill": self._empty_stage(),
            "decode": self._empty_stage(),
        }

        vision_tower = model.model.embed_tokens_extend.image_embed
        backbone = model.model
        self._hooks.append(vision_tower.register_forward_pre_hook(self._vision_pre))
        self._hooks.append(vision_tower.register_forward_hook(self._vision_post))
        self._hooks.append(backbone.register_forward_pre_hook(self._backbone_pre))
        self._hooks.append(backbone.register_forward_hook(self._backbone_post))

    @staticmethod
    def _empty_stage() -> Dict[str, float]:
        return {
            "total_ms": 0.0,
            "calls": 0.0,
            "max_allocated_bytes": 0.0,
            "max_reserved_bytes": 0.0,
            "max_active_delta_allocated_bytes": 0.0,
        }

    def _cuda_sync(self) -> None:
        if self.cuda_enabled:
            torch.cuda.synchronize(self.device)

    def _cuda_mem(self) -> Tuple[int, int]:
        if not self.cuda_enabled:
            return 0, 0
        return torch.cuda.memory_allocated(self.device), torch.cuda.memory_reserved(self.device)

    def _stage_start(self, key: str) -> None:
        self._cuda_sync()
        if self.cuda_enabled:
            torch.cuda.reset_peak_memory_stats(self.device)
        alloc_before, reserved_before = self._cuda_mem()
        self._active[key] = {
            "start": time.perf_counter(),
            "alloc_before": alloc_before,
            "reserved_before": reserved_before,
        }

    def _stage_end(self, key: str) -> Tuple[float, int, int, int]:
        item = self._active.pop(key, None)
        if item is None:
            return 0.0, 0, 0, 0
        self._cuda_sync()
        elapsed_ms = (time.perf_counter() - item["start"]) * 1000.0
        alloc_after, reserved_after = self._cuda_mem()
        if self.cuda_enabled:
            peak_alloc = torch.cuda.max_memory_allocated(self.device)
        else:
            peak_alloc = 0
        return elapsed_ms, peak_alloc, reserved_after, max(0, alloc_after - item["alloc_before"])

    def _record_stage(self, stage_name: str, elapsed_ms: float, peak_alloc: int, reserved: int, delta_alloc: int) -> None:
        stage = self.stats[stage_name]
        stage["total_ms"] += max(0.0, elapsed_ms)
        stage["calls"] += 1.0
        stage["max_allocated_bytes"] = max(stage["max_allocated_bytes"], float(peak_alloc))
        stage["max_reserved_bytes"] = max(stage["max_reserved_bytes"], float(reserved))
        stage["max_active_delta_allocated_bytes"] = max(
            stage["max_active_delta_allocated_bytes"],
            float(delta_alloc),
        )

    def _vision_pre(self, module, inputs) -> None:
        self._stage_start("vision")

    def _vision_post(self, module, inputs, output) -> None:
        elapsed_ms, peak_alloc, reserved, delta_alloc = self._stage_end("vision")
        self._last_vision_ms = elapsed_ms
        self._record_stage("vision_tower", elapsed_ms, peak_alloc, reserved, delta_alloc)

    def _backbone_pre(self, module, inputs) -> None:
        self._stage_start("backbone")

    def _backbone_post(self, module, inputs, output) -> None:
        elapsed_ms, peak_alloc, reserved, delta_alloc = self._stage_end("backbone")
        self._last_backbone_mem = (peak_alloc, reserved, delta_alloc)
        if self._backbone_call_count == 0:
            self._record_stage(
                "prefill",
                max(0.0, elapsed_ms - self._last_vision_ms),
                peak_alloc,
                reserved,
                delta_alloc,
            )
        else:
            self._record_stage("decode", elapsed_ms, peak_alloc, reserved, delta_alloc)
        self._backbone_call_count += 1

    def finalize_sample(self) -> None:
        if self._backbone_call_count == 1:
            peak_alloc, reserved, delta_alloc = self._last_backbone_mem
            self._record_stage("decode", 0.0, peak_alloc, reserved, delta_alloc)
        self._backbone_call_count = 0
        self._last_vision_ms = 0.0
        self._last_backbone_mem = (0, 0, 0)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for name, stage in self.stats.items():
            calls = max(1.0, stage["calls"])
            out[name] = {
                "total_ms": stage["total_ms"],
                "calls": int(stage["calls"]),
                "avg_ms": stage["total_ms"] / calls,
                "max_allocated_bytes": int(stage["max_allocated_bytes"]),
                "max_reserved_bytes": int(stage["max_reserved_bytes"]),
                "max_active_delta_allocated_bytes": int(stage["max_active_delta_allocated_bytes"]),
            }
        return out

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _load_awq_manifest(package_dir: str) -> Tuple[str, str]:
    manifest_path = os.path.join(package_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for key in ("base_model_path", "awq_delta"):
        if key not in manifest:
            raise KeyError(f"manifest missing required field: {key}")

    base_model_path = manifest["base_model_path"]
    if not os.path.isdir(base_model_path):
        raise FileNotFoundError(f"base_model_path directory not found: {base_model_path}")

    delta_rel_or_abs = manifest["awq_delta"]
    delta_path = (
        delta_rel_or_abs
        if os.path.isabs(delta_rel_or_abs)
        else os.path.join(package_dir, delta_rel_or_abs)
    )
    if not os.path.exists(delta_path):
        raise FileNotFoundError(f"awq delta not found: {delta_path}")
    return base_model_path, delta_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DriveLM AWQ batched inference")
    p.add_argument("--awq_package_dir", type=str, required=True, help="Exported AWQ package dir")
    p.add_argument("--data", type=str, default="data/DriveLM_nuScenes/split_448/val")
    p.add_argument("--collate_fn", type=str, default="drivelm_nus_phi4_collate_fn_val")
    p.add_argument("--output", type=str, default="data/DriveLM_nuScenes/refs/infer_results_l10_awq.json")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--processor_base",
        type=str,
        default="/root/autodl-tmp/phi-4-multimodal-finetuned/",
        help="Base processor path used to initialize multimodal processor.",
    )
    p.add_argument(
        "--processor",
        type=str,
        default=None,
        help="Optional tokenizer/processor dir override for tokenizer overlay.",
    )
    p.add_argument("--use_optical_flow", action="store_true")
    p.add_argument("--flow_root", type=str, default="")
    p.add_argument("--flow_scale_u", type=float, default=8.778)
    p.add_argument("--flow_scale_v", type=float, default=2.888)
    p.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--profile_output",
        type=str,
        default="",
        help="Output JSON path for inference profile. Default: <output_basename>.profile.json",
    )
    return p


@torch.no_grad()
def main() -> None:
    args = _build_parser().parse_args()
    run_start = time.perf_counter()
    run_start_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    base_model_path, delta_path = _load_awq_manifest(args.awq_package_dir)

    if not os.path.isdir(args.processor_base):
        raise FileNotFoundError(f"processor_base directory not found: {args.processor_base}")

    if args.processor:
        tokenizer_src = args.processor
    elif os.path.isfile(os.path.join(base_model_path, "tokenizer.json")) or os.path.isfile(
        os.path.join(base_model_path, "added_tokens.json")
    ):
        tokenizer_src = base_model_path
    else:
        tokenizer_src = args.processor_base

    print(f"[inference_awq] processor base = {args.processor_base}")
    print(f"[inference_awq] tokenizer src  = {tokenizer_src}")
    print(f"[inference_awq] base model     = {base_model_path}")
    print(f"[inference_awq] awq delta      = {delta_path}")

    processor = AutoProcessor.from_pretrained(args.processor_base, trust_remote_code=True)
    if tokenizer_src != args.processor_base:
        processor.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
        print(f"[inference_awq] tokenizer vocab size = {len(processor.tokenizer)}")
        print(f"[inference_awq] tokenize('<loc_30>') -> {processor.tokenizer.tokenize('<loc_30>')}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    apply_awq_delta_to_model(model, delta_path, map_location="cpu")
    model.to(args.device)
    model.eval()
    generation_config = GenerationConfig.from_pretrained(args.processor_base)
    profiler = StageProfiler(model, device=args.device) if args.profile else None

    collate_fn = build_collate_fn(args.collate_fn)
    val_collate_fn = partial(
        collate_fn,
        processor=processor,
        dtype=torch.bfloat16,
        use_optical_flow=args.use_optical_flow,
        flow_root=args.flow_root or "",
        flow_scale_u=args.flow_scale_u,
        flow_scale_v=args.flow_scale_v,
    )

    dataset = load_from_disk(args.data)
    total = len(dataset)
    if args.limit is not None and args.limit > 0:
        n = min(args.limit, total)
        dataset = Subset(dataset, list(range(n)))
        print(f"[inference_awq] --limit={args.limit} -> running on first {n}/{total} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=val_collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print(
        f"[inference_awq] batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens}, "
        f"num_batches={len(dataloader)}"
    )

    outputs = []
    total_input_tokens = 0
    total_generated_tokens = 0
    for batch in tqdm(dataloader):
        inputs, questions, ids = batch
        inputs = inputs.to(args.device)
        input_len = inputs["input_ids"].shape[-1]
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            generation_config=generation_config,
        )
        generated = generated[:, input_len:]
        total_input_tokens += int(input_len * generated.shape[0])
        total_generated_tokens += int(generated.shape[0] * generated.shape[1])
        answers = processor.batch_decode(generated, skip_special_tokens=False)
        answers = [_postprocess_generated(a) for a in answers]
        for sid, q, a in zip(ids, questions, answers):
            outputs.append({"id": sid, "question": q, "answer": a})
        if profiler is not None:
            profiler.finalize_sample()

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)

    run_end = time.perf_counter()
    run_end_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    if profiler is not None:
        profile_path = args.profile_output or (os.path.splitext(args.output)[0] + ".profile.json")
        profile = {
            "timestamps": {
                "start_utc": run_start_iso,
                "end_utc": run_end_iso,
                "total_seconds": run_end - run_start,
            },
            "config": {
                "script": "tools/inference_awq_batch.py",
                "args": vars(args),
                "resolved": {
                    "base_model_path": base_model_path,
                    "delta_path": delta_path,
                    "tokenizer_src": tokenizer_src,
                    "generation_config_src": args.processor_base,
                },
            },
            "runtime": {
                "device": args.device,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": _gpu_name(args.device, profiler.cuda_enabled),
            },
            "dataset": {
                "path": args.data,
                "total_samples": total,
                "evaluated_samples": len(outputs),
                "num_batches": len(dataloader),
            },
            "tokens": {
                "total_input_tokens": total_input_tokens,
                "total_generated_tokens": total_generated_tokens,
            },
            "stages": profiler.summary(),
        }
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        profiler.close()
        print(f"[inference_awq] profile saved to: {profile_path}")

    print(f"[inference_awq] done. output saved to: {args.output}")


if __name__ == "__main__":
    main()

