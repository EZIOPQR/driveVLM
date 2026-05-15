#!/usr/bin/env python3
"""
Batched inference for DriveLM using exported GPTQ package.
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
from functools import partial
from typing import Tuple

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, GenerationConfig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from drivevlms.build import build_collate_fn
from tools.gptq_int4_decoder import apply_gptq_delta_to_model
from tools.inference_awq_batch import StageProfiler, _gpu_name, _postprocess_generated


def _load_gptq_manifest(package_dir: str) -> Tuple[str, str]:
    manifest_path = os.path.join(package_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for key in ("base_model_path", "gptq_delta"):
        if key not in manifest:
            raise KeyError(f"manifest missing required field: {key}")

    base_model_path = manifest["base_model_path"]
    if not os.path.isdir(base_model_path):
        raise FileNotFoundError(f"base_model_path directory not found: {base_model_path}")

    delta_rel_or_abs = manifest["gptq_delta"]
    delta_path = (
        delta_rel_or_abs if os.path.isabs(delta_rel_or_abs) else os.path.join(package_dir, delta_rel_or_abs)
    )
    if not os.path.exists(delta_path):
        raise FileNotFoundError(f"gptq delta not found: {delta_path}")
    return base_model_path, delta_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DriveLM GPTQ batched inference")
    p.add_argument("--gptq_package_dir", type=str, required=True, help="Exported GPTQ package dir")
    p.add_argument("--data", type=str, default="data/DriveLM_nuScenes/split_448/val")
    p.add_argument("--collate_fn", type=str, default="drivelm_nus_phi4_collate_fn_val")
    p.add_argument("--output", type=str, default="data/DriveLM_nuScenes/refs/infer_results_l10_gptq.json")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--processor_base", type=str, default="/root/autodl-tmp/phi-4-multimodal-finetuned/")
    p.add_argument("--processor", type=str, default=None)
    p.add_argument("--use_optical_flow", action="store_true")
    p.add_argument("--flow_root", type=str, default="")
    p.add_argument("--flow_scale_u", type=float, default=8.778)
    p.add_argument("--flow_scale_v", type=float, default=2.888)
    p.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--profile_output", type=str, default="")
    return p


@torch.no_grad()
def main() -> None:
    args = _build_parser().parse_args()
    run_start = time.perf_counter()
    run_start_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    base_model_path, delta_path = _load_gptq_manifest(args.gptq_package_dir)

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

    processor = AutoProcessor.from_pretrained(args.processor_base, trust_remote_code=True)
    if tokenizer_src != args.processor_base:
        processor.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    apply_gptq_delta_to_model(model, delta_path, map_location="cpu")
    model.to(args.device).eval()
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

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=val_collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
    )

    outputs = []
    total_input_tokens = 0
    total_generated_tokens = 0
    for batch in tqdm(dataloader):
        inputs, questions, ids = batch
        inputs = inputs.to(args.device)
        input_len = inputs["input_ids"].shape[-1]
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, generation_config=generation_config)
        generated = generated[:, input_len:]
        total_input_tokens += int(input_len * generated.shape[0])
        total_generated_tokens += int(generated.shape[0] * generated.shape[1])
        answers = [_postprocess_generated(a) for a in processor.batch_decode(generated, skip_special_tokens=False)]
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
                "script": "tools/inference_gptq_batch.py",
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
        print(f"[inference_gptq] profile saved to: {profile_path}")

    print(f"[inference_gptq] done. output saved to: {args.output}")


if __name__ == "__main__":
    main()

