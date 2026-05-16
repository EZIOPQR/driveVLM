import time
import datetime as dt
import json
import argparse
import re
import os
from functools import partial
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, GenerationConfig

from drivevlms.build import build_collate_fn


_LOC_RE = re.compile(r"<loc_(\d+)>")
_CTRL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")


def _postprocess_generated(text: str, stride: int = 4) -> str:
    """Convert ``<loc_k>`` back to numeric pixel values and strip Phi-4 control tokens."""
    text = _LOC_RE.sub(lambda m: f"{int(m.group(1)) * stride:.2f}", text)
    text = _CTRL_TOKEN_RE.sub("", text)
    return text.strip()


class LatencyProfiler:
    """Collect latency and CUDA memory stats for vision/prefill/decode."""

    def __init__(self, model, device="cuda"):
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
        alloc_before, _ = self._cuda_mem()
        self._active[key] = {"start": time.perf_counter(), "alloc_before": alloc_before}

    def _stage_end(self, key: str) -> Tuple[float, int, int, int]:
        item = self._active.pop(key, None)
        if item is None:
            return 0.0, 0, 0, 0
        self._cuda_sync()
        elapsed_ms = (time.perf_counter() - item["start"]) * 1000.0
        alloc_after, reserved_after = self._cuda_mem()
        peak_alloc = torch.cuda.max_memory_allocated(self.device) if self.cuda_enabled else 0
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

    def _vision_pre(self, module, input):
        self._stage_start("vision")

    def _vision_post(self, module, input, output):
        elapsed_ms, peak_alloc, reserved, delta_alloc = self._stage_end("vision")
        self._last_vision_ms = elapsed_ms
        self._record_stage("vision_tower", elapsed_ms, peak_alloc, reserved, delta_alloc)

    def _backbone_pre(self, module, input):
        self._stage_start("backbone")

    def _backbone_post(self, module, input, output):
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

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _gpu_name(device: str, cuda_enabled: bool) -> str:
    if not cuda_enabled:
        return ""
    if ":" in str(device):
        idx = int(str(device).split(":")[1])
    else:
        idx = torch.cuda.current_device()
    return torch.cuda.get_device_name(idx)

@torch.no_grad()
def main(args):
    run_start = time.perf_counter()
    run_start_iso = dt.datetime.now(dt.timezone.utc).isoformat()

    # Keep processor implementation from base Phi-4, then optionally overlay tokenizer
    # from finetuned checkpoint to preserve added special tokens (e.g. <loc_*>).
    base_model = "/root/autodl-tmp/phi-4-multimodal-finetuned/"
    if args.processor:
        tokenizer_src = args.processor
    elif os.path.isfile(os.path.join(args.model, "tokenizer.json")) \
            or os.path.isfile(os.path.join(args.model, "added_tokens.json")):
        tokenizer_src = args.model
    else:
        tokenizer_src = base_model
    print(f"[inference] processor base = {base_model}")
    print(f"[inference] tokenizer src  = {tokenizer_src}")

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer_src != base_model:
        processor.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_src, trust_remote_code=True
        )
        print(f"[inference] tokenizer vocab size = {len(processor.tokenizer)}")
        print(f"[inference] tokenize('<loc_30>') -> {processor.tokenizer.tokenize('<loc_30>')}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True
    )
    model.to(args.device)
    model.eval()
    generation_config = GenerationConfig.from_pretrained(base_model)
    generation_config.num_logits_to_keep = 1

    profiler = LatencyProfiler(model, device=args.device) if args.profile else None

    # prepare dataset
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
    if args.limit is not None and args.limit > 0:
        n = min(args.limit, len(dataset))
        dataset = Subset(dataset, list(range(n)))
        print(f"[inference] --limit={args.limit} -> running on first {n}/{len(dataset.dataset)} samples")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=val_collate_fn,
        num_workers=8,
        shuffle=False,
    )

    def infer(inputs):
        input_len = inputs["input_ids"].shape[-1]
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            generation_config=generation_config,
            num_logits_to_keep=1,
        )
        output = output[:, input_len:]
        results = processor.batch_decode(output, skip_special_tokens=False)
        results = [_postprocess_generated(r) for r in results]
        return results, input_len, output.shape[-1]

    def flatten(x):
        return x[0] if isinstance(x, list) else x

    data_dict = []
    total_input_tokens = 0
    total_generated_tokens = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            inputs, question, ids = batch
            results, input_len, num_tokens = infer(inputs.to(args.device))
            total_input_tokens += int(input_len)
            total_generated_tokens += int(num_tokens)
            if profiler is not None:
                profiler.finalize_sample()

            data_dict.append(
                {'id': flatten(ids), 'question': flatten(question), 'answer': flatten(results)}
            )

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=4)

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
                "script": "tools/inference.py",
                "args": vars(args),
                "resolved": {
                    "processor_base": base_model,
                    "tokenizer_src": tokenizer_src,
                    "model_path": args.model,
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
                "evaluated_samples": len(data_dict),
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
        profiler.remove_hooks()
        print(f"[inference] profile saved to: {profile_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='DriveLM Inference')
    parser.add_argument("--data", type=str, default="data/DriveLM_nuScenes/split_448/val")
    parser.add_argument("--collate_fn", type=str, default="drivelm_nus_phi4_collate_fn_val")
    parser.add_argument("--output", type=str, default="data/DriveLM_nuScenes/refs/infer_smoke.json")
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/epoch-4/", help="Path to model or checkpoint")
    parser.add_argument(
        "--processor", type=str, default=None,
        help="Path to processor/tokenizer dir. If unset, uses --model when it has tokenizer files, else base Phi-4 path.",
    )
    parser.add_argument("--device", default="cuda", help="Device to run inference")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Generation cap per sample")
    parser.add_argument("--limit", type=int, default=None, help="Only run on the first N samples (useful for smoke test). Default: run full set.")
    parser.add_argument(
        "--use_optical_flow",
        action="store_true",
        help="5-channel SigLIP; requires precomputed .npz under --flow_root.",
    )
    parser.add_argument("--flow_root", type=str, default="", help="flow/CAM/*.npz root")
    parser.add_argument("--flow_scale_u", type=float, default=8.778)
    parser.add_argument("--flow_scale_v", type=float, default=2.888)
    parser.add_argument("--profile", action="store_true", default=True,
                        help="Enable latency profiling (vision tower / LLM prefill / LLM decode)")
    parser.add_argument("--no_profile", dest="profile", action="store_false",
                        help="Disable latency profiling")
    parser.add_argument(
        "--profile_output",
        type=str,
        default="",
        help="Output JSON path for inference profile. Default: <output_basename>.profile.json",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())