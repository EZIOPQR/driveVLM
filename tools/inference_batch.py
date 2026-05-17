"""Batched inference for DriveLM (Phi-4 path).

Mirrors tools/inference.py but exposes --batch_size and --max_new_tokens, and
relies on the (now batch-capable) drivelm_nus_phi4_collate_fn_val to feed
left-padded inputs to model.generate().

Output schema is identical to tools/inference.py to keep tools/evaluation.py
working without changes:
    [{"id": str, "question": str, "answer": str}, ...]
"""
import argparse
import datetime as dt
import json
import os
import re
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from drivevlms.build import build_collate_fn


_LOC_RE = re.compile(r"<loc_(\d+)>")
_CTRL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")
_IMAGE_SPECIAL_TOKEN_ID = 200010


def _postprocess_generated(text: str, stride: int = 4) -> str:
    """Convert ``<loc_k>`` back to numeric pixel values and strip Phi-4 control tokens."""
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


def _compress_image_special_tokens(
    inputs: Dict[str, torch.Tensor],
    keep_ratio: float,
    image_token_id: int,
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    if keep_ratio >= 1.0:
        return inputs
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size = input_ids.shape[0]
    kept_ids: List[torch.Tensor] = []
    max_len = 0
    for b in range(batch_size):
        valid = attention_mask[b].bool()
        seq_ids = input_ids[b][valid]
        image_pos = torch.nonzero(seq_ids == image_token_id, as_tuple=False).squeeze(-1)
        if image_pos.numel() == 0:
            seq_keep = seq_ids
        else:
            keep_n = max(1, int(round(float(image_pos.numel()) * keep_ratio)))
            keep_mask = torch.ones_like(seq_ids, dtype=torch.bool)
            keep_mask[image_pos[keep_n:]] = False
            seq_keep = seq_ids[keep_mask]
        kept_ids.append(seq_keep)
        max_len = max(max_len, int(seq_keep.numel()))
    new_input_ids = input_ids.new_full((batch_size, max_len), pad_token_id)
    new_attention_mask = attention_mask.new_zeros((batch_size, max_len))
    for b, seq in enumerate(kept_ids):
        seq_len = int(seq.numel())
        new_input_ids[b, -seq_len:] = seq
        new_attention_mask[b, -seq_len:] = 1
    inputs["input_ids"] = new_input_ids
    inputs["attention_mask"] = new_attention_mask
    return inputs


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
        self._current_backbone_stage = "prefill"
        self.stats: Dict[str, Dict[str, float]] = {
            "vision_tower": self._empty_stage(),
            "prefill": self._empty_stage(),
            "decode": self._empty_stage(),
        }
        self._part_active: Dict[str, List[float]] = {}
        self.backbone_parts: Dict[str, Dict[str, Dict[str, float]]] = {
            "prefill": self._empty_backbone_parts(),
            "decode": self._empty_backbone_parts(),
        }

        vision_tower = model.model.embed_tokens_extend.image_embed
        backbone = model.model
        self._hooks.append(vision_tower.register_forward_pre_hook(self._vision_pre))
        self._hooks.append(vision_tower.register_forward_hook(self._vision_post))
        self._hooks.append(backbone.register_forward_pre_hook(self._backbone_pre))
        self._hooks.append(backbone.register_forward_hook(self._backbone_post))
        self._register_backbone_part_hooks(model)

    @staticmethod
    def _empty_stage() -> Dict[str, float]:
        return {
            "total_ms": 0.0,
            "calls": 0.0,
            "max_allocated_bytes": 0.0,
            "max_reserved_bytes": 0.0,
            "max_active_delta_allocated_bytes": 0.0,
        }

    @staticmethod
    def _empty_backbone_parts() -> Dict[str, Dict[str, float]]:
        return {
            "qkv": {"total_ms": 0.0, "calls": 0.0},
            "attention_total": {"total_ms": 0.0, "calls": 0.0},
            "wo": {"total_ms": 0.0, "calls": 0.0},
            "ffn_gate_up": {"total_ms": 0.0, "calls": 0.0},
            "ffn_down": {"total_ms": 0.0, "calls": 0.0},
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
        self._active[key] = {
            "start": time.perf_counter(),
            "alloc_before": alloc_before,
        }

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

    def _record_backbone_part(self, stage_name: str, part_name: str, elapsed_ms: float) -> None:
        item = self.backbone_parts[stage_name][part_name]
        item["total_ms"] += max(0.0, elapsed_ms)
        item["calls"] += 1.0

    @staticmethod
    def _resolve_submodule(parent: torch.nn.Module, candidates: List[str]) -> Optional[torch.nn.Module]:
        for name in candidates:
            if hasattr(parent, name):
                return getattr(parent, name)
        return None

    def _register_part_hook(self, module: Optional[torch.nn.Module], part_name: str) -> None:
        if module is None:
            return

        def _pre(_module, _inputs, part=part_name):
            self._cuda_sync()
            self._part_active.setdefault(part, []).append(time.perf_counter())

        def _post(_module, _inputs, _output, part=part_name):
            stack = self._part_active.get(part)
            if not stack:
                return
            start = stack.pop()
            self._cuda_sync()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._record_backbone_part(self._current_backbone_stage, part, elapsed_ms)

        self._hooks.append(module.register_forward_pre_hook(_pre))
        self._hooks.append(module.register_forward_hook(_post))

    def _register_backbone_part_hooks(self, model: torch.nn.Module) -> None:
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            return
        for layer in layers:
            attn = self._resolve_submodule(layer, ["self_attn", "attention"])
            self._register_part_hook(attn, "attention_total")
            if attn is not None:
                self._register_part_hook(self._resolve_submodule(attn, ["qkv_proj", "wqkv"]), "qkv")
                self._register_part_hook(self._resolve_submodule(attn, ["o_proj", "wo"]), "wo")
            mlp = self._resolve_submodule(layer, ["mlp", "feed_forward"])
            if mlp is not None:
                self._register_part_hook(self._resolve_submodule(mlp, ["gate_up_proj", "w1w3"]), "ffn_gate_up")
                self._register_part_hook(self._resolve_submodule(mlp, ["down_proj", "w2"]), "ffn_down")

    def _vision_pre(self, module, inputs) -> None:
        self._stage_start("vision")

    def _vision_post(self, module, inputs, output) -> None:
        elapsed_ms, peak_alloc, reserved, delta_alloc = self._stage_end("vision")
        self._last_vision_ms = elapsed_ms
        self._record_stage("vision_tower", elapsed_ms, peak_alloc, reserved, delta_alloc)

    def _backbone_pre(self, module, inputs) -> None:
        self._current_backbone_stage = "prefill" if self._backbone_call_count == 0 else "decode"
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
        out["backbone_parts"] = self._summarize_backbone_parts()
        return out

    def _summarize_backbone_parts(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        total_backbone_ms = self.stats["prefill"]["total_ms"] + self.stats["decode"]["total_ms"]
        for stage_name in ("prefill", "decode"):
            stage_total_ms = self.stats[stage_name]["total_ms"]
            part_items: Dict[str, Dict[str, float]] = {}
            for part_name, item in self.backbone_parts[stage_name].items():
                part_total = item["total_ms"]
                calls = int(item["calls"])
                part_items[part_name] = {
                    "total_ms": part_total,
                    "calls": calls,
                    "avg_ms": (part_total / max(1, calls)),
                    "share_of_stage_backbone": (part_total / stage_total_ms if stage_total_ms > 0 else 0.0),
                }
            attn_core_ms = max(
                0.0,
                part_items["attention_total"]["total_ms"] - part_items["qkv"]["total_ms"] - part_items["wo"]["total_ms"],
            )
            ffn_total_ms = part_items["ffn_gate_up"]["total_ms"] + part_items["ffn_down"]["total_ms"]
            part_items["attention_core"] = {
                "total_ms": attn_core_ms,
                "calls": part_items["attention_total"]["calls"],
                "avg_ms": attn_core_ms / max(1, part_items["attention_total"]["calls"]),
                "share_of_stage_backbone": (attn_core_ms / stage_total_ms if stage_total_ms > 0 else 0.0),
            }
            part_items["ffn_total"] = {
                "total_ms": ffn_total_ms,
                "calls": min(part_items["ffn_gate_up"]["calls"], part_items["ffn_down"]["calls"]),
                "avg_ms": ffn_total_ms / max(1, min(part_items["ffn_gate_up"]["calls"], part_items["ffn_down"]["calls"])),
                "share_of_stage_backbone": (ffn_total_ms / stage_total_ms if stage_total_ms > 0 else 0.0),
            }
            summary[stage_name] = part_items
        summary["total"] = {
            "backbone_total_ms": total_backbone_ms,
            "prefill_share": (self.stats["prefill"]["total_ms"] / total_backbone_ms if total_backbone_ms > 0 else 0.0),
            "decode_share": (self.stats["decode"]["total_ms"] / total_backbone_ms if total_backbone_ms > 0 else 0.0),
        }
        return summary

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


@torch.no_grad()
def main(args):
    run_start = time.perf_counter()
    run_start_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    # Strategy:
    # - Always load the processor (image + audio + tokenizer) from the BASE Phi-4
    #   path, because the fine-tuned checkpoint serialized an older
    #   ``preprocessor_config.json`` whose audio fields no longer match the current
    #   ``Phi4MMAudioFeatureExtractor.__init__`` signature.
    # - Then OVERLAY the tokenizer from the checkpoint (or --processor) so the
    #   added <loc_*> tokens come through.
    BASE = "/root/autodl-tmp/phi-4-multimodal-finetuned/"
    if args.processor:
        tokenizer_src = args.processor
    elif os.path.isfile(os.path.join(args.model, "tokenizer.json")) \
            or os.path.isfile(os.path.join(args.model, "added_tokens.json")):
        tokenizer_src = args.model
    else:
        tokenizer_src = BASE
    print(f"[inference_batch] processor base = {BASE}")
    print(f"[inference_batch] tokenizer src  = {tokenizer_src}")

    processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    if tokenizer_src != BASE:
        from transformers import AutoTokenizer
        processor.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_src, trust_remote_code=True
        )
        print(f"[inference_batch] tokenizer vocab size = {len(processor.tokenizer)}")
        # Sanity: confirm <loc_*> are atomic single tokens
        sample_toks = processor.tokenizer.tokenize("<loc_30>")
        print(f"[inference_batch] tokenize('<loc_30>') -> {sample_toks}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()
    image_embed = model.model.embed_tokens_extend.image_embed
    if hasattr(image_embed, "configure_token_pruning"):
        image_embed.configure_token_pruning(
            enabled=args.vision_token_prune,
            keep_ratio=args.vision_token_keep_ratio,
            layer_idx=args.vision_token_prune_layer_idx,
        )
    generation_config = GenerationConfig.from_pretrained(BASE)
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
        print(f"[inference_batch] --limit={args.limit} -> running on first {n}/{total} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=val_collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print(f"[inference_batch] batch_size={args.batch_size}, "
          f"max_new_tokens={args.max_new_tokens}, "
          f"num_batches={len(dataloader)}")

    data_dict = []
    total_input_tokens = 0
    total_generated_tokens = 0
    for batch in tqdm(dataloader):
        inputs, questions, ids = batch
        inputs = inputs.to(args.device)
        if args.vision_token_prune:
            inputs = _compress_image_special_tokens(
                inputs=inputs,
                keep_ratio=args.vision_token_keep_ratio,
                image_token_id=_IMAGE_SPECIAL_TOKEN_ID,
                pad_token_id=processor.tokenizer.pad_token_id or 0,
            )
        input_len = inputs["input_ids"].shape[-1]
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            generation_config=generation_config,
        )
        # Same input_len for every sample in batch (left padding aligns them).
        output = output[:, input_len:]
        total_input_tokens += int(input_len * output.shape[0])
        total_generated_tokens += int(output.shape[0] * output.shape[1])
        # Keep <loc_*> in the decoded text, then post-process to numeric coords.
        answers = processor.batch_decode(output, skip_special_tokens=False)
        answers = [_postprocess_generated(a) for a in answers]

        assert len(answers) == len(ids) == len(questions), (
            f"len mismatch: answers={len(answers)} ids={len(ids)} questions={len(questions)}"
        )
        for sid, q, a in zip(ids, questions, answers):
            data_dict.append({"id": sid, "question": q, "answer": a})
        if profiler is not None:
            profiler.finalize_sample()

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
                "script": "tools/inference_batch.py",
                "args": vars(args),
                "resolved": {
                    "processor_base": BASE,
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
                "total_samples": total,
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
        profiler.close()
        print(f"[inference_batch] profile saved to: {profile_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="DriveLM Batched Inference (Phi-4)")
    parser.add_argument("--data", type=str, default="data/DriveLM_nuScenes/split_448/val")
    parser.add_argument(
        "--collate_fn", type=str, default="drivelm_nus_phi4_collate_fn_val",
        help="Must point at a val collate fn that supports batch>1. "
             "drivelm_nus_phi4_collate_fn_val is now batch-capable.",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/DriveLM_nuScenes/refs/infer_results_batch.json",
    )
    parser.add_argument(
        "--model", type=str,
        default="/root/autodl-tmp/epoch-4",
        help="Path to model or fine-tuned checkpoint",
    )
    parser.add_argument(
        "--processor", type=str, default=None,
        help="Path to processor/tokenizer dir. If unset, uses --model when it has "
             "tokenizer files (recommended for loc-tokens checkpoints), else falls "
             "back to the base Phi-4 path.",
    )
    parser.add_argument("--device", default="cuda", help="Device to run inference")
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="DataLoader batch size. Phi-4 bf16 + 6*448 imgs: 4 fits ~24GB, 8 needs more.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Generation cap. 256 is plenty for DriveLM answers; lower if you want more speed.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run on the first N samples (for smoke test). Default: full set.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--use_optical_flow",
        action="store_true",
        help="5-channel SigLIP input; run compute_flow_from_sweeps.py and set --flow-root.",
    )
    parser.add_argument(
        "--flow_root",
        type=str,
        default="",
        help="Root with CAM_*/<jpg_stem>.npz (default: config flow_root when training).",
    )
    parser.add_argument(
        "--flow_scale_u",
        type=float,
        default=8.778,
        help="Normalization divisor for flow u channel.",
    )
    parser.add_argument(
        "--flow_scale_v",
        type=float,
        default=2.888,
        help="Normalization divisor for flow v channel.",
    )
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--vision_token_prune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable attention-guided token merging in vision tower and compress image placeholders.",
    )
    parser.add_argument(
        "--vision_token_keep_ratio",
        type=float,
        default=0.35,
        help="Fraction of vision tokens to keep (typically 0.2~0.5).",
    )
    parser.add_argument(
        "--vision_token_prune_layer_idx",
        type=int,
        default=-2,
        help="Vision layer index used for attention-guided token importance.",
    )
    parser.add_argument(
        "--profile_output",
        type=str,
        default="",
        help="Output JSON path for inference profile. Default: <output_basename>.profile.json",
    )
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
