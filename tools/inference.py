import time
import json
import argparse
import re
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from drivevlms.build import build_collate_fn


_LOC_RE = re.compile(r"<loc_(\d+)>")
_CTRL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")


def _postprocess_generated(text: str, stride: int = 4) -> str:
    """Convert ``<loc_k>`` back to numeric pixel values and strip Phi-4 control tokens."""
    text = _LOC_RE.sub(lambda m: f"{int(m.group(1)) * stride:.2f}", text)
    text = _CTRL_TOKEN_RE.sub("", text)
    return text.strip()


class LatencyProfiler:
    """Measures vision tower, LLM prefill, and LLM decode latencies via forward hooks."""

    def __init__(self, model, device="cuda"):
        self.device = device
        self._hooks = []
        self.reset()

        vision_tower = model.model.embed_tokens_extend.image_embed
        backbone = model.model

        self._hooks.append(vision_tower.register_forward_pre_hook(self._vision_pre))
        self._hooks.append(vision_tower.register_forward_hook(self._vision_post))
        self._hooks.append(backbone.register_forward_pre_hook(self._backbone_pre))
        self._hooks.append(backbone.register_forward_hook(self._backbone_post))

    def reset(self):
        self.vision_ms = 0.0
        self.prefill_ms = 0.0
        self.decode_ms = 0.0
        self._backbone_call_count = 0
        self._vision_start = 0.0
        self._backbone_start = 0.0

    def _vision_pre(self, module, input):
        torch.cuda.synchronize(self.device)
        self._vision_start = time.perf_counter()

    def _vision_post(self, module, input, output):
        torch.cuda.synchronize(self.device)
        self.vision_ms += (time.perf_counter() - self._vision_start) * 1000

    def _backbone_pre(self, module, input):
        torch.cuda.synchronize(self.device)
        self._backbone_start = time.perf_counter()

    def _backbone_post(self, module, input, output):
        torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - self._backbone_start) * 1000
        if self._backbone_call_count == 0:
            self.prefill_ms = elapsed_ms - self.vision_ms
        else:
            self.decode_ms += elapsed_ms
        self._backbone_call_count += 1

    @property
    def num_decode_steps(self):
        return max(0, self._backbone_call_count - 1)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

@torch.no_grad()
def main(args):

    # Load model and processor
    base_model = "/root/autodl-tmp/models/Phi-4-multimodal-instruct"
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True
    )
    model.to(args.device)
    model.eval()
    generation_config = GenerationConfig.from_pretrained(base_model)

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
            max_new_tokens=1000,
            generation_config=generation_config
        )
        output = output[:, input_len:]
        results = processor.batch_decode(output, skip_special_tokens=False)
        results = [_postprocess_generated(r) for r in results]
        return results, input_len, output.shape[-1]

    def flatten(x):
        return x[0] if isinstance(x, list) else x

    warmup_steps = args.warmup if profiler else 0
    all_vision_ms = []
    all_prefill_ms_per_token = []
    all_decode_ms_per_token = []

    data_dict = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            inputs, question, ids = batch
            if profiler:
                profiler.reset()
            results, input_len, num_tokens = infer(inputs.to(args.device))
            if profiler and step >= warmup_steps:
                all_vision_ms.append(profiler.vision_ms)
                all_prefill_ms_per_token.append(profiler.prefill_ms / max(input_len, 1))
                all_decode_ms_per_token.append(profiler.decode_ms / max(num_tokens, 1))

            data_dict.append(
                {'id': flatten(ids), 'question': flatten(question), 'answer': flatten(results)}
            )

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=4)

    if profiler:
        profiler.remove_hooks()
        vision_arr = np.array(all_vision_ms)
        prefill_arr = np.array(all_prefill_ms_per_token)
        decode_arr = np.array(all_decode_ms_per_token)

        print("\n" + "=" * 60)
        print("Latency Profiling Summary")
        print("=" * 60)
        print(f"  Samples: {len(all_vision_ms)} (warmup skipped: {warmup_steps})")
        print(f"  Vision Tower : {vision_arr.mean():.4f} ± {vision_arr.std():.4f} ms")
        print(f"  LLM Prefill  : {prefill_arr.mean():.4f} ± {prefill_arr.std():.4f} ms/token")
        print(f"  LLM Decode   : {decode_arr.mean():.4f} ± {decode_arr.std():.4f} ms/token")
        print("=" * 60)

def parse_args():
    parser = argparse.ArgumentParser(description='DriveLM Inference')
    parser.add_argument("--data", type=str, default="data/DriveLM_nuScenes/split/val")
    parser.add_argument("--collate_fn", type=str, default="drivelm_nus_phi4_collate_fn_val")
    parser.add_argument("--output", type=str, default="data/DriveLM_nuScenes/refs/infer_results_21-49.json")
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/models/Phi-4-multimodal-instruct", help="Path to model or checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to run inference")
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
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup samples to skip in latency statistics")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())