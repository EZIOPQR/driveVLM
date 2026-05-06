import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, GenerationConfig
from peft import PeftModel
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from datasets import load_from_disk
from tqdm import tqdm
from functools import partial
import argparse
from drivevlms.build import build_collate_fn
import json

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
    generation_config = GenerationConfig.from_pretrained(base_model)

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
        results = processor.batch_decode(output, skip_special_tokens=True)
        return results

    def flatten(x):
        return x[0] if isinstance(x, list) else x
    
    data_dict = []
    with torch.no_grad():
        cnt = 0
        for batch in tqdm(dataloader):
            cnt += 1
            inputs, question, ids = batch
            results = infer(inputs.to(args.device))
            data_dict.append(
                {'id': flatten(ids), 'question': flatten(question), 'answer': flatten(results)}
            )

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=4)

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
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())