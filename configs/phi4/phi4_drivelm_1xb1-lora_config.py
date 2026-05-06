import torch
from typing import Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class DriveLMNusPhi4Config:
    # Weights checkpoint (e.g. finetune ``final_model``). Processor files are often absent there.
    model_name: str = "/root/autodl-tmp/pretrained/phi4/FULL-2026-05-03_02-39/final_model"
    # Full Phi-4 multimodal directory with tokenizer + processor (required when ``model_name`` is weights-only).
    processor_model_name: str = "/root/autodl-tmp/models/Phi-4-multimodal-instruct"
    model_preparation: str = "prepare_model_and_processor_phi4"
    collate_fn_train: str = "drivelm_nus_phi4_collate_fn"
    collate_fn_val: str = None
    peft_name: Optional[str] = None
    dataset_name: str = "data/DriveLM_nuScenes/split/train"
    wandb_project = None
    run_name: str = f"FULL-{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    output_dir: str = "/root/autodl-tmp/pretrained/phi4/" + f"{run_name}"

    num_train_epochs: int = 1
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 8
    lr: float = 5e-6
    # SigLIP patch_embedding lr; None -> same as ``lr``.
    lr_patch_conv: Optional[float] = 5e-5
    lora_r: int = 32
    warmup_steps: int = 150
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0

    seed: int = 42
    dtype = torch.bfloat16
    quantization: bool = False
    use_flash_attention: bool = True
    use_lora: bool = True

    resume_from_checkpoint: bool = False
    save_lora_adapter_when_checkpointing: bool = True

    save_steps: int = 99999
    log_steps: int = 10  # log to wandb & tensorboard, gathered loss (slower)
    print_steps: int = 10  # local print, loss on GPU0

    find_unused_parameters: bool = True

    # SigLIP: freeze vision encoder/patch/projection by default; train only LLM LoRA.
    train_siglip_encoder: bool = False
    train_siglip_patch_conv: bool = False
    train_image_projection: bool = False
    train_llm_lora: bool = True

    # Sweep-based optical flow as extra images:
    # - False: 6 RGB images
    # - True: 6 RGB + 6 flow images (flow kept at 14x14, not channel-expanded)
    use_optical_flow: bool = True
    flow_root: str = "/root/autodl-tmp/flow"
    flow_scale: float = 14.0

config = DriveLMNusPhi4Config()