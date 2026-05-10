import torch
from typing import Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field


def _default_dataset_mix() -> List[Tuple[str, float]]:
    return [
        ("data/DriveLM_nuScenes/split_448/train",        0.6),
        ("data/nus_detection_qa/split/train",            0.4),
    ]


@dataclass
class DriveLMNusPhi4LocTokensConfig:
    # Weights checkpoint (e.g. previous finetune ``final_model``). Processor files are
    # often absent there, so ``processor_model_name`` should point to a directory with
    # tokenizer/processor files (or a previous ``--add_loc_tokens`` run that already saved
    # the expanded tokenizer alongside its weights).
    model_name: str = "/data/ckpt/pretrained/phi4/LOC-2026-05-08_23-22/epoch-3"
    # Load processor (image + audio + tokenizer) from the BASE Phi-4 path because
    # the fine-tuned checkpoint serialized an older preprocessor_config.json whose
    # audio fields no longer match Phi4MMAudioFeatureExtractor.__init__. The
    # tokenizer is then extended via add_loc_tokens; since the checkpoint already
    # has 200141 rows in its embed_tokens, resize_token_embeddings is a no-op and
    # the trained <loc_*> embeddings are preserved intact.
    processor_model_name: str = "/data/huggingface/Phi-4-multimodal-instruct"
    model_preparation: str = "prepare_model_and_processor_phi4"
    collate_fn_train: str = "drivelm_nus_phi4_collate_fn"
    collate_fn_val: str = None
    peft_name: Optional[str] = None

    # Mixed dataset: list of (path, weight). Weights auto-normalize to a probability
    # distribution for ``interleave_datasets``.
    dataset_names: List[Tuple[str, float]] = field(default_factory=_default_dataset_mix)
    # Legacy single-path field; kept empty so the loader uses ``dataset_names``.
    dataset_name: str = ""

    wandb_project = None
    run_name: str = "LOC-2026-05-09_18-49"
    output_dir: str = "/data/ckpt/pretrained/phi4/" + f"{run_name}"

    num_train_epochs: int = 4
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 16
    lr: float = 5e-6
    # SigLIP patch_embedding lr; None -> same as ``lr``.
    lr_patch_conv: Optional[float] = 5e-5
    # New: token-embedding lr for fast learning of the new <loc_*> rows.
    lr_embed: Optional[float] = 1e-5
    lora_r: int = 32
    warmup_steps: int = 80
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0

    seed: int = 42
    dtype = torch.bfloat16
    quantization: bool = False
    use_flash_attention: bool = True
    use_lora: bool = True

    resume_from_checkpoint: bool = True
    save_lora_adapter_when_checkpointing: bool = True

    save_steps: int = 99999
    log_steps: int = 10
    print_steps: int = 10

    # DataLoader workers per rank. With 8 GPUs the previous hardcoded 16 spawned
    # 8*16=128 worker processes; on autodl containers that hits rayon (HF
    # ``tokenizers``) thread-pool init failures (``Resource temporarily
    # unavailable``). 2-4 is plenty for image jpg loading.
    dataloader_num_workers: int = 4

    find_unused_parameters: bool = True

    # SigLIP knobs (same defaults as the LoRA baseline).
    train_siglip_encoder: bool = False
    train_siglip_patch_conv: bool = False
    train_image_projection: bool = False
    train_llm_lora: bool = True

    # New: enable <loc_k> coordinate tokens.
    add_loc_tokens: bool = True
    n_loc_tokens: int = 112
    loc_token_stride: int = 4

    # Optical flow: OFF by default for the loc-tokens experiment because the
    # nuScenes detection-QA samples reference frames whose precomputed flow .npz
    # sidecars likely don't exist under ``flow_root``. Re-enable only after you
    # generate flow for every CAM_FRONT/*.jpg in the mixed dataset.
    use_optical_flow: bool = False
    flow_root: str = "/root/autodl-tmp/flow_old"
    flow_scale_u: float = 8.778
    flow_scale_v: float = 2.888


config = DriveLMNusPhi4LocTokensConfig()
