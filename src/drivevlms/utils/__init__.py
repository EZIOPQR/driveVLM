import importlib.util
import json
import os

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


def _phi4_patch_embedding_params(model: torch.nn.Module):
    """SigLIP patch conv on Phi-4 MM; return [] if absent (e.g. PaliGemma)."""
    try:
        m = model
        if hasattr(model, "module"):
            m = model.module
        patch = m.model.embed_tokens_extend.image_embed.img_processor.embeddings.patch_embedding
        return list(patch.parameters())
    except Exception:
        return []


def _adamw_param_groups(config, model: torch.nn.Module):
    """Trainable params: group 0 = patch conv, group 1 = everything else (when split makes sense)."""
    lr = config.lr
    lr_patch = getattr(config, "lr_patch_conv", None)
    if lr_patch is None:
        lr_patch = lr
    wd = config.weight_decay

    patch_list = _phi4_patch_embedding_params(model)
    patch_ids = {id(p) for p in patch_list}

    patch_trainable = [p for p in patch_list if p.requires_grad]
    other_trainable = [
        p for p in model.parameters() if p.requires_grad and id(p) not in patch_ids
    ]

    groups = []
    if patch_trainable:
        groups.append({"params": patch_trainable, "lr": lr_patch, "weight_decay": wd})
    if other_trainable:
        groups.append({"params": other_trainable, "lr": lr, "weight_decay": wd})

    if not groups:
        raise ValueError("No trainable parameters for optimizer (check requires_grad).")
    return groups


def prepare_training_dataloader(config, collate_fn):
    mixed_dataset = load_from_disk(config.dataset_name)
    train_dataloader = DataLoader(
        mixed_dataset,
        batch_size=config.batch_size_per_gpu,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    return train_dataloader


def prepare_optimizer_and_scheduler(config, model, num_training_steps):
    optimizer = torch.optim.AdamW(_adamw_param_groups(config, model))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, lr_scheduler


def save_checkpoint(accelerator, model, epoch, step, config, loss, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = f"{config.output_dir}/checkpoint-{step}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if config.use_lora and config.save_lora_adapter_when_checkpointing:
        save_lora_adapter(accelerator, model, checkpoint_dir)

    training_info = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "latest_checkpoint": checkpoint_dir,
    }
    with open(f"{checkpoint_dir}/training_info.json", "w") as f:
        json.dump(training_info, f)
    with open(f"{config.output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f)

    accelerator.save_state(checkpoint_dir, safe_serialization=False)


def load_checkpoint(accelerator, checkpoint_dir):
    accelerator.load_state(checkpoint_dir)


def save_lora_adapter(accelerator, model, path):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )


def write_log_to_json(log_data, step, file_path=None):
    log_entry = {"step": step, **log_data}
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def load_dataclass_config(config_path):
    """动态加载 Python 配置文件，并返回 dataclass 配置对象"""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    if not hasattr(config_module, "config"):
        raise ValueError(f"Config file {config_path} must define a `config` instance")

    return config_module.config
