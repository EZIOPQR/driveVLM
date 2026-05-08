import os
import torch
import warnings
from typing import Optional

from accelerate import PartialState
from transformers import (AutoProcessor, 
                        AutoModelForCausalLM, 
                        BitsAndBytesConfig)
from ..registry import register_prepare_model_and_processor
import copy
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer
from drivevlms.models.phi4_bjxx import Phi4MMProcessor, Phi4MMForCausalLM
from accelerate import Accelerator

_PHI4_DEFAULT_HUB_REVISION = "607bf62a754018e31fb4b55abbc7d72cce4ffee5"


def _is_local_dir(path: str) -> bool:
    return bool(path) and os.path.isdir(os.path.expanduser(path))


def _revision_for_pretrained_path(
    path: str, explicit: Optional[str], *, default_hub_revision: str
) -> dict:
    """Pass ``revision`` only when needed (local dirs omit unless ``explicit`` is set)."""
    if explicit is not None:
        return {"revision": explicit} if explicit else {}
    if _is_local_dir(path):
        return {}
    return {"revision": default_hub_revision}


def _load_phi4_processor(config):
    proc_path = getattr(config, "processor_model_name", None) or config.model_name
    kw = dict(trust_remote_code=True)
    kw.update(
        _revision_for_pretrained_path(
            proc_path,
            getattr(config, "processor_revision", None),
            default_hub_revision=_PHI4_DEFAULT_HUB_REVISION,
        )
    )
    return AutoProcessor.from_pretrained(proc_path, **kw)


def _set_llm_adapter_trainable(model: torch.nn.Module, adapter: str, trainable: bool) -> None:
    """Unfreeze PEFT LoRA weights for a named adapter (e.g. ``vision``, ``domain``) on the LLM."""
    for module in model.model.modules():
        if not isinstance(module, LoraLayer):
            continue
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None or adapter not in lora_a:
            continue
        for p in lora_a[adapter].parameters():
            p.requires_grad = trainable
        for p in lora_b[adapter].parameters():
            p.requires_grad = trainable


def _apply_siglip_patch_trainable(model: torch.nn.Module) -> None:
    """Unfreeze SigLIP ``patch_embedding`` (first Conv2d)."""
    patch = model.model.embed_tokens_extend.image_embed.img_processor.embeddings.patch_embedding
    for p in patch.parameters():
        p.requires_grad = True


def _apply_siglip_full_encoder_trainable(model: torch.nn.Module) -> None:
    """Unfreeze full SigLIP ``img_processor`` (patch + ViT blocks)."""
    img_proc = model.model.embed_tokens_extend.image_embed.img_processor
    for p in img_proc.parameters():
        p.requires_grad = True


def _apply_image_projection_trainable(model: torch.nn.Module) -> None:
    """Unfreeze ``Phi4MMImageEmbedding.img_projection`` (vision → LM dim)."""
    proj = model.model.embed_tokens_extend.image_embed.img_projection
    for p in proj.parameters():
        p.requires_grad = True


def _configure_image_embed_trainability(model: torch.nn.Module, config) -> None:
    """After freezing ``image_embed``, optionally unfreeze patch / projection / full SigLIP."""
    if getattr(config, "train_siglip_encoder", False):
        _apply_siglip_full_encoder_trainable(model)
    else:
        if getattr(config, "train_siglip_patch_conv", True):
            _apply_siglip_patch_trainable(model)
        if getattr(config, "train_image_projection", True):
            _apply_image_projection_trainable(model)


def _add_loc_tokens(model: torch.nn.Module, processor, n: int = 112) -> int:
    """Expand tokenizer + embedding/lm_head with `<loc_0>..<loc_{n-1}>`.

    Returns the number of tokens actually added. Idempotent across resumes.
    Must be called BEFORE LoRA / adapter activation, while embed/lm_head are still
    plain ``nn.Embedding`` / ``nn.Linear``.
    """
    tokenizer = processor.tokenizer
    new_tokens = [f"<loc_{i}>" for i in range(n)]
    existing = set(tokenizer.get_vocab().keys())
    to_add = [t for t in new_tokens if t not in existing]
    if not to_add:
        return 0
    tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    old_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        # Mean-init new rows from existing vocab to keep loss in a sane range.
        mean = emb[:old_size].mean(dim=0, keepdim=True)
        emb[old_size:] = mean
        # Phi4 ties lm_head.weight to embed_tokens.weight; reassert tie just in case.
        model.tie_weights()
    return len(to_add)


@register_prepare_model_and_processor
def prepare_model_and_processor_phi4(config):
    processor = _load_phi4_processor(config)
    model = None
    _rev_kw = _revision_for_pretrained_path(
        config.model_name,
        getattr(config, "model_revision", None),
        default_hub_revision=_PHI4_DEFAULT_HUB_REVISION,
    )
    if config.quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=config.dtype,
        )
        if config.use_flash_attention:
            assert (
                config.dtype == torch.bfloat16
            ), "Flash attention only supports bfloat16"
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": PartialState().local_process_index},
                trust_remote_code=True,
                **_rev_kw,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=bnb_config,
                trust_remote_code=True,
                **_rev_kw,
            )
    else:
        if config.use_flash_attention:
            assert (
                config.dtype == torch.bfloat16
            ), "Flash attention only supports bfloat16"
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
                use_cache=False,
                **_rev_kw,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                _attn_implementation="sdpa",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                use_cache=False,
                **_rev_kw,
            )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    # Optional: expand tokenizer with <loc_k> coordinate tokens. Done BEFORE LoRA so
    # we operate on the bare embed_tokens / lm_head (LoRA wrapping happens later via
    # ``model.set_lora_adapter('vision')`` which only touches LLM linear projections).
    if getattr(config, "add_loc_tokens", False):
        n_added = _add_loc_tokens(
            model, processor, n=getattr(config, "n_loc_tokens", 112)
        )
        if n_added:
            print(f"[phi4_preparation] added {n_added} <loc_*> tokens; "
                  f"new vocab size = {len(processor.tokenizer)}")

    # remove parameters irrelevant to vision tasks
    del model.model.embed_tokens_extend.audio_embed  # remove audio encoder
    for layer in model.model.layers:
        # remove audio lora
        del layer.mlp.down_proj.lora_dropout.speech
        del layer.mlp.down_proj.lora_A.speech
        del layer.mlp.down_proj.lora_B.speech
        del layer.mlp.gate_up_proj.lora_dropout.speech
        del layer.mlp.gate_up_proj.lora_A.speech
        del layer.mlp.gate_up_proj.lora_B.speech
        del layer.self_attn.o_proj.lora_dropout.speech
        del layer.self_attn.o_proj.lora_A.speech
        del layer.self_attn.o_proj.lora_B.speech
        del layer.self_attn.qkv_proj.lora_dropout.speech
        del layer.self_attn.qkv_proj.lora_A.speech
        del layer.self_attn.qkv_proj.lora_B.speech


    # Multimodal ``vision`` adapter = LoRA on the LLM (see model config ``vision_lora``).
    # SigLIP: by default train patch conv + img_projection only (not ViT encoder).
    model.set_lora_adapter('vision')
    image_embed = model.model.embed_tokens_extend.image_embed
    for param in image_embed.parameters():
        param.requires_grad = False

    _configure_image_embed_trainability(model, config)

    if getattr(config, "train_llm_lora", True):
        _set_llm_adapter_trainable(model, "vision", True)

    # When new <loc_*> tokens were added, the corresponding rows in ``embed_tokens`` /
    # ``lm_head`` (tied) carry mean-init weights that the model has never been trained
    # on. They are not LoRA targets, so we must explicitly mark the embedding layer
    # trainable to learn them. Skip when not adding new tokens to preserve the LoRA-only
    # training surface.
    if getattr(config, "add_loc_tokens", False):
        model.get_input_embeddings().weight.requires_grad = True

    # Cast trainable (LoRA) params to fp32 so AdamW first/second moments are fp32.
    # bf16 Adam state has only ~7-bit mantissa; sqrt(v_hat) underflows on Ada GPUs and
    # produces NaN updates on the very first step. Base model weights stay bf16.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    return model, processor


@register_prepare_model_and_processor
def prepare_model_and_processor_phi4_add_lora(config):
    processor = _load_phi4_processor(config)
    _rev_kw = _revision_for_pretrained_path(
        config.model_name,
        getattr(config, "model_revision", None),
        default_hub_revision=_PHI4_DEFAULT_HUB_REVISION,
    )
    model = Phi4MMForCausalLM.from_pretrained(
        config.model_name,
        _attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        **_rev_kw,
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    # remove parameters irrelevant to vision tasks
    del model.model.embed_tokens_extend.audio_embed  # remove audio encoder
    for layer in model.model.layers:
        # remove audio lora
        del layer.mlp.down_proj.lora_dropout.speech
        del layer.mlp.down_proj.lora_A.speech
        del layer.mlp.down_proj.lora_B.speech
        del layer.mlp.gate_up_proj.lora_dropout.speech
        del layer.mlp.gate_up_proj.lora_A.speech
        del layer.mlp.gate_up_proj.lora_B.speech
        del layer.self_attn.o_proj.lora_dropout.speech
        del layer.self_attn.o_proj.lora_A.speech
        del layer.self_attn.o_proj.lora_B.speech
        del layer.self_attn.qkv_proj.lora_dropout.speech
        del layer.self_attn.qkv_proj.lora_A.speech
        del layer.self_attn.qkv_proj.lora_B.speech

    # 激活domain
    model.set_lora_adapter('domain')
    image_embed = model.model.embed_tokens_extend.image_embed
    for p in image_embed.parameters():
        p.requires_grad = False
    _configure_image_embed_trainability(model, config)
    if getattr(config, "train_llm_lora", True):
        _set_llm_adapter_trainable(model, "domain", True)
    return model, processor



@register_prepare_model_and_processor
def prepare_model_and_processor_phi4_merge_vision(config):
    processor = _load_phi4_processor(config)
    _rev_kw = _revision_for_pretrained_path(
        config.model_name,
        getattr(config, "model_revision", None),
        default_hub_revision=_PHI4_DEFAULT_HUB_REVISION,
    )
    model = Phi4MMForCausalLM.from_pretrained(
        config.model_name,
        _attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        **_rev_kw,
    )

    # remove parameters irrelevant to vision tasks
    del model.model.embed_tokens_extend.audio_embed  # remove audio encoder
    for layer in model.model.layers:
        # remove audio lora
        del layer.mlp.down_proj.lora_dropout.speech
        del layer.mlp.down_proj.lora_A.speech
        del layer.mlp.down_proj.lora_B.speech
        del layer.mlp.gate_up_proj.lora_dropout.speech
        del layer.mlp.gate_up_proj.lora_A.speech
        del layer.mlp.gate_up_proj.lora_B.speech
        del layer.self_attn.o_proj.lora_dropout.speech
        del layer.self_attn.o_proj.lora_A.speech
        del layer.self_attn.o_proj.lora_B.speech
        del layer.self_attn.qkv_proj.lora_dropout.speech
        del layer.self_attn.qkv_proj.lora_A.speech
        del layer.self_attn.qkv_proj.lora_B.speech

    model.set_lora_adapter('vision')
    image_embed = model.model.embed_tokens_extend.image_embed
    for p in image_embed.parameters():
        p.requires_grad = False
    _configure_image_embed_trainability(model, config)
    if getattr(config, "train_llm_lora", True):
        _set_llm_adapter_trainable(model, "vision", True)

    # return model, processor
    def merge_and_remove_lora(model):
        """
        合并 vision LoRA 权重到 base_layer，并删除 vision 和 speech LoRA。
        """
        vision_lora_layers = []
        speech_lora_layers = []

        # 查找 vision 和 speech LoRA 层
        for name, module in list(model.named_modules()):
            if isinstance(module, LoraLayer):
                if "vision" in module.lora_A.keys():
                    vision_lora_layers.append((name, module))
                elif "speech" in module.lora_A.keys():
                    speech_lora_layers.append((name, module))
        # 合并 vision LoRA 权重
        for name, module in vision_lora_layers:
            if module.merged:
                warnings.warn(f"Layer {name} is already merged, skipping.")
            else:
                try:
                    module.merge()
                    #将lora层替换为合并后的base_layer
                    parent_name = '.'.join(name.split('.')[:-1]) # 'model.layers.0.self_attn'
                    parent_module = model.get_submodule(parent_name)
                    setattr(parent_module, name.split('.')[-1], module.base_layer)
                except Exception as e:
                    warnings.warn(f"Merge layer {name} failed, error: {e}")

    merge_and_remove_lora(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model, processor