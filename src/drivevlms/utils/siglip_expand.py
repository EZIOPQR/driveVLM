"""Expand SigLIP patch embedding from 3 to N input channels (in-place on model)."""

import torch
from torch import nn


def expand_siglip_vision_patch_in_channels(model, in_channels: int = 5) -> None:
    """Replace vision embeddings Conv2d 3 -> N; copy RGB weights, new input slices zero.

    Expects ``model.model.embed_tokens_extend.image_embed.img_processor`` to be
    a ``SiglipVisionTransformer`` as in Phi-4-MM.
    """
    vision = model.model.embed_tokens_extend.image_embed.img_processor
    old: nn.Conv2d = vision.embeddings.patch_embedding
    if old.in_channels == in_channels:
        return
    if old.in_channels != 3:
        raise ValueError(f"Expected 3 input channels, got {old.in_channels}")

    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight[:, :3] = old.weight[:, :3].clone()
        new.weight[:, 3:].zero_()
        if old.bias is not None:
            new.bias.copy_(old.bias)

    vision.embeddings.patch_embedding = new
    vision.config.num_channels = in_channels
