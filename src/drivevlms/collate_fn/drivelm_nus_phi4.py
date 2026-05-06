from transformers import BatchFeature
from ..registry import register_collate_fn
from PIL import Image
import os
import cv2
import numpy as np
from drivevlms.utils.flow_io import flow_npz_path_for_image

def format_prompt_phi4(instruction, input=None, include_flow_images: bool = False):

    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "<|user|><|image_1|><|image_2|><|image_3|><|image_4|><|image_5|><|image_6|>"
            "You are an autonomous-driving perception assistant. The six images above are the ego vehicle's surround-view cameras. Follow the rules below carefully.\n\n"
            "## Camera order\n"
            "The 6 images are provided in this fixed order:\n"
            "1. `CAM_FRONT`\n"
            "2. `CAM_FRONT_LEFT`\n"
            "3. `CAM_FRONT_RIGHT`\n"
            "4. `CAM_BACK`\n"
            "5. `CAM_BACK_LEFT`\n"
            "6. `CAM_BACK_RIGHT`\n\n"
            "## Object reference tag format\n"
            "Whenever an object is referenced — **both when you read the question and when you write the answer** — use exactly this tag:\n\n"
            "`<c{{ID}},{{CAMERA}},{{X}},{{Y}}>`\n\n"
            "- `c{{ID}}` — sequential object handle: `c1`, `c2`, `c3`, ...\n"
            "- `{{CAMERA}}` — one of the 6 camera names above (uppercase, with underscores, no quotes)\n"
            "- `{{X}}`, `{{Y}}` — pixel coordinates of the object center in the **224x224** image coordinate system, as floats with 2 decimal places\n\n"
            "Example: `<c1,CAM_FRONT_RIGHT,25.06,103.09>`\n\n"
            "## Reading rule\n"
            "When the question contains tags like `<ci,CAM_xxx,x,y>`, treat each tag as a pointer to one specific object visible in the named camera at the given pixel location. Resolve every tag to a real object before answering.\n\n"
            "## Answering rules\n"
            "1. **Multiple-choice question** (ends with `Please select the correct answer from the following options: A. ... B. ... C. ... D. ...`): respond with **only** the single letter `A`, `B`, `C`, or `D`. No explanation.\n"
            "2. **Yes/No question**: respond with `Yes.` or `No.`.\n"
            "3. **If the answer references objects in the scene**: list them using the tag format above, separated by commas. \n"
            "4. **Otherwise**: write one concise sentence (two at most). Do not repeat the question; do not pad.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:<|end|><|assistant|>"
        ),
        "prompt_no_input_with_flow": (
            "<|user|><|image_1|><|image_2|><|image_3|><|image_4|><|image_5|><|image_6|>"
            "<|image_7|>"
            "You are an autonomous-driving perception assistant. The first six images are the ego vehicle's surround-view cameras. "
            "The seventh image is the optical-flow visualization of CAM_FRONT.\n\n"
            "## Camera order\n"
            "The 6 RGB images are provided in this fixed order:\n"
            "1. `CAM_FRONT`\n"
            "2. `CAM_FRONT_LEFT`\n"
            "3. `CAM_FRONT_RIGHT`\n"
            "4. `CAM_BACK`\n"
            "5. `CAM_BACK_LEFT`\n"
            "6. `CAM_BACK_RIGHT`\n\n"
            "## Object reference tag format\n"
            "Whenever an object is referenced — **both when you read the question and when you write the answer** — use exactly this tag:\n\n"
            "`<c{{ID}},{{CAMERA}},{{X}},{{Y}}>`\n\n"
            "- `c{{ID}}` — sequential object handle: `c1`, `c2`, `c3`, ...\n"
            "- `{{CAMERA}}` — one of the 6 camera names above (uppercase, with underscores, no quotes)\n"
            "- `{{X}}`, `{{Y}}` — pixel coordinates of the object center in the **224x224** image coordinate system, as floats with 2 decimal places\n\n"
            "Example: `<c1,CAM_FRONT_RIGHT,25.06,103.09>`\n\n"
            "## Reading rule\n"
            "When the question contains tags like `<ci,CAM_xxx,x,y>`, treat each tag as a pointer to one specific object visible in the named camera at the given pixel location. Resolve every tag to a real object before answering.\n\n"
            "## Answering rules\n"
            "1. **Multiple-choice question** (ends with `Please select the correct answer from the following options: A. ... B. ... C. ... D. ...`): respond with **only** the single letter `A`, `B`, `C`, or `D`. No explanation.\n"
            "2. **Yes/No question**: respond with `Yes.` or `No.`.\n"
            "3. **If the answer references objects in the scene**: list them using the tag format above, separated by commas. \n"
            "4. **Otherwise**: write one concise sentence (two at most). Do not repeat the question; do not pad.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:<|end|><|assistant|>"
        )
    }
    if input is None:
        key = "prompt_no_input_with_flow" if include_flow_images else "prompt_no_input"
        return PROMPT_DICT[key].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})
    
def format_answer(answer):
    return answer + '<|end|><|endoftext|>'

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), 'All tensors must have the same number of dimensions'

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output

import torch

_IGNORE_INDEX = -100
_MAX_TRAINING_LENGTH = 8192
def _flow_rgb_image_for_path(
    image_path: str,
    flow_root: str,
    flow_scale_u: float,
    flow_scale_v: float,
) -> Image.Image:
    """Load CAM/*.npz flow, visualize as Middlebury HSV (H=direction, S=1, V=magnitude), 448x448."""
    if float(flow_scale_u) == 0.0 or float(flow_scale_v) == 0.0:
        raise ValueError("flow_scale_u and flow_scale_v must be non-zero")
    flow_path = flow_npz_path_for_image(image_path, flow_root)
    if flow_root and os.path.isfile(flow_path):
        z = np.load(flow_path)
        raw_valid = z.get("valid", np.array(True))
        valid = bool(raw_valid.item()) if isinstance(raw_valid, np.ndarray) and raw_valid.size else bool(raw_valid)
        if valid:
            u = np.asarray(z["u"], dtype=np.float32)
            v = np.asarray(z["v"], dtype=np.float32)
        else:
            u = np.zeros_like(np.asarray(z["u"], dtype=np.float32))
            v = np.zeros_like(u)
    else:
        u = np.zeros((448, 448), dtype=np.float32)
        v = np.zeros((448, 448), dtype=np.float32)
    flow_hw = u.shape[0]
    u = u / float(flow_scale_u)
    v = v / float(flow_scale_v)
    mag = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v, u)
    hsv = np.zeros((flow_hw, flow_hw, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (np.clip(mag, 0, 1) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img = Image.fromarray(rgb, mode="RGB")
    if flow_hw != 448:
        img = img.resize((448, 448), Image.NEAREST)
    return img


@register_collate_fn
def drivelm_nus_phi4_collate_fn(
    examples,
    processor,
    dtype,
    use_optical_flow: bool = False,
    flow_root: str = "",
    flow_scale_u: float = 8.778,
    flow_scale_v: float = 2.888,
):
    prompts = [
        format_prompt_phi4(
            example["conversations"][0]['value'],
            include_flow_images=bool(use_optical_flow and flow_root),
        )
        for example in examples
    ]
    answers = [format_answer(example["conversations"][1]['value']) for example in examples]
    images = []
    for example in examples:
        rgb_images = [Image.open(example["image_paths"][i]).convert("RGB") for i in range(6)]
        if use_optical_flow and flow_root:
            flow_img = _flow_rgb_image_for_path(example["image_paths"][0], flow_root, flow_scale_u, flow_scale_v)
            images.append(rgb_images + [flow_img])
        else:
            images.append(rgb_images)
    input_ids_list = []
    labels_list = []
    input_image_embeds_list = []
    image_attention_mask_list = []
    image_sizes_list = []

    for example, prompt, answer, image in zip(examples, prompts, answers, images):
        image = [img.resize((448, 448)) for img in image]
        inputs = processor([prompt], images=image, return_tensors='pt')
        answer_ids = processor.tokenizer(answer, return_tensors='pt').input_ids
        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -answer_ids.shape[1] :] = answer_ids

        if input_ids.size(1) > _MAX_TRAINING_LENGTH:
            input_ids = input_ids[:, :_MAX_TRAINING_LENGTH]
            labels = labels[:, :_MAX_TRAINING_LENGTH]
            if torch.all(labels == _IGNORE_INDEX).item():
                # workaround to make sure loss compute won't fail
                labels[:, -1] = processor.tokenizer.eos_token_id
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        input_image_embeds_list.append(inputs.input_image_embeds)
        image_attention_mask_list.append(inputs.image_attention_mask)
        image_sizes_list.append(inputs.image_sizes)

    input_ids = pad_sequence(input_ids_list, padding_side='right', padding_value=0)
    labels = pad_sequence(labels_list, padding_side='right', padding_value=0)
    attention_mask = (input_ids != 0).long()
    input_image_embeds = cat_with_pad(input_image_embeds_list, dim=0)
    image_attention_mask = cat_with_pad(image_attention_mask_list, dim=0)
    image_sizes = torch.cat(image_sizes_list)
    return BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_image_embeds': input_image_embeds,
            'image_attention_mask': image_attention_mask,
            'image_sizes': image_sizes,
            'input_mode': 1,  # vision mode
        }
    )

@register_collate_fn
def drivelm_nus_phi4_collate_fn_val(
    examples,
    processor,
    dtype,
    use_optical_flow: bool = False,
    flow_root: str = "",
    flow_scale_u: float = 8.778,
    flow_scale_v: float = 2.888,
):
    ids = [example["id"] for example in examples]
    questions = [example["conversations"][0]['value'] for example in examples]
    prompts = [
        format_prompt_phi4(
            example["conversations"][0]['value'],
            include_flow_images=bool(use_optical_flow and flow_root),
        )
        for example in examples
    ]
    flat_images = []
    for example in examples:
        rgb_images = [
            Image.open(example["image_paths"][i]).convert("RGB").resize((448, 448))
            for i in range(6)
        ]
        flat_images.extend(rgb_images)
        if use_optical_flow and flow_root:
            flow_img = _flow_rgb_image_for_path(example["image_paths"][0], flow_root, flow_scale_u, flow_scale_v)
            flat_images.append(flow_img)
    tokens = processor(
        text=prompts, images=flat_images, return_tensors="pt", padding="longest"
    )
    return tokens, questions, ids


