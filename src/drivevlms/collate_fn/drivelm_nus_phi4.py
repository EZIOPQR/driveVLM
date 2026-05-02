from transformers import BatchFeature
from ..registry import register_collate_fn
from PIL import Image

def format_prompt_phi4(instruction, input=None):

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
        )
    }
    if input is None:
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
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
import copy
_IGNORE_INDEX = -100
_MAX_TRAINING_LENGTH = 8192
@register_collate_fn
def drivelm_nus_phi4_collate_fn(examples, processor, dtype):
    prompts = [format_prompt_phi4(example["conversations"][0]['value']) for example in examples]
    answers = [format_answer(example["conversations"][1]['value']) for example in examples]
    images = []
    for example in examples:
        image = [Image.open(example["image_paths"][i]).convert("RGB") for i in range(6)]
        # image = [Image.open(example["image_paths"][0]).convert("RGB")]
        images.append(image)
    input_ids_list = []
    labels_list = []
    input_image_embeds_list = []
    image_attention_mask_list = []
    image_sizes_list = []

    for prompt, answer, image in zip(prompts, answers, images):
        image = [img.resize((448, 448), ) for img in image]
        inputs = processor([prompt], images=image, return_tensors='pt')
        answer_ids = processor.tokenizer(answer, return_tensors='pt').input_ids
        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        labels = copy.deepcopy(input_ids)
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
def drivelm_nus_phi4_collate_fn_val(examples, processor, dtype):
    ids = [example["id"] for example in examples]
    questions = [example["conversations"][0]['value'] for example in examples]
    prompts = [format_prompt_phi4(example["conversations"][0]['value']) for example in examples]
    # Flatten 6 cameras x B samples into a single list of B*6 PIL images, in the order
    # sample0_cam0..5, sample1_cam0..5, ..., matching the 6 <|image_k|> tokens per prompt.
    # The Phi-4 processor handles batched left padding internally
    # (see processing_phi4mm.py, "batched inference requires left padding").
    flat_images = []
    for example in examples:
        for i in range(6):
            img = Image.open(example["image_paths"][i]).convert("RGB").resize((448, 448))
            flat_images.append(img)
    tokens = processor(
        text=prompts, images=flat_images, return_tensors="pt", padding="longest"
    )

    return tokens, questions, ids


