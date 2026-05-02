import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import GenerationConfig
from PIL import Image
import argparse
import os
import cv2
import numpy as np
import re

_imgs_filename = [
    "data/DriveLM_nuScenes/nuscenes/samples/CAM_FRONT/n008-2018-09-18-12-07-26-0400__CAM_FRONT__1537287220662404.jpg",
      "data/DriveLM_nuScenes/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-09-18-12-07-26-0400__CAM_FRONT_LEFT__1537287220654799.jpg",
      "data/DriveLM_nuScenes/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-09-18-12-07-26-0400__CAM_FRONT_RIGHT__1537287220670482.jpg",
      "data/DriveLM_nuScenes/nuscenes/samples/CAM_BACK/n008-2018-09-18-12-07-26-0400__CAM_BACK__1537287220687558.jpg",
      "data/DriveLM_nuScenes/nuscenes/samples/CAM_BACK_LEFT/n008-2018-09-18-12-07-26-0400__CAM_BACK_LEFT__1537287220697405.jpg",
      "data/DriveLM_nuScenes/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-09-18-12-07-26-0400__CAM_BACK_RIGHT__1537287220678113.jpg"
]

# Load model and processor
model_path = "/root/autodl-tmp/models/Phi-4-multimodal-instruct"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True
)

# model = AutoModelForCausalLM.from_pretrained(
#     '/data2/private-data/zhangn/pretrained/phi4/FULL-2025-03-24_21-59/hf_ckpt',
#     torch_dtype=torch.float16, 
#     _attn_implementation='sdpa',
#     revision ="607bf62a754018e31fb4b55abbc7d72cce4ffee5",
#     trust_remote_code=True
# )

#TODO since phi4 finetune vision lora is self-contained in model
# the loading method shoud be differenct maybe

# model = PeftModel.from_pretrained(model, '/data2/private-data/zhangn/pretrained/paligemma/FULL-2025-03-15_21-49/final_model/')
# model = model.merge_and_unload()

model.to('cuda')

# Load generation config
generation_config = GenerationConfig.from_pretrained(model_path)

# Define prompt structure
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

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


def format_prompt(instruction, input=None):

    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "<|user|><|image_1|><|image_2|><|image_3|><|image_4|><|image_5|><|image_6|>"
            "Below is an instruction describing a driving perception task, along with six images from different views around the ego vehicle.\n"
            "Each image corresponds to a specific camera: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT.\n"
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:<|end|><|assistant|>"
        )
    }
    if input is None:
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})


def tokenize(texts, images, processor, device='cuda'):
    return processor(
        text=texts, images=images, return_tensors="pt", padding="longest"
    ).to(device)


def visualize(imgs_filename, detection_text, save_path="visualized_output.jpg", model_size=448):
    if isinstance(detection_text, list):
        detection_text = detection_text[0]
    matches = re.findall(r"<([^>]+)>", detection_text)
    detections = []
    for match in matches:
        parts = match.split(",")
        if len(parts) == 4:
            obj_id, cam, x, y = parts
            detections.append({
                "id": obj_id.strip(),
                "camera": cam.strip(),
                "x": float(x),
                "y": float(y)
            })

    images = {}
    for img_path in imgs_filename:
        img = cv2.imread(img_path)
        if img is None:
            continue
        cam = os.path.basename(img_path).split("__")[1]
        images[cam] = img

    for det in detections:
        cam = det["camera"]
        obj_id = det["id"]
        if cam in images:
            img = images[cam]
            h, w = img.shape[:2]
            x = int(det["x"] / model_size * w)
            y = int(det["y"] / model_size * h)
            cv2.circle(img, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(img, obj_id, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)

    front_order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
    back_order  = ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
    target_size = (640, 360)

    row_front = cv2.hconcat([cv2.resize(images[cam], target_size) for cam in front_order])
    row_back  = cv2.hconcat([cv2.resize(images[cam], target_size) for cam in back_order])
    combined_image = cv2.vconcat([row_front, row_back])

    cv2.imwrite(save_path, combined_image)
    print(f"Visualized image saved to: {save_path}")


@torch.no_grad()
def main(args):
    sample_prompts = [
        'What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.'
    ]

    for instruction in sample_prompts:
        prompt = format_prompt(instruction)
        print(f"\nGenerating for prompt: {repr(prompt)}")
        images = [Image.open(cam).convert("RGB") for cam in _imgs_filename]
        images = [image.resize((448, 448), ) for image in images]
        reason_inputs = tokenize([prompt], images, processor, args.device)
        reason_results = infer(reason_inputs)
        print(reason_results)

        visualize(_imgs_filename, reason_results)

def parse_args():
    parser = argparse.ArgumentParser(description='DriveLM Phi4 Inference')
    parser.add_argument("--device", default="cuda", help="Device to run inference")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())