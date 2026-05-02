# DriveVLMs v0.2 项目说明文档

> 本文档基于对整个 `DriveVLMs_v3` 仓库的源码走查撰写，并结合原始论文
> *DriveLM: Driving with Graph Visual Question Answering*（arXiv:2312.14150v3，
> 见仓库根目录下 `2312.14150v3.pdf`）来解释数据组织、模型选型与推理流程。
> 本文档面向希望快速了解项目组成、新增模型/数据集，或复现训练与推理的同学。

---

## 1. 项目定位

`DriveVLMs` 是一套 **自动驾驶场景下视觉-语言模型（VLM）的微调 + 推理 + 评测框架**：

- **训练数据**：目前对接的是 [DriveLM-nuScenes](https://github.com/OpenDriveLab/DriveLM)，即
  论文中提出的 `DriveLM-Data` 在 nuScenes 上的实例。原始数据由六路环视相机
  （CAM_FRONT / CAM_FRONT_LEFT / CAM_FRONT_RIGHT / CAM_BACK / CAM_BACK_LEFT / CAM_BACK_RIGHT）
  与论文所定义的 **Graph VQA** 组成（P1 感知 → P2 预测 → P3 规划 → B 行为）。
- **支持的 VLM**：
  - `google/paligemma-3b-pt-224`（以及社区 fine-tune 版本 `lykong/paligemma-finetuned`）
  - `microsoft/Phi-4-multimodal-instruct`（在本项目中裁掉了 audio 分支，只保留 vision LoRA）
- **能力**：
  - 将原始 DriveLM JSON 转成 HuggingFace `datasets.Dataset` 格式
  - 使用 `accelerate` + `peft` 做单卡 / DDP 微调（含 LoRA、梯度累积、断点续训）
  - 批量推理整个 val split
  - 使用 BLEU / ROUGE-L / CIDEr / match score / accuracy 等指标评测
  - 提供单条样本的 Demo 可视化脚本（把模型输出的 `<c_i,CAM_xxx,x,y>` 坐标回绘到原图）

---

## 2. 目录结构与代码组成

```
DriveVLMs_v3/
├── README.md                         # 用户侧使用说明（安装、训练、推理）
├── setup.py                          # 包定义，依赖见下方 §2.1
├── 2312.14150v3.pdf                  # 原论文
├── paligemma_update.json             # 训练日志（由 tools/finetune.py 写入）
│
├── configs/                          # 训练配置（dataclass）
│   ├── paligemma/paligemma_drivelm_config.py
│   └── phi4/phi4_drivelm_1xb1-lora_config.py
│
├── data/DriveLM_nuScenes/            # 数据目录（大文件不入库）
│   ├── QA_dataset_nus/               # 原始 DriveLM JSON
│   ├── nuscenes/samples/             # 原始 6 路环视图像
│   ├── refs/                         # 规整后的 JSON：train_cot / val_cot / val_qa_style / infer_epoch*
│   └── split/{train,val}/            # HuggingFace Dataset style（save_to_disk 输出）
│
├── demos/                            # 单样本推理 demo，含输出可视化
│   ├── DriveLM_demo_paligemma.py
│   ├── DriveLM_demo_paligemma2.py
│   └── DriveLM_demo_phi4.py
│
├── tools/                            # 命令行入口
│   ├── create_data/create_drivelm_nus.py   # 数据预处理 / 切分
│   ├── finetune.py                         # 微调入口（支持 DDP）
│   ├── inference.py                        # 全集推理
│   └── evaluation.py                       # 与 GT 对比，计算 BLEU/ROUGE/CIDEr/F1/Acc
│
└── src/drivevlms/                    # 核心 Python 包（package_dir={"": "src"}）
    ├── __init__.py                   # 触发 collate_fn / preparation / models 的自动注册
    ├── registry.py                   # 两个全局注册表：COLLATE_FN_REGISTRY, PREPARE_REGISTRY
    ├── build.py                      # 从字符串名取出对应 fn（训练入口使用）
    ├── utils.py                      # dataloader、优化器、checkpoint、动态加载 config
    ├── metric.py / metrics/          # 旧版 + 新版 metric，语言评测以 language_evaluation 为主
    │
    ├── preparation/                  # "如何加载模型 + 装 LoRA" 的策略族
    │   ├── paligemma_preparation.py          # PaliGemma + LoRA / quant / flash attention
    │   └── phi4_preparation.py               # Phi-4-MM：裁剪 audio 分支，切到 vision LoRA
    │
    ├── collate_fn/                   # "如何把一个样本 batch 成模型输入" 的策略族
    │   ├── drivelm_nus_paligemma.py
    │   ├── drivelm_nus_phi4.py
    │   ├── occ_vla_paligemma.py              # 另一套数据（OCC-VLA）的 paligemma 版本
    │   └── occ_vla_phi4.py
    │
    └── models/phi4_bjxx/             # 本项目裁剪/魔改过的 Phi-4-MM 源码（关键！）
        ├── configuration_phi4mm.py           # Phi4MMConfig
        ├── modeling_phi4mm.py                # Phi4MMModel / Phi4MMForCausalLM
        ├── processing_phi4mm.py              # Phi4MMProcessor + ImageProcessor + AudioFeatureExtractor
        ├── vision_siglip_navit.py            # SigLIP 视觉塔
        └── speech_conformer_encoder.py       # Conformer 语音塔（项目中默认会 del 掉）
```

### 2.1 依赖（`setup.py`）

| 依赖 | 版本 | 说明 |
| --- | --- | --- |
| `transformers` | **4.48.2**（锁死） | 与项目内魔改过的 Phi-4-MM 代码配套 |
| `accelerate` | ≥ 1.3.0 | 用于混合精度 + DDP |
| `peft` | 0.15.1 | LoRA 注入 / 保存 / merge |
| `soundfile / scipy / torchvision / pillow / backoff` | 见 `setup.py` | Phi-4-MM 的图像/音频前处理 |

### 2.2 注册机制（registry）

`src/drivevlms/registry.py` 维护两个全局 dict：

```python
COLLATE_FN_REGISTRY = {}   # name -> collate fn
PREPARE_REGISTRY   = {}    # name -> "加载模型 + 注入 LoRA" 的函数
```

两个装饰器 `@register_collate_fn`、`@register_prepare_model_and_processor` 分别在
`collate_fn/*.py` 和 `preparation/*.py` 中 **import 期** 把函数塞进表里。
`src/drivevlms/__init__.py` 通过

```python
from .collate_fn import *
from .preparation import *
from .models import *
```

触发一次侧效应导入，保证 `build.py` 中的 `build_collate_fn(name)` /
`build_preparation(name)` 能按配置文件里给的字符串取出实现。

### 2.3 Config 的约定

训练/推理入口都通过 `drivevlms.utils.load_dataclass_config(path)` 动态 import 一个
Python 文件，并要求文件里有名为 `config` 的 dataclass 实例。典型字段见
`configs/phi4/phi4_drivelm_1xb1-lora_config.py`：

| 字段 | 作用 |
| --- | --- |
| `model_name` | HF 模型名或本地路径 |
| `model_preparation` | 会被 `build_preparation(...)` 查表，指向 `preparation/` 中的函数 |
| `collate_fn_train` | 会被 `build_collate_fn(...)` 查表 |
| `dataset_name` | 训练集的 HF Dataset 磁盘路径 |
| `use_lora / lora_r / use_flash_attention / quantization / dtype` | 微调策略开关 |
| `batch_size_per_gpu / gradient_accumulation_steps / lr / warmup_steps / num_train_epochs` | 标准训练超参 |
| `resume_from_checkpoint / save_steps / log_steps / print_steps` | 恢复与日志 |
| `find_unused_parameters` | 仅 Phi-4 需要置 True（见 §3.2） |

---

## 3. 模型侧：两条技术路线

### 3.1 PaliGemma 路线
`src/drivevlms/preparation/paligemma_preparation.py`：
- 用 `PaliGemmaForConditionalGeneration.from_pretrained` 直接加载。
- 根据 config 可选：4bit NF4 量化（`BitsAndBytesConfig`）、`flash_attention_2`（要求 bf16）。
- 支持继续套一层 `PeftModel.from_pretrained(model, config.peft_name)` 再 `merge_and_unload()`。
- 默认 **解冻 `vision_tower` + `multi_modal_projector`** 的全部参数（全参数微调）。
- LLM 部分通过 `peft.LoraConfig(target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj])` 注入 LoRA。

### 3.2 Phi-4-Multimodal 路线
本项目 **直接复制并裁剪了 HuggingFace 上的 Phi-4-MM 源码**，放在
`src/drivevlms/models/phi4_bjxx/`，并在 README 中明确声明：

> 为便于 LLM 在芯片上部署，`PaliGemma` 与 `phi4` 的 transformers 源码被做了修改；
> `src/drivevlms/models` 目录下的就是修改后的版本。

关键改动：
- `processing_phi4mm.py` 的 `Phi4MMImageProcessor.preprocess` 禁用了默认的
  HD dynamic crop 流程，强制把输入当作 **单尺度 448×448** 的图走，省掉了
  attention mask 的多 crop 重排（见文件 200~260 行大段注释掉的代码与 `# new` 标记）。
  这样每张图的 token 数固定为 `(448/14/2)^2 + 16 = 16*16/?`（实际见代码 line 241）。
- `modeling_phi4mm.py::Phi4MMForCausalLM.forward` 里会根据 processor 生成的
  `input_mode`（0/1/2/3 对应 `LANGUAGE / VISION / SPEECH / VISION_SPEECH`）
  **动态切换** peft 的 LoRA adapter：
  ```python
  if input_mode in [VISION_SPEECH, VISION]:
      self.set_lora_adapter('vision')
  elif input_mode == SPEECH:
      self.set_lora_adapter('speech')
  elif input_mode == LANGUAGE:
      self.unset_lora_adapter()
  ```
- `preparation/phi4_preparation.py` 在加载完 base 模型后：
  1. `del model.model.embed_tokens_extend.audio_embed` — 丢掉 Conformer 音频塔。
  2. 遍历每一层 MLP / self-attn，`del` 掉所有 `speech` adapter 的 `lora_A/lora_B/lora_dropout` —
     只保留 `vision` adapter，避免 DDP 下未使用参数报错。
  3. `model.set_lora_adapter('vision')`；并把 `embed_tokens_extend.image_embed`
     （SigLIP 投影 + GN）的参数全部 `requires_grad = True`。
  4. `gradient_checkpointing_enable(use_reentrant=False)`。
- 该路线 **必须** 在 DDP 下把 `find_unused_parameters=True`，因此 `tools/finetune.py`
  专门提供了一个 `MyAccelerator` 子类（见 §4.2）。

Phi-4-MM 的 embedding 流程（`Phi4MMImageAudioEmbedding.forward`）：
1. 输入里 `<|image_1|> ... <|image_6|>` 被 processor 先替成单个的 `<|endoftext10|>`
   （id 200010），再 **展开为 `num_img_tokens` 个同样的 token**。
2. forward 时用 `input_ids == 200010` 找出所有图像位置，用
   `Phi4MMImageEmbedding`（SigLIP backbone + MLP 投影 + 可学习分隔符 `glb_GN/sub_GN`）
   把图像 patch 投到 LLM 维度，再通过 `hidden_states.index_put(positions)` 贴回去，
   文本位置则走标准 `wte(input_ids)`，最后送进 32 层 `Phi4MMDecoderLayer`。

---

## 4. 端到端流程

### 4.1 数据准备流程 （`tools/create_data/create_drivelm_nus.py`）

```text
v1_1_train_nus.json
    │
    ├─ extract_data()       按场景/关键帧遍历，按规则挑选指定的 QA：
    │                         · perception：挑一条含所有重要物体描述的 QA + 一条 "What is the moving status of object …" 的多选
    │                         · prediction：挑含所有 <c_i,CAM_xxx,x,y> 位置的 QA + 一条 yes/no 题
    │                         · planning  ：挑 actions / collision / safe actions 各一条
    │                         · behavior  ：全部保留
    │                       并给每条 QA 打上 tag：
    │                         0=accuracy, 1=chatGPT（未启用）, 2=language, 3=match
    │
    ├─ loop_test()          对挑出的 QA：
    │                         · moving status 题 → rule_based1 生成 A/B/C/D 四选一
    │                         · behavior       → rule_based2 生成 A/B/C/D 四选一
    │
    ├─ split_by_key_frame() 以 key_frame 为最小单位，种子 42 随机打乱后
    │                       80/20 划分 train/val
    │
    ├─ convert_coors_system()
    │                       把所有 "<c_i,CAM,x,y>" 文本里的像素坐标
    │                       从原始 1600x900 → 目标 224x224（v0.2 更新点）
    │                       PaliGemma 用 224，Phi-4 在 collate 时再 resize 到 448
    │
    ├─ convert2vlm()        转成 LLaMA-Adapter 风格的对话：
    │                         {"id", "image", "conversations":[{"from":"human","value":Q},
    │                                                          {"from":"gpt","value":A}]}
    │
    └─ convert_to_hf_dataset()  再转成 HF Dataset 并 `save_to_disk` 到
                                  data/DriveLM_nuScenes/split/{train,val}/
                                同时写出人读版 JSON：
                                  refs/train_cot.json    （切分后但未拉平，用于 eval 时对齐 GT）
                                  refs/val_cot.json
                                  refs/val_qa_style.json
```

数据集中每条样本（HF Dataset 的 row）形如：

```python
{
  "id": "<scene_id>_<frame_id>_<qa_idx>",
  "image_paths": [CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
                  CAM_BACK,  CAM_BACK_LEFT,  CAM_BACK_RIGHT],
  "conversations": [
    {"from": "human", "value": "<question 文本>"},
    {"from": "gpt",   "value": "<answer   文本>"},
  ],
}
```

这与论文 §3 的 Graph VQA 结构一一对应：Perception / Prediction / Planning / Behavior
四类 QA 节点，对应的位置标签 `<c_i,CAM_xxx,x,y>` 充当 Graph VQA 中的 **object-level 边**。

### 4.2 微调流程（`tools/finetune.py`）

```text
python tools/finetune.py configs/phi4/phi4_drivelm_1xb1-lora_config.py
```

伪流程：

```
load_dataclass_config(args.config)  →  config (dataclass)
set_seed(config.seed)

# 1. 根据 find_unused_parameters 选择标准 Accelerator 或 MyAccelerator
#    MyAccelerator 重写 prepare_model，把 DDP 的 find_unused_parameters=True
accelerator = (MyAccelerator|Accelerator)(mixed_precision=bf16|no, ...)

# 2. 根据 config.model_preparation 从 PREPARE_REGISTRY 查表
prepare = build_preparation(config.model_preparation)
model, processor = prepare(config)          # 内部完成：加载 + 量化 + LoRA + 冻结策略

# 3. 根据 config.collate_fn_train 从 COLLATE_FN_REGISTRY 查表
collate_fn = build_collate_fn(config.collate_fn_train)
train_collate_fn = partial(collate_fn, processor=processor, dtype=config.dtype)
dataloader = DataLoader(load_from_disk(config.dataset_name),
                        batch_size=config.batch_size_per_gpu,
                        collate_fn=train_collate_fn,
                        shuffle=True, num_workers=16, pin_memory=True)

# 4. AdamW + cosine warmup
optimizer, scheduler = prepare_optimizer_and_scheduler(config, model, num_steps)
dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

# 5. 断点恢复
if config.resume_from_checkpoint and exists(output_dir/training_info.json):
    load_checkpoint(accelerator, training_info['latest_checkpoint'])
    skipped_dataloader = accelerator.skip_first_batches(dataloader, skip_batch_count)

# 6. 训练循环（N epoch × num_batches）
for batch in dataloader:
    with accelerator.accumulate(model):
        loss = model(**batch).loss
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()

    # 周期性日志（tensorboard + wandb + paligemma_update.json）与 checkpoint
    if global_step % config.save_steps == 0: save_checkpoint(...)

# 7. 每个 epoch 结尾：save_checkpoint + save_lora_adapter 到 output_dir/epoch-{N}
# 8. 训练结束：output_dir/final_model 存 LoRA adapter + 完整 unwrapped_model
```

`collate_fn` 两条路线的关键差异：

| 步骤 | PaliGemma (`drivelm_nus_paligemma_collate_fn_train`) | Phi-4 (`drivelm_nus_phi4_collate_fn`) |
| --- | --- | --- |
| Prompt 构造 | `"You are an autonomous driving labeler. You have access to six camera images (front, front-right, front-left, back, back-right, back-left).\n{instruction}"` | `"<|user|><|image_1|>...<|image_6|>...### Instruction:\n{instruction}\n\n### Response:<|end|><|assistant|>"` |
| 答案监督 | 通过 `processor(..., suffix=labels)` 自动把 answer 当作 suffix 计算 loss | 手动把 `prompt_ids + answer_ids` 拼在一起，prompt 部分的 label 置为 0（配合 0-pad），answer 部分直接作为 label |
| 图像尺寸 | 由 processor 默认 224×224 | `image.resize((448,448))`，给 Phi4MM 的 single-scale 分支用 |
| 输出字段 | `BatchFeature` 标准键 | 额外包含 `input_image_embeds, image_attention_mask, image_sizes, input_mode=1` |
| max_len 截断 | 无 | `_MAX_TRAINING_LENGTH = 8192`，超过则截断并兜底塞 eos，避免 label 全被忽略时 loss 为 NaN |

### 4.3 推理流程（`tools/inference.py`）

```text
python tools/inference.py \
    --model /path/to/trained_or_base_phi4 \
    --data  data/DriveLM_nuScenes/split/val \
    --collate_fn drivelm_nus_phi4_collate_fn_val \
    --output data/DriveLM_nuScenes/refs/infer_results.json
```

内部流程：

```text
1. AutoProcessor.from_pretrained(base_model)
   AutoModelForCausalLM.from_pretrained(args.model, bf16, flash_attention_2)
   GenerationConfig.from_pretrained(base_model)
   model.to(cuda)

2. collate_fn = build_collate_fn("drivelm_nus_phi4_collate_fn_val")
   （val 版本只做 prompt + image→tensor，不做 answer 拼接；
     另外返回 questions, ids 给下游 JSON 聚合）

3. DataLoader(Dataset, batch_size=1, collate_fn=val_collate_fn, shuffle=False)

4. for batch in dataloader:
       inputs, question, ids = batch
       input_len = inputs["input_ids"].shape[-1]
       output = model.generate(**inputs,
                               max_new_tokens=1000,
                               generation_config=generation_config)
       output = output[:, input_len:]      # 只保留新生成部分
       results = processor.batch_decode(output, skip_special_tokens=True)
       data_dict.append({"id": id, "question": question, "answer": results})

   # 每步都增量写出 JSON，便于中途崩溃后续推
```

对应 Phi-4-MM 内部的单 step 生成（`Phi4MMForCausalLM.forward`）：

```
processor(..., images)
   ├─ Phi4MMImageProcessor.preprocess
   │     · 每张图 → (1,3,448,448)，stack 成 (6,1,3,448,448)
   │     · image_attention_mask 全 1（禁用了 dynamic HD crop）
   │     · num_img_tokens = (448/(14*2))^2 + 16
   ├─ tokenizer：把 "<|image_k|>" 全部替换成 "<|endoftext10|>"
   ├─ _convert_images_audios_text_to_inputs：
   │     · 每遇到一个 <|endoftext10|> token，就把它展开成 num_img_tokens 份
   │     · left-pad（生成时必须左 pad），pad 用 tokenizer.pad_token_id
   └─ 输出 BatchFeature: input_ids, attention_mask,
                          input_image_embeds, image_sizes, image_attention_mask,
                          input_audio_embeds(空), audio_embed_sizes(空),
                          input_mode=VISION(1)

model.generate → 反复调用 Phi4MMForCausalLM.forward
   · 首次 step：读取 input_mode=VISION，set_lora_adapter('vision')
   · Phi4MMModel.forward 里，inputs_embeds is None 走 embed_tokens_extend
        → Phi4MMImageAudioEmbedding.forward
        → 图像位置用 SigLIP+projection 填充，文本位置用 wte
   · 32 层 Phi4MMDecoderLayer（带 LoRA）→ RMSNorm → lm_head → logits
   · 第二 step 起：只送新 token 的 input_ids；Phi4MMImageAudioEmbedding 已把
     self.input_image_embeds 置为 None，不再重复过 SigLIP（KV cache 继续复用图像向量）
```

PaliGemma 的推理路径类似，只是处理器直接走 HF 原版，不涉及 `input_mode` 切换 LoRA。

### 4.4 评测流程（`tools/evaluation.py`）

```text
python tools/evaluation.py \
    --src data/DriveLM_nuScenes/refs/infer_results.json \
    --tgt data/DriveLM_nuScenes/refs/val_cot.json
```

`evaluation_suit` 按 QA 的 `tag` 分桶：

| tag | 含义 | 指标 |
| --- | --- | --- |
| 0 | 多选 / 是非题 | 精确匹配 accuracy |
| 1 | 开放式 planning（GPT 评分占位，当前版本未启用） | — |
| 2 | perception 自由生成 | BLEU-1~4 / ROUGE-L / CIDEr（`language_evaluation.CocoEvaluator`） |
| 3 | prediction 定位题 `<c_i,CAM,x,y>` | `match_result`：提取所有浮点对，再以 L1 距离 < 16 判定匹配，算 F1 |

其中 `eval_graph` 还会限制 **只对那些坐标与前一条 perception 预测匹配的问题**计算指标，
对应论文中 Graph VQA 的图结构一致性约束：当上层 perception 错时，下游 prediction/planning 的题也会被跳过。

最终还会按固定权重把三档分数归一化到 `[0,1]`：

- `language` 四项 BLEU / ROUGE / CIDEr 合成（CIDEr 比重更大，除以 3 而不是 12）
- `match` 除以 100
- `accuracy` 直接 0~1

### 4.5 Demo（`demos/DriveLM_demo_phi4.py` / `DriveLM_demo_paligemma.py`）

单样本推理示例，把 6 张图喂给模型，打印生成答案，并 **把答案里形如
`<c1,CAM_FRONT_RIGHT,112.3,42.5>` 的点回绘到对应相机图上**，按
`[CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT] / [CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT]`
上下两行拼成 `visualized_output.jpg`。

对 Phi-4 demo 需要注意：模型输出的坐标是 448×448 尺度，可视化时会用
`x * w / 448` 还原到原始 1600×900 分辨率。

---

## 5. 与论文 (DriveLM, arXiv:2312.14150v3) 的对应关系

| 论文概念 | 本项目对应实现 |
| --- | --- |
| DriveLM-Data (nuScenes split) | `data/DriveLM_nuScenes/QA_dataset_nus/v1_1_train_nus.json` 作为输入；`create_drivelm_nus.py` 复现 Perception / Prediction / Planning / Behavior 四类 QA 的采样、加选项与坐标缩放；`refs/train_cot.json` + `refs/val_cot.json` 保留 Graph 结构用于评测 |
| Graph VQA（object-level / task-level 依赖）| Perception 题先定位 `<c_i,CAM,x,y>` → 后续 Prediction / Planning 题都引用同一组 `<c_i,...>`；`evaluation.py::set_graph/eval_graph` 用 L1<16 的匹配确认"图"是否一致，再决定是否计分 |
| DriveLM-Agent（VLM baseline） | `preparation/paligemma_preparation.py` + `preparation/phi4_preparation.py` 以 PaliGemma / Phi-4-MM 作为 Agent 骨架；通过 LoRA + 部分参数解冻适配驾驶场景 |
| DriveLM-Metrics | `evaluation.py` 里的 language（BLEU/ROUGE/CIDEr）、match（F1 of 2D 点）、accuracy 三类，与论文 §4.3 的指标大体对齐（不含 GPT 打分，代码中留有占位） |
| P1→P2→P3→B 任务链 | 数据集里每帧关键帧内 QA 的 `id` 以 `scene_id_frame_id_qaidx` 命名，评测时同一帧内先过 perception 设"图"，再逐条处理 prediction/planning/behavior |
| CARLA split & Motion 任务 | **本仓库暂未实现**，只覆盖 nuScenes 的 P1–P3 + Behavior；behavior 的 21 选项列表来自论文附录并在 `rule_based2` 中硬编码 |

---

## 6. 新增模型 / 数据集的扩展点

README 已给出清单，对应代码位置如下：

1. **新增 `preparation` 函数**：在 `src/drivevlms/preparation/` 新建文件，用
   `@register_prepare_model_and_processor` 装饰，返回 `(model, processor)`。
2. **新增 `collate_fn`**：在 `src/drivevlms/collate_fn/` 新建文件，用
   `@register_collate_fn` 装饰；注意 Phi-4 类模型需要手动拼接 label
   （PaliGemma 只需要把 answer 当 `suffix` 传给 processor）。
3. **新增 config**：在 `configs/<model>/` 下放一个 `.py`，dataclass 实例必须叫
   `config`，字段参考 §2.3。`model_preparation` 与 `collate_fn_train` 字段
   必须能在两个注册表里查到。
4. **新增数据集**：按 §4.1 的流程把原始 JSON 转成与 `drivelm_nus_paligemma_collate_fn_*`
   / `drivelm_nus_phi4_collate_fn*` 兼容的字段（`image_paths` + `conversations`），
   否则需要另写 collate。参考 `occ_vla_*.py`（另一种数据集 `cam_front/...` 字段的写法）。
5. **裁剪 / 魔改模型**：像 `models/phi4_bjxx/` 那样，把修改过的 `configuration / modeling / processing` 放进去，
   在 `preparation` 中 `from drivevlms.models.phi4_bjxx import Phi4MMProcessor, Phi4MMForCausalLM` 调用。

---

## 7. 当前已知限制与注意事项

- `inference.py` 里 `base_model` 路径被硬编码为 `"/root/autodl-tmp/models/Phi-4-multimodal-instruct"`
  （用来初始化 `processor` 与 `GenerationConfig`）。如果迁移到其他机器，需改成对应的本地路径或 HF repo id。
- `drivelm_nus_phi4_collate_fn_val` 当前 **只支持 batch size = 1**（hardcoded `images[0]`），与 `tools/inference.py` 的 DataLoader 设置一致。
- PaliGemma demo (`DriveLM_demo_paligemma.py`) 里有多个 `prompt_no_input` 定义，**后者会覆盖前者**（Python 字典后入覆盖），
  实际生效的 prompt 是最下面那条 `"answer en You are an   driving labeler. ..."`，如需更换请直接编辑该块。
- Phi-4 路线必须 `find_unused_parameters=True`（见 `tools/finetune.py::MyAccelerator`），
  因为 SigLIP 视觉塔里存在一部分前向不过的参数。
- `src/drivevlms/metrics/drivelm_nus_metric.py` 的 `cal_metrics_drivelm_paligemma2` 仅占位（`pass`），
  目前最终指标以 `tools/evaluation.py` 为准。
- 论文中的 **Motion（连续轨迹）** 任务与 **CARLA split / zero-shot to Waymo** 实验
  在本仓库里没有落地，当前代码库是论文 §3–§4 的 nuScenes QA 子集。

---

## 8. 一句话流程总结

> **数据**：`v1_1_train_nus.json → extract_data → loop_test（多选增广）→ 80/20 split → 坐标 1600×900→224 → convert2vlm → HF Dataset`
> **训练**：`config.py → load_dataclass_config → build_preparation(注册表) → build_collate_fn(注册表) → accelerate.prepare → AdamW+cosine → per-step accumulate → per-epoch save_lora_adapter`
> **推理**：`Dataset(val) → val_collate_fn → model.generate → batch_decode → JSON`
> **评测**：`JSON + val_cot.json → evaluation_suit → accuracy / BLEU·ROUGE·CIDEr / match-F1`
> 全链路围绕论文提出的 **Graph VQA** 数据结构展开，VLM 选型是可替换的（PaliGemma / Phi-4-MM），
> 扩展点集中在 `preparation/`、`collate_fn/`、`configs/` 三处。
