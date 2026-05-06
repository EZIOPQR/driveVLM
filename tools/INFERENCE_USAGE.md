# Inference & Evaluation 使用说明

针对 DriveLM-nuScenes + Phi-4-multimodal-instruct 流程。所有命令默认在仓库根目录 `/root/DriveVLMs_v3` 下、`DriveVLMs` conda 环境中运行。

## 通用前置

```bash
conda activate DriveVLMs
cd /root/DriveVLMs_v3
```

| 路径 | 说明 |
| --- | --- |
| `data/DriveLM_nuScenes/split/val` | HF Dataset 格式的 val split（5894 条） |
| `data/DriveLM_nuScenes/refs/val_cot.json` | 评测对齐用的 GT，按场景/帧组织 |
| `/root/autodl-tmp/models/Phi-4-multimodal-instruct` | Phi-4 base 模型，**必须存在**——`inference*.py` 用它加载 processor/generation_config |
| `/root/autodl-tmp/pretrained/phi4/<run_name>/final_model` | 微调后的权重，作为 `--model` 传入 |

## 1. `tools/inference.py` — 单样本推理（batch=1）

最朴素的版本，每次 forward 只跑一条。适合 debug 或显存极紧张的环境。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--data` | `data/DriveLM_nuScenes/split/val` | HF Dataset 路径 |
| `--collate_fn` | `drivelm_nus_phi4_collate_fn_val` | 注册表里的 val collate 名 |
| `--model` | base 模型路径 | 推理用的权重，可指向 fine-tune checkpoint |
| `--output` | `data/DriveLM_nuScenes/refs/infer_results_21-49.json` | 结果 JSON，**增量写盘**（崩了不丢） |
| `--device` | `cuda` | |
| `--limit N` | `None` | 只跑前 N 条，做 smoke test 用 |
| `--use_optical_flow` | off | 需 `--flow_root`；输入变为 6 RGB + 6 flow 图像 |
| `--flow_root` | `""` | 光流 `.npz` 根目录 |
| `--flow_scale` | `32` | 与训练一致 |

示例：

```bash
# 全集
python tools/inference.py \
    --model /root/autodl-tmp/pretrained/phi4/<run_name>/final_model \
    --output data/DriveLM_nuScenes/refs/infer_full.json

# 20 条 smoke test
python tools/inference.py \
    --model /root/autodl-tmp/pretrained/phi4/<run_name>/final_model \
    --output data/DriveLM_nuScenes/refs/infer_smoke20.json \
    --limit 20
```

## 2. `tools/inference_batch.py` — 批量推理（推荐）

支持任意 `--batch_size`，配合 `--max_new_tokens` 可以显著加速。底层依赖修复后的 `drivelm_nus_phi4_collate_fn_val`（已支持 batch>1，processor 自动左 padding）。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--data` / `--collate_fn` / `--model` / `--device` / `--limit` / `--output` | 同上 | |
| `--batch_size` | `4` | bf16 + 12 图（6×448 RGB + 6×14 flow）：B=4 约 24GB 显存，B=8 需要更大显存 |
| `--max_new_tokens` | `256` | DriveLM 答案最长 ~80 token，256 足够；调小可继续提速 |
| `--num_workers` | `8` | DataLoader 进程数 |
| `--use_optical_flow` | off | 与训练一致时打开；需 `--flow_root` |
| `--flow_root` | `""` | `flow/CAM/*.npz` 根目录 |
| `--flow_scale` | `32` | 与训练 `flow_scale` 一致 |

示例：

```bash
# 全集 batched
python tools/inference_batch.py \
    --model /root/autodl-tmp/pretrained/phi4/<run_name>/final_model \
    --output data/DriveLM_nuScenes/refs/infer_batch_full.json \
    --batch_size 4 --max_new_tokens 256

# 20 条 smoke
python tools/inference_batch.py \
    --model /root/autodl-tmp/pretrained/phi4/<run_name>/final_model \
    --output data/DriveLM_nuScenes/refs/infer_batch_smoke.json \
    --batch_size 4 --limit 20 --max_new_tokens 256
```

输出 schema 与 `inference.py` 完全一致：`[{"id": str, "question": str, "answer": str}, ...]`，可直接喂给 `evaluation.py`。

## 3. `tools/evaluation.py` — 评测

把推理结果 JSON 与 GT 对比，按 QA tag 分桶算指标。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--src` | `data/DriveLM_nuScenes/refs/infer_results.json` | 推理结果（`inference*.py` 的输出） |
| `--tgt` | `data/DriveLM_nuScenes/refs/val_cot.json` | GT 文件 |

示例：

```bash
python tools/evaluation.py \
    --src data/DriveLM_nuScenes/refs/infer_batch_full.json \
    --tgt data/DriveLM_nuScenes/refs/val_cot.json
```

桶与指标：

| tag | 含义 | 指标 |
| --- | --- | --- |
| 0 | MCQ / 是非题 | exact-match accuracy |
| 1 | chatGPT 评分（占位，未启用） | — |
| 2 | perception 自由生成 | BLEU-1~4 / ROUGE-L / CIDEr |
| 3 | prediction `<c_i,CAM,x,y>` | match-F1（L1 距离 < 16 视为匹配） |

输出会先打印每个桶的 size，再分别打印三类原始分数和归一化后的合成分。空桶会被跳过并提示，不会再像旧版本那样除零崩溃。

> 已知 latent bug：`evaluation.py` line ~178 的 `pred_file[idx]["answer"][0]` 在 `answer` 是字符串时取的是**首字符**——这会让 BLEU/ROUGE/CIDEr 失真。当前 `inference*.py` 的输出 schema 与之兼容（即同样的旧结果对比），如果以后想拿到真实的 language 指标，需要把这个 `[0]` 拿掉。

## 4. `tools/visualize_eval.py` — 离线 HTML 报告（per-sample review）

`evaluation.py` 只给汇总指标。当全集错误样本上千条时，靠分数没法定位"模型到底错在哪"。`visualize_eval.py` 把每条 prediction 渲染成一个卡片，带 6 张相机原图、Q/GT/Pred、单条分数，**离线 HTML，无需起服务**，把单个 .html 拷到本地数据集目录里浏览器打开即可。

**前提**：你的本地机器上需要有 `DriveLM_nuScenes/` 数据集副本，目录结构 `DriveLM_nuScenes/nuscenes/samples/CAM_*/*.jpg`。脚本不生成缩略图，HTML 中 `<img>` 直接指向 `nuscenes/samples/CAM_*/...jpg` 这种相对路径，因此 **HTML 必须放在 `DriveLM_nuScenes/` 这一级目录下**才能解析到图。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--src` | (必填) | 推理结果 JSON |
| `--gt` | `data/DriveLM_nuScenes/refs/val_cot.json` | GT |
| `--out` | `viz_eval/` | 输出目录，会写入 `{out}/{src_basename}.html` |
| `--limit N` | `None` | 只处理前 N 条对齐样本，用于 debug |

示例：

```bash
# 单个 ckpt
python tools/visualize_eval.py \
    --src data/DriveLM_nuScenes/refs/infer_epoch3.json \
    --out viz_eval/

# 4 个 ckpt 全跑
for m in baseline epoch1 epoch2 epoch3; do
  python tools/visualize_eval.py \
      --src data/DriveLM_nuScenes/refs/infer_${m}.json \
      --out viz_eval/
done
```

输出：

```
viz_eval/
├── infer_baseline.html
├── infer_epoch1.html
├── infer_epoch2.html
└── infer_epoch3.html      # 单文件，CSS/JS/数据全内嵌（~10 MB）
```

本地查看流程：

```bash
# 1. 把 .html 拷到本地的 DriveLM_nuScenes/ 数据集目录
scp autodl:/root/DriveVLMs_v3/viz_eval/infer_epoch3.html ~/DriveLM_nuScenes/

# 2. 本地浏览器打开
open ~/DriveLM_nuScenes/infer_epoch3.html
```

打分逻辑（与 `evaluation.py` 一致 + 单条扩展）：

| tag | 单条 score | 颜色桶 |
| --- | --- | --- |
| 0 (MCQ / 是非) | exact match → 1.0 / 0.0 | 绿 / 红 |
| 2 (perception 自由生成) | 单句 ROUGE-L F1（LCS） | ≥0.8 绿 / ≥0.3 黄 / 红 |
| 3 (`<c_i,CAM,x,y>` 坐标) | match-F1（L1<16，逻辑同 `evaluation.py`） | 同上 |
| 1 (planning) | **跳过**——`evaluation.py` 也没 auto score | — |

HTML 内功能（纯 vanilla JS，不依赖 CDN）：

- **顶部 sticky toolbar**：tag 过滤 / score 过滤 / 关键词搜索（Q+GT+Pred）/ 排序（默认 score 升序，最差先看）/ 分页（50/页）
- **卡片折叠**：头部一行预览 GT/Pred；点击展开看完整内容 + 6 张相机图 + 标签高亮
- **`<c1,CAM_FRONT,...>` 标签高亮**：GT 和 Pred 中所有对象引用 tag 都加蓝色 chip
- **tag=2 diff**：GT 中 Pred 没有的 token 标灰删除线；Pred 中 GT 没有的标黄
- **tag=3 坐标匹配表**：明确列出 matched / missed (gt 漏的) / extra (pred 多的)，含 L1 距离
- **图片懒加载**：浏览器只在卡片展开 + 图片进入视口时才请求原图，首次打开秒级响应

性能：脚本只读 JSON 输出 HTML，秒级。单 HTML ~5-15 MB（含 3449 条 records 全 JSON）；浏览器分页只渲染当前页 50 个 DOM 节点。

## 5. 一条龙

```bash
python tools/inference_batch.py \
    --model /root/autodl-tmp/pretrained/phi4/<run_name>/final_model \
    --output data/DriveLM_nuScenes/refs/run.json \
    --batch_size 4 --max_new_tokens 256 \
&& \
python tools/evaluation.py \
    --src data/DriveLM_nuScenes/refs/run.json \
    --tgt data/DriveLM_nuScenes/refs/val_cot.json
```

## 6. 光流侧车（12 图输入，可选）

先用 `sweeps/` 离线生成与 `samples/CAM/*.jpg` 同主文件名的 `flow/CAM/<stem>.npz`（字段 `u`,`v`，14×14）：

```bash
python tools/create_data/compute_flow_from_sweeps.py \\
    data/DriveLM_nuScenes/QA_dataset_nus/v1_1_train_nus.json \\
    --nuscenes-root /path/to/nuscenes \\
    --flow-root data/DriveLM_nuScenes/flow \\
    --out-size 14
```

- **训练**：在 `configs/phi4/phi4_drivelm_1xb1-lora_config.py` 中设 `use_optical_flow=True`，并配置 `flow_root` 与 `flow_scale`。此时每个样本输入为 12 图：前 6 张 RGB，后 6 张 flow。
- **推理**：`inference.py` / `inference_batch.py` 增加 `--use_optical_flow --flow_root <dir> [--flow_scale 32]`；不再需要扩展 SigLIP 首层卷积通道。

示例：

```bash
python tools/inference_batch.py \\
    --model /path/to/final_model \\
    --use_optical_flow --flow_root data/DriveLM_nuScenes/flow \\
    --flow_scale 32 --batch_size 4
```

## 7. 排错速查

| 现象 | 原因 / 处理 |
| --- | --- |
| `ZeroDivisionError` in `eval_match` / `eval_acc` | 该桶为空（小 `--limit` 容易出现）。已在 `evaluation.py` 改成跳过并提示，重新拉一下代码即可。 |
| `Some weights ... newly initialized: audio_embed.* / *.lora.speech.*` | 预期。`phi4_preparation.py` 训练前把音频塔和 speech LoRA 删了，checkpoint 里没存；推理 `input_mode=VISION` 不会用到。 |
| OOM | 调小 `--batch_size` 或 `--max_new_tokens`；或换回 `inference.py`（batch=1）。 |
| 多次跑同一条样本答案不一样 | `GenerationConfig.from_pretrained(base_model)` 默认 `do_sample=True`。要确定性，可在脚本里临时 `generation_config.do_sample = False`。 |
| `flash_attention_2` 报错 | 卡不支持（需要 Ampere 及以上）。改 `inference*.py` 里 `_attn_implementation='eager'`。 |
