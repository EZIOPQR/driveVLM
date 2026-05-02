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
| `--batch_size` | `4` | bf16 + 6×448 图：B=4 约 24GB 显存，B=8 需要 40GB+ |
| `--max_new_tokens` | `256` | DriveLM 答案最长 ~80 token，256 足够；调小可继续提速 |
| `--num_workers` | `8` | DataLoader 进程数 |

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

## 4. 一条龙

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

## 5. 排错速查

| 现象 | 原因 / 处理 |
| --- | --- |
| `ZeroDivisionError` in `eval_match` / `eval_acc` | 该桶为空（小 `--limit` 容易出现）。已在 `evaluation.py` 改成跳过并提示，重新拉一下代码即可。 |
| `Some weights ... newly initialized: audio_embed.* / *.lora.speech.*` | 预期。`phi4_preparation.py` 训练前把音频塔和 speech LoRA 删了，checkpoint 里没存；推理 `input_mode=VISION` 不会用到。 |
| OOM | 调小 `--batch_size` 或 `--max_new_tokens`；或换回 `inference.py`（batch=1）。 |
| 多次跑同一条样本答案不一样 | `GenerationConfig.from_pretrained(base_model)` 默认 `do_sample=True`。要确定性，可在脚本里临时 `generation_config.do_sample = False`。 |
| `flash_attention_2` 报错 | 卡不支持（需要 Ampere 及以上）。改 `inference*.py` 里 `_attn_implementation='eager'`。 |
