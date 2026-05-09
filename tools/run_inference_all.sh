#!/usr/bin/env bash
# Spawn 2 tmux windows running inference_batch in parallel:
#   DriveLM val × epoch-1 (GPU 0) and epoch-3 (GPU 1).
#
# Usage:  bash tools/run_inference_all.sh [CKPT_ROOT] [SESSION_NAME]
# Defaults:
#   CKPT_ROOT    = /root/autodl-tmp/pretrained/phi4/LOC-2026-05-08_23-22
#   SESSION_NAME = infer
#
# Attach with:  tmux attach -t infer
# Cycle windows: Ctrl-b n / Ctrl-b p (or Ctrl-b 0..5)
# Kill all:     tmux kill-session -t infer

set -euo pipefail

CKPT_ROOT="${1:-/root/autodl-tmp/pretrained/phi4/LOC-2026-05-08_23-22}"
SESSION="${2:-infer}"

PROJECT_DIR=/root/DriveVLMs_v3
PYTHON=/root/miniconda3/envs/DriveVLMs/bin/python
DRIVELM_VAL=${PROJECT_DIR}/data/DriveLM_nuScenes/split_448/val
DETECT_VAL=/root/autodl-tmp/nus_detection_qa/split_local/val
OUT_DIR=${PROJECT_DIR}/data/DriveLM_nuScenes/refs

mkdir -p "${OUT_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[!] tmux session '${SESSION}' already exists. Kill it first:"
  echo "    tmux kill-session -t ${SESSION}"
  exit 1
fi

# 2 jobs: (window-name, gpu, dataset, epoch, output_filename)
JOBS=(
  "e1_drivelm  0  ${DRIVELM_VAL}  1  infer_loc_drivelm_epoch1.json"
  "e3_drivelm  1  ${DRIVELM_VAL}  3  infer_loc_drivelm_epoch3.json"
)

# Create the session in detached mode (window 0 is replaced by first job)
tmux new-session -d -s "${SESSION}" -n "_init"
tmux send-keys  -t "${SESSION}:_init"  "echo 'spawning 2 inference jobs...'" C-m

for i in "${!JOBS[@]}"; do
  read -r NAME GPU DATA EPOCH OUTNAME <<<"${JOBS[$i]}"
  CKPT="${CKPT_ROOT}/epoch-${EPOCH}"
  OUT="${OUT_DIR}/${OUTNAME}"

  CMD="cd ${PROJECT_DIR} && \
CUDA_VISIBLE_DEVICES=${GPU} \
${PYTHON} tools/inference_batch.py \
  --data ${DATA} \
  --model ${CKPT} \
  --output ${OUT} \
  --batch_size 4 \
  --max_new_tokens 256 \
  --num_workers 2 2>&1 | tee ${OUT_DIR}/${NAME}.log"

  if [[ $i -eq 0 ]]; then
    # First job replaces the placeholder window
    tmux rename-window -t "${SESSION}:_init" "${NAME}"
  else
    tmux new-window -t "${SESSION}" -n "${NAME}"
  fi
  tmux send-keys -t "${SESSION}:${NAME}" "${CMD}" C-m
done

echo "Spawned 2 windows in tmux session '${SESSION}':"
tmux list-windows -t "${SESSION}" -F "  #{window_index}: #{window_name}"
echo ""
echo "Attach: tmux attach -t ${SESSION}"
echo "Logs:   tail -f ${OUT_DIR}/<name>.log"
