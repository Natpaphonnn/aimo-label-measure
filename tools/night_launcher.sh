#!/usr/bin/env bash
# Starts the night generation (night-2 recipe since 2026-10-17: 60 AMC originals, sample indices 8-11 (Kaggle does 0-7), 8192 cap, bf16 batch 8, resumes the night-1 file; DECISIONS #36) only when (a) Qwen3.5-4B weights are complete, (b) local time is 21:00-04:59, (c) on AC power.
# Gives up (does nothing) if runs/STOP exists. The run itself stops at 07:30, on battery, or on runs/STOP.
cd "$(dirname "$0")/.."
M=data/models/Qwen3.5-4B
while true; do
  [ -f runs/STOP ] && { echo "$(date) STOP file present; launcher exits"; exit 0; }
  ready=1; for f in model.safetensors-00001-of-00002.safetensors model.safetensors-00002-of-00002.safetensors; do
    { [ -f "$M/$f" ] && [ ! -d "$M/$f.parts" ]; } || ready=0; done
  h=$(date +%H); night=0; { [ "$h" -ge 21 ] || [ "$h" -lt 5 ]; } && night=1
  ac=0; pmset -g batt | grep -q "AC Power" && ac=1
  if [ $ready = 1 ] && [ $night = 1 ] && [ $ac = 1 ]; then
    echo "$(date) launching night-1 run"
    caffeinate -i .venv-gen/bin/python src/selflabel/gen_mlx.py --model $M --problems data/pool/night2_qwen35_amc.jsonl \
      --out runs/gen/Qwen_Qwen3.5-4B/qwen35_amc_originals_8k.jsonl --n-samples 12 --sample-start 8 --max-tokens 8192 --batch 8 --require-ac --until 07:30
    echo "$(date) run exited with $? (re-arming; delete the launcher process or create runs/STOP to end)"; sleep 600
  fi
  sleep 300
done
