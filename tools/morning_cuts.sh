#!/bin/bash
# After the night generation ends (gen_mlx.py gone, or 07:40), compute forced answers at cut points for all Qwen3.5-4B AMC records
# (Kaggle shards + Mac samples) with the MLX script (chunked prefill, no OOM). Mac must be on AC. Logs to runs/gen/Qwen_Qwen3.5-4B/morning_cuts.log
cd "$(dirname "$0")/.."
LOG=runs/gen/Qwen_Qwen3.5-4B/morning_cuts.log
while pgrep -f "selflabel/gen_mlx.py" >/dev/null && [ "$(date +%H%M)" -lt 0740 ]; do sleep 300; done
pkill -f "selflabel/gen_mlx.py" 2>/dev/null; sleep 20
echo "$(date) start cuts" >> $LOG
for pair in "kaggle_amc_originals_8k" "qwen35_amc_originals_8k"; do
  caffeinate -i .venv-gen/bin/python src/selflabel/force_at_cuts.py --model data/models/Qwen3.5-4B --records runs/gen/Qwen_Qwen3.5-4B/$pair.jsonl \
    --problems data/pool/night2_qwen35_amc.jsonl --cuts 512 1024 2048 3072 4096 8192 --batch 8 --out runs/gen/Qwen_Qwen3.5-4B/$pair.cuts.jsonl >> $LOG 2>&1
  echo "$(date) done $pair" >> $LOG
done
echo "$(date) ALL DONE" >> $LOG
