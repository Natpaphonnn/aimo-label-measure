#!/usr/bin/env bash
# Night run for the test-time wave (DECISIONS #44): Qwen3.5-4B, 109 problems (73 byeolki-labelled outside our AMC set + 36
# train-main-v2 statements) x 8 samples at the test-time cut 3072, MLX bf16 batch 8, token_ids + forced answers kept; resumable.
# Starts between 21:00 and 05:00 (NOW=1 skips the hour check) when power_ok; stops at 07:30, on battery below 20%, or when runs/STOP exists.
cd "$(dirname "$0")/.."
# Power rule (the author 2026-10-22): AC, or battery >= 40% to start; mem_guard stops our jobs on battery below 20%.
power_ok() { b=$(pmset -g batt); echo "$b" | grep -q "AC Power" && return 0; [ "$(echo "$b" | grep -o '[0-9]*%' | head -1 | tr -d %)" -ge 40 ]; }
# Before the night run, whenever on AC: the 344 AMC states that the 16:50 run lost (killed by hand at a 20 GB footprint).
# tools/mem_guard.sh runs alongside and kills any of our model processes above a 14 GB footprint or below 10% free memory.
M=data/models/Qwen3.5-4B; OUT=runs/gen/Qwen_Qwen3.5-4B/tt_wave_3072.jsonl
pgrep -f tools/mem_guard.sh >/dev/null || { nohup tools/mem_guard.sh 14 10 20 >> runs/mem_guard.log 2>&1 & }
while true; do
  [ -f runs/STOP ] && { echo "$(date) STOP file present; launcher exits"; exit 0; }
  if [ ! -f runs/pilot/qwen_rest_states.npz ] && [ ! -f runs/pilot/qwen_rest_states.tried ] && power_ok; then
    touch runs/pilot/qwen_rest_states.tried   # one attempt only (a mem_guard kill must not loop)
    echo "$(date) AMC rest states (344 samples)"
    caffeinate -i .venv/bin/python src/features/trace_states.py --model $M --records runs/pilot/qwen_amc_rest_cut3072.jsonl \
      --problems data/pool/night2_qwen35_amc.jsonl --cuts 2048 3072 --layers 0.25 0.5 0.75 --out runs/pilot/qwen_rest_states --max-len 3072
    echo "$(date) AMC rest states exited with $?"; continue
  fi
  h=$(date +%H); night=0; { [ "$h" -ge 21 ] || [ "$h" -lt 5 ] || [ "${NOW:-0}" = 1 ]; } && night=1
  if [ $night = 1 ] && power_ok; then
    echo "$(date) launching tt-wave run"
    caffeinate -i .venv-gen/bin/python src/selflabel/gen_mlx.py --model $M --problems data/pool/tt_wave_problems.jsonl \
      --out $OUT --n-samples 8 --max-tokens 3072 --batch 8 --min-battery 20 --until 07:30
    echo "$(date) run exited with $?"
    if [ "$(date +%H)" -ge 7 ] && [ "$(date +%H)" -lt 20 ]; then
      # morning: teacher-forced states of whatever was generated (cut 3072; ~15 s per sample on MPS), only on AC
      if power_ok && [ ! -f runs/STOP ]; then
        echo "$(date) states for $OUT"; mkdir -p runs/features/Qwen_Qwen3.5-4B
        caffeinate -i .venv/bin/python src/features/trace_states.py --model $M --records $OUT --problems data/pool/tt_wave_problems.jsonl \
          --cuts 3072 --layers 0.25 0.5 0.75 --out runs/features/Qwen_Qwen3.5-4B/tt_wave_states --max-len 3072
        echo "$(date) states exited with $?"
      fi
      echo "$(date) daytime: launcher exits"; exit 0
    fi
    sleep 600
  fi
  sleep 300
done
