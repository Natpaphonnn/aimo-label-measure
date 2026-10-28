#!/usr/bin/env bash
# Parallel resumable downloads into data/models/ (link ~125 KB/s per stream, ~350 KB/s with 8-12 streams). Order = priority.
set -uo pipefail; cd "$(dirname "$0")/.."
PY=.venv/bin/python
# 1) Qwen3.5-4B bf16 (exact competition weights), pinned revision
$PY tools/pdl.py --repo Qwen/Qwen3.5-4B --rev 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --dest data/models/Qwen3.5-4B --streams 12 --chunk-mb 32 \
  --files config.json chat_template.jinja tokenizer.json tokenizer_config.json vocab.json merges.txt model.safetensors.index.json preprocessor_config.json model.safetensors-00001-of-00002.safetensors model.safetensors-00002-of-00002.safetensors
# 2) 8-bit MLX builds for the Mac (documented deviation from bf16): DeepSeek then Olmo
$PY tools/pdl.py --repo mlx-community/DeepSeek-R1-0528-Qwen3-8B-8bit --dest data/models/DeepSeek-R1-0528-Qwen3-8B-8bit --streams 12 --chunk-mb 32 \
  --files config.json tokenizer.json tokenizer_config.json special_tokens_map.json model.safetensors.index.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors
$PY tools/pdl.py --repo mlx-community/Olmo-3-7B-Think-8bit --dest data/models/Olmo-3-7B-Think-8bit --streams 12 --chunk-mb 32 \
  --files config.json tokenizer.json tokenizer_config.json special_tokens_map.json model.safetensors.index.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors
echo "ALL MODELS DONE $(date)"
