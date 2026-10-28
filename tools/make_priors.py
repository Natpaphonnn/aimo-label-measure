#!/usr/bin/env python3
"""Write per-(model_id, reasoning_effort) robust shares from the labelled rows of train-main-v2 to a submission's priors.json.
Usage: python tools/make_priors.py submissions/<name>/artifacts/priors.json"""
import json, pathlib, sys
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; HF = ROOT / "data/hf/train-main-v2"
df = pd.read_parquet(next(HF.glob("data/*.parquet"))); df = df[df.is_robust.notna()].copy(); df["y"] = df.is_robust.astype(bool)
pri = {f"{m}|{e or 'default'}": float(g.y.mean()) for (m, e), g in df.groupby(["model_id", "reasoning_effort"], dropna=False)}
pri["__global__"] = float(df.y.mean()); pri["__source__"] = f"train-main-v2@{(HF / '_revision.txt').read_text().strip()[:8]} n={len(df)}"
out = pathlib.Path(sys.argv[1]); out.write_text(json.dumps(pri, indent=1) + "\n"); print(json.dumps(pri, indent=1))
