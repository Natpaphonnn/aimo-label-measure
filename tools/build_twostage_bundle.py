#!/usr/bin/env python3
"""Build a two-stage submission (DECISIONS #44/#47) from every available sample: copy submissions/hybrid-v1 to
submissions/<name>, fit the artifact with analysis/wb_two_stage.export_artifact (stage 1 refit on all gold samples, stage 2 on
cross-fitted scores), check the bundle's numpy decision against sklearn, and set the deployment K.

Usage: python tools/build_twostage_bundle.py --name twostage-wb-c8 --arm wb --k 8 --target consistent [--models qwen skywork deepseek]
Then: python tools/build_submission.py submissions/<name>
"""
from __future__ import annotations
import argparse, json, pathlib, shutil, sys
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from run_wb import BLOCKS, collect  # noqa: E402
from wb_two_stage import check_artifact, consistency, export_artifact  # noqa: E402


def layers_of(name: str) -> list[int]:
    for _, _, st, _, _ in BLOCKS[name]:
        for s in (st if isinstance(st, list) else [st]):
            f = pathlib.Path(str(s) + ".npz")
            if f.exists(): return [int(x) for x in np.load(f)["layers"]]
    raise SystemExit(f"no states for {name}")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--arm", choices=["wb", "bb", "hybrid"], required=True)
    ap.add_argument("--k", type=int, required=True); ap.add_argument("--target", choices=["correct", "consistent"], default="correct")
    ap.add_argument("--models", nargs="+", default=["qwen", "skywork", "deepseek"]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balanced", action="store_true", help="stage 2 with class_weight=balanced (DECISIONS #50)")
    a = ap.parse_args()
    df = collect(a.models).reset_index(drop=True)
    df["t"] = df.correct.map(lambda v: None if v is None or v != v else float(v)) if a.target == "correct" else consistency(df)
    mids = dict(zip(df.groupby("model_id").size().index, [None] * df.model_id.nunique()))
    layers = {}
    for name in a.models:
        mid = BLOCKS[name][0][0]
        if mid in mids: layers[mid] = layers_of(name)
    labels = pd.read_csv(ROOT / "data/labels/case_labels.csv")
    run = {"k": a.k, "target": a.target, "models": a.models, "n_samples": int(len(df)), "balanced": a.balanced}
    art = export_artifact(df, labels, a.arm, a.k, layers, a.seed, run, "balanced" if a.balanced else None)
    worst = check_artifact(art, df, a.k) if a.arm != "bb" else 0.0
    print(f"stage 1 models: {sorted(art['stage1'])}; stage 2 arms: { {k: v['n_cases'] for k, v in art['stage2'].items()} }; bundle vs sklearn max |dp| = {worst:.2e}")
    assert worst < 1e-9
    dst = ROOT / "submissions" / a.name
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(ROOT / "submissions/hybrid-v1", dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "src/runtime", dst / "runtime", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))   # as tests/test_hybrid_bundle.py
    (dst / "artifacts/hybrid_model.json").write_text(json.dumps(art))
    print(f"wrote {dst}/artifacts/hybrid_model.json ({(dst / 'artifacts/hybrid_model.json').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
