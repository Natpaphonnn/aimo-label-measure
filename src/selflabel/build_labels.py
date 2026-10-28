#!/usr/bin/env python3
"""Aggregate per-sample records into per-(model, problem) solve rates, family drops and labels.

Correctness mode: 'natural' (only answers given before the cap count), 'forced' (forced answers count for
capped samples), matching the organizers' FAQ (natural) and byeolki (forced) respectively.
Label rules: 'band' (robust iff max_drop <= 0.10, non-robust iff >= 0.25, else None) and 'any' (non-robust iff any
family accuracy < base accuracy). Base-accuracy exclusion: cases with base < min_base get label None (reported).
"""
from __future__ import annotations
import argparse, collections, json, pathlib
import pandas as pd

def load_records(paths: list[pathlib.Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l); r.pop("text", None); rows.append(r)
    return pd.DataFrame(rows)

def aggregate(df: pd.DataFrame, mode: str = "natural", min_base: float = 0.0, families: list[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    if mode == "forced":
        df["ok"] = df.apply(lambda r: bool(r["forced_correct"]) if r["finish_reason"] == "length" and r.get("forced_correct") is not None else bool(r["correct"]), axis=1)
    else:
        df["ok"] = df["correct"].astype(bool)
    g = df.groupby(["model_id", "problem_id", "family"]).agg(acc=("ok", "mean"), n=("ok", "size"), trunc=("finish_reason", lambda s: (s == "length").mean()), tokens=("n_tokens", "mean")).reset_index()
    out = []
    for (m, p), grp in g.groupby(["model_id", "problem_id"]):
        fam = {r.family: r for r in grp.itertuples()}
        if "original" not in fam: continue
        base = fam["original"].acc; n0 = fam["original"].n
        row = {"model_id": m, "problem_id": p, "base": base, "n_base": n0, "trunc_rate": fam["original"].trunc, "mean_tokens": fam["original"].tokens}
        fams = [f for f in (families or sorted(fam)) if f != "original" and f in fam]
        drops = {f: base - fam[f].acc for f in fams}
        row.update({f"acc_{f}": fam[f].acc for f in fams}); row["n_families"] = len(fams)
        row["max_drop"] = max(drops.values()) if drops else None
        row["worst_family"] = max(drops, key=drops.get) if drops else None
        if drops and base >= min_base:
            md = row["max_drop"]; row["label_band"] = True if md <= 0.10 + 1e-9 else (False if md >= 0.25 - 1e-9 else None)
            row["label_any"] = not any(fam[f].acc < base - 1e-9 for f in fams)
        else:
            row["label_band"] = None; row["label_any"] = None
        row["excluded_low_base"] = bool(drops) and base < min_base
        out.append(row)
    return pd.DataFrame(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("records", nargs="+", type=pathlib.Path); ap.add_argument("--mode", default="natural", choices=["natural", "forced"])
    ap.add_argument("--min-base", type=float, default=0.0); ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args(); df = load_records(a.records); agg = aggregate(df, a.mode, a.min_base)
    pd.set_option("display.width", 220)
    for m, grp in agg.groupby("model_id"):
        lab = grp.dropna(subset=["label_band"])
        print(f"{m}: cases {len(grp)}, base==0 {(grp.base==0).sum()}, 0<base<1 {((grp.base>0)&(grp.base<1)).sum()}, base==1 {(grp.base==1).sum()}, "
              f"labelled(band) {len(lab)} robust {int(lab.label_band.sum())} nonrobust {int((~lab.label_band.astype(bool)).sum())}, label_any robust {int(grp.label_any.fillna(False).sum())}")
    if a.out: a.out.parent.mkdir(parents=True, exist_ok=True); agg.to_csv(a.out, index=False); print("wrote", a.out)
