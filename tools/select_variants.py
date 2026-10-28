#!/usr/bin/env python3
"""Second wave of self-labelling: emit the perturbed variants of the problems a model SOLVES (DECISIONS #18, #36).

Reads per-sample records (gen_mlx.py / Kaggle kernel output, family == original), computes the per-problem base solve rate
(mode natural or forced, as build_labels.py) and writes a problems jsonl for tools/kaggle_push_gen.py / gen_mlx.py containing
the variants (data/variants/*.jsonl: rephrase, domain, expert_no_solution) of every problem with base >= --min-base and at
least --min-samples samples. The 60-problem AMC pool is covered by amc_b2-b5 + pilot_v1 (59/60; 3 variants per problem). Fields: problem_id, family, variant, problem (the variant text), answer (NaN for no-solution).
Usage: python tools/select_variants.py --records runs/kaggle/kx-gen-qwen35-amc-*/out3/*.jsonl runs/gen/Qwen_Qwen3.5-4B/qwen35_amc_originals_8k.jsonl \
          --variants data/variants/amc_b2.jsonl data/variants/amc_b3.jsonl data/variants/amc_b4.jsonl data/variants/amc_b5.jsonl data/variants/pilot_v1.jsonl \
          --out data/pool/wave2_qwen35_amc_variants.jsonl [--families rephrase domain expert_no_solution] [--mode forced]
"""
from __future__ import annotations
import argparse, collections, json, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--records", type=pathlib.Path, nargs="+", required=True); ap.add_argument("--variants", type=pathlib.Path, nargs="+", required=True)
ap.add_argument("--out", type=pathlib.Path, required=True); ap.add_argument("--min-base", type=float, default=0.5); ap.add_argument("--min-samples", type=int, default=4)
ap.add_argument("--mode", choices=["natural", "forced"], default="forced", help="forced: forced answers count for capped samples (byeolki convention)")
ap.add_argument("--families", nargs="*", default=None); ap.add_argument("--exclude", type=pathlib.Path, nargs="*", default=[], help="variant files to drop (e.g. data/variants/excluded_variants.jsonl)")
a = ap.parse_args()

ok = collections.defaultdict(list)
for f in a.records:
    for l in f.read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        if r.get("family", "original") != "original": continue
        c = bool(r["correct"])
        if a.mode == "forced" and r.get("finish_reason") == "length" and r.get("forced_correct") is not None: c = bool(r["forced_correct"])
        ok[r["problem_id"]].append(c)
base = {p: sum(v) / len(v) for p, v in ok.items() if len(v) >= a.min_samples}
solved = {p for p, b in base.items() if b >= a.min_base}
excluded = {(json.loads(l)["problem_id"], json.loads(l)["family"], int(json.loads(l).get("variant", 0))) for f in a.exclude for l in f.read_text().splitlines() if l.strip()}
rows, seen = [], set()
for f in a.variants:
    for l in f.read_text().splitlines():
        if not l.strip(): continue
        v = json.loads(l); key = (v["problem_id"], v["family"], int(v.get("variant", 0)))
        if v["problem_id"] not in solved or key in seen or key in excluded: continue
        if a.families and v["family"] not in a.families: continue
        seen.add(key); rows.append({"problem_id": v["problem_id"], "family": v["family"], "variant": int(v.get("variant", 0)), "problem": v["text"], "answer": v["answer"], "source": str(f)})
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
fam = collections.Counter(r["family"] for r in rows)
print(f"problems with >= {a.min_samples} samples: {len(base)}; solved (base >= {a.min_base}, {a.mode}): {len(solved)}; "
      f"variants written: {len(rows)} {dict(fam)} -> {a.out}")
print("base rates:", " ".join(f"{p[:6]}={b:.2f}" for p, b in sorted(base.items(), key=lambda x: -x[1])))
