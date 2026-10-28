#!/usr/bin/env python3
"""Build the leakage-filtered self-labelling problem pool.

Sources: byeolki/aimo-interp data/problems.jsonl (MIT; integer-answer problems from AI-MO/aimo-validation-aime,
AI-MO/aimo-validation-amc and MathArena sets) plus the AI-MO/aimo-validation-aime rows API for year tags.
Exclusions (organizer leakage advice 2026-10-09 + our rule): AIME 2024, 2025, 2026; any statement in
train-main-v2, val-sample or aimo-interp-challenge-sample-full (normalized text match).
Writes data/pool/problems.jsonl and data/pool/SUMMARY.md.
"""
import collections, json, pathlib, re, urllib.parse, urllib.request
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
BY = ROOT / "external/byeolki_aimo-interp/data"
OUT = ROOT / "data/pool"; OUT.mkdir(parents=True, exist_ok=True)
import sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "analysis")); from datapaths import ROOT as _R, need  # noqa: E402
need(BY / "problems.jsonl")
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()

def rows_api(dataset, split="train"):
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": split, "offset": offset, "length": 100})
        with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=60) as r:
            page = [x["row"] for x in json.load(r)["rows"]]
        rows += page
        if len(page) < 100: return rows
        offset += 100

aime = rows_api("AI-MO/aimo-validation-aime")
year_of = {norm(r["problem"]): re.search(r"(20\d\d)", r.get("url", "")).group(1) if re.search(r"(20\d\d)", r.get("url", "")) else "?" for r in aime}
official = set()
for name, col in [("train-main-v2", "problem"), ("val-sample", "original_problem"), ("aimo-interp-challenge-sample-full", "original_problem")]:
    df = pd.concat([pd.read_parquet(f) for f in (ROOT / "data/hf" / name / "data").glob("*.parquet")])
    official |= {norm(x) for x in df[col]}
probs = [json.loads(l) for l in (BY / "problems.jsonl").read_text().splitlines() if l.strip()]
small_ids = {json.loads(l)["problem_id"] for l in (BY / "problems_small.jsonl").read_text().splitlines() if l.strip()}
manual = {json.loads(l)["problem_id"]: json.loads(l)["reason"] for l in (OUT / "exclusions.jsonl").read_text().splitlines() if l.strip()} if (OUT / "exclusions.jsonl").exists() else {}
kept, excluded = [], collections.Counter()
for p in probs:
    src = p["source"]; year = year_of.get(norm(p["problem"]), "")
    tag = f"{src}" + (f"-{year}" if src == "aimo-validation-aime" else "")
    if src in {"aime_2025", "aime_2026"} or (src == "aimo-validation-aime" and year == "2024"):
        excluded[tag] += 1; continue
    if p["problem_id"] in manual:
        excluded["manual (data/pool/exclusions.jsonl)"] += 1; continue
    if norm(p["problem"]) in official:
        excluded["official-overlap"] += 1; continue
    kept.append({**p, "source_tag": tag, "in_byeolki_small": p["problem_id"] in small_ids})
(OUT / "problems.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in kept))
c = collections.Counter(p["source_tag"] for p in kept); cs = collections.Counter(p["source_tag"] for p in kept if p["in_byeolki_small"])
md = ["# Problem pool", "", f"Kept {len(kept)} of {len(probs)} problems; excluded: {dict(excluded)}", "", "| source | kept | of which in byeolki's 150-problem labelled subset |", "|---|---|---|"]
md += [f"| {k} | {v} | {cs.get(k,0)} |" for k, v in sorted(c.items())]
md += ["", "Answer range: " + f"min {min(p['answer'] for p in kept)}, max {max(p['answer'] for p in kept)}", f"Problem length (chars): median {int(pd.Series([len(p['problem']) for p in kept]).median())}, max {max(len(p['problem']) for p in kept)}"]
(OUT / "SUMMARY.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
