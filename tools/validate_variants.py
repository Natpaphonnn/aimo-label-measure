#!/usr/bin/env python3
"""Validate a variants jsonl batch against data/pool/problems.jsonl: schema, gold answers, families, NaN for no-solution,
mechanism tags, LaTeX brace/dollar balance, figure code preserved, number-diff warnings (digits present in the variant but not
the original are suspicious; missing digits are usually words). Exit code 1 on hard failures."""
import json, re, sys, collections, pathlib
KEYS = {"problem_id", "family", "variant", "method", "text", "answer", "author_notes", "verification", "confidence"}
FAM = {"rephrase", "domain", "expert_no_solution", "expert_perturbations", "distract", "rename", "typos"}
MECH = {"missing_quantity", "contradiction", "missing_selector", "nonexistence", "missing_normalisation"}
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "analysis")); from datapaths import ROOT as _R, need  # noqa: E402
need(_R / "data/pool/problems.jsonl")
pool = {json.loads(l)["problem_id"]: json.loads(l) for l in open("data/pool/problems.jsonl") if l.strip()}
hard, warn = [], []
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    for i, r in enumerate(rows):
        tag = f"{pathlib.Path(path).name}:{i+1} {r.get('problem_id','?')[:8]} {r.get('family','?')}"
        if not KEYS <= set(r): hard.append(f"{tag}: missing keys {KEYS - set(r)}"); continue
        if r["family"] not in FAM: hard.append(f"{tag}: bad family")
        if r["problem_id"] not in pool: hard.append(f"{tag}: unknown problem"); continue
        orig = pool[r["problem_id"]]
        if r["family"] == "expert_no_solution":
            if r["answer"] != "NaN": hard.append(f"{tag}: no-solution answer must be NaN")
            if r.get("mechanism") not in MECH: hard.append(f"{tag}: missing/bad mechanism tag")
        elif r["answer"] != orig["answer"]: hard.append(f"{tag}: answer {r['answer']} != gold {orig['answer']}")
        t = r["text"]
        if t.count("$") % 2: warn.append(f"{tag}: odd number of $")
        if t.count("{") != t.count("}"): warn.append(f"{tag}: unbalanced braces")
        asy = re.findall(r"\[asy\].*?\[/asy\]", orig["problem"], re.S)
        if asy and r["family"] != "expert_no_solution" and any(a not in t for a in asy): hard.append(f"{tag}: figure code altered or dropped")
        if r["family"] in ("rephrase", "domain", "expert_perturbations"):
            o, v = collections.Counter(re.findall(r"\d+(?:\.\d+)?", orig["problem"])), collections.Counter(re.findall(r"\d+(?:\.\d+)?", t))
            extra = {k: c for k, c in (v - o).items() if int(float(k)) > 1}
            if extra: warn.append(f"{tag}: numbers not in original {extra}")
        if not (0 <= float(r["confidence"]) <= 1): hard.append(f"{tag}: confidence out of range")
        if len(t) < 0.5 * len(orig["problem"]) and r["family"] != "expert_no_solution": warn.append(f"{tag}: variant much shorter than original")
    print(f"{path}: {len(rows)} rows, families {dict(collections.Counter(r['family'] for r in rows))}")
for w in warn: print("WARN", w)
for h in hard: print("FAIL", h)
sys.exit(1 if hard else 0)
