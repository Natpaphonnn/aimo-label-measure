#!/usr/bin/env python3
"""Import byeolki/aimo-interp generations (MIT) into our per-sample record format (no text available).
Writes runs/gen/<safe_model>/byeolki_8k.jsonl for problems in our leakage-filtered pool. Budget: 8192 reasoning
tokens, forced answer on truncation (their 'answers' are already forced-or-natural), T=1.0 top_k=40 top_p=0.95, vLLM."""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
BY = ROOT / "external/byeolki_aimo-interp/data"
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "analysis")); from datapaths import ROOT as _R, need  # noqa: E402
need(_R / "data/pool/problems.jsonl")
pool = {json.loads(l)["problem_id"]: json.loads(l) for l in (ROOT / "data/pool/problems.jsonl").read_text().splitlines() if l.strip()}
for f in sorted((BY / "generations").glob("*.jsonl")):
    model = f.stem.replace("_", "/", 1); out = ROOT / "runs/gen" / f.stem / "byeolki_8k.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as w:
        for l in f.read_text().splitlines():
            if not l.strip(): continue
            r = json.loads(l)
            if r["problem_id"] not in pool: continue
            for i, (ans, ok, ln, tr) in enumerate(zip(r["answers"], r["correct"], r["lengths"], r["truncated"])):
                w.write(json.dumps({"model_id": model, "problem_id": r["problem_id"], "family": r["family"], "variant": r["variant"], "sample_idx": i,
                                    "gold": r["answer"], "n_tokens": ln, "finish_reason": "length" if tr else "stop",
                                    # byeolki stores the forced answer in 'answers' for truncated samples; natural answer unknown then
                                    "answer": None if tr else ans, "correct": (False if tr else bool(ok)),
                                    "forced_answer": ans if tr else None, "forced_correct": (bool(ok) if tr else None),
                                    "max_tokens": 8192, "backend": "vllm-byeolki", "source": "byeolki/aimo-interp (MIT)"}) + "\n"); n += 1
    print(f"{model}: {n} sample records -> {out}")
