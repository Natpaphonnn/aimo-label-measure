#!/usr/bin/env python3
"""Apply the template families (rename, typos, distract) to every pool problem. Writes data/variants/templates_v1.jsonl
(one row per variant with the same schema as llm-authored variants) and prints coverage per family."""
import hashlib, json, pathlib, sys, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from selflabel.perturb import FAMILIES, make_variant
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "analysis")); from datapaths import ROOT as _R, need  # noqa: E402
need(_R / "data/pool/problems.jsonl")
pool = [json.loads(l) for l in (ROOT / "data/pool/problems.jsonl").read_text().splitlines() if l.strip()]
out = ROOT / "data/variants/templates_v1.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
rows, skipped = [], collections.Counter()
for p in pool:
    for fam in FAMILIES:
        seed = int(hashlib.sha256(f"{p['problem_id']}:{fam}:0".encode()).hexdigest()[:8], 16)
        text, note = make_variant(p["problem"], fam, seed)
        if note is None:
            skipped[fam] += 1; continue
        rows.append({"problem_id": p["problem_id"], "family": fam, "variant": 0, "method": "template", "text": text, "answer": p["answer"], "author_notes": note, "verification": "template: math spans and numbers unchanged by construction", "confidence": 1.0})
out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
c = collections.Counter(r["family"] for r in rows)
print(f"{len(rows)} variants for {len(pool)} problems -> {out}; per family {dict(c)}; skipped (nothing to change) {dict(skipped)}")
