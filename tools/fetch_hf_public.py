#!/usr/bin/env python3
"""Download public aimo-interp HF datasets at pinned revisions (stdlib only, no token needed)."""
import json, sys, urllib.request, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "hf"
DATASETS = ["val-sample","train-main-v2","augmented-sample-math-agg","augmented-sample-math",
            "augmented-sample-math-full-filtered","augmented-sample-math-agg-filtered","aimo-interp-challenge-sample-full"]
UA = {"User-Agent": "aimo-interp-fetch/0.1"}
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()
for d in DATASETS:
    out = ROOT / d; out.mkdir(parents=True, exist_ok=True)
    info = json.loads(get(f"https://huggingface.co/api/datasets/aimo-interp/{d}").decode("utf-8"), strict=False)
    sha = info["sha"]; (out / "_revision.txt").write_text(sha + "\n")
    (out / "_api.json").write_text(json.dumps(info, indent=1))
    print(f"=== {d} @ {sha} lastModified={info.get('lastModified')}")
    for s in info["siblings"]:
        f = s["rfilename"]; p = out / f; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(get(f"https://huggingface.co/datasets/aimo-interp/{d}/resolve/{sha}/{f}"))
        print(f"    {f}  {p.stat().st_size} bytes")
