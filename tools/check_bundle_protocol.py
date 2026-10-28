#!/usr/bin/env python3
"""Per-case protocol check of a built submission directory (the #46 'emulator' gate): both tracks, same and fresh process,
official scoring program; coverage must be 1.0 with 0 invalid predictions. Models are hidden (empty HF cache), so every case
takes the fallback path end to end; the sampling path itself is checked by the local model smoke (scratch wb_smoke.py).

Usage: python tools/check_bundle_protocol.py submissions/<name>
"""
import os, pathlib, shutil, sys, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_submission_contract import make_cases  # noqa: E402
from test_per_case_protocol import run  # noqa: E402

if len(sys.argv) != 2 or sys.argv[1].startswith("-"): sys.exit("usage: python tools/check_bundle_protocol.py <bundle source dir, e.g. submissions/hybrid-v1>")
src = pathlib.Path(sys.argv[1]).resolve(); ok = True
for track, n_robust, n_total in (("main", 10, 21), ("small", 7, 14)):
    with tempfile.TemporaryDirectory() as t:
        root = pathlib.Path(t); make_cases(root, n_robust=n_robust, n_total=n_total)
        sub = root / "sub"; shutil.copytree(src, sub, ignore=shutil.ignore_patterns("__pycache__"))
        if track == "small": (sub / "small.txt").write_text("small\n")
        os.environ.update({"HF_HOME": str(root / "hf"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        for mode in ("same", "fresh"):
            s, timing = run(sub, root, track, mode)
            good = s["coverage"] == 1.0 and s["invalid_predictions"] == 0 and len(timing) == n_total
            ok &= good; print(f"{src.name} {track} {mode}: coverage {s['coverage']} invalid {s['invalid_predictions']} acc {s['accuracy']:.3f} "
                              f"max case {max(x.get('seconds', 0) for x in timing):.1f}s -> {'ok' if good else 'FAIL'}")
sys.exit(0 if ok else 1)
