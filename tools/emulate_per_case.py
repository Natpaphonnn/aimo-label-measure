#!/usr/bin/env python3
"""Local emulator of the per-problem ingestion protocol (organizers, Discord 2026-10-16; the new ingestion code is not public).

Each case is one call `are_robust(model_id, reasoning_effort=<effort>, problems=[problem])`; the output must be a list of one
native bool, otherwise that case is invalid. Writes predictions.jsonl and ingestion_summary.json in the format the official
scoring program reads, plus timing.jsonl (seconds per call).

Modes (which one the server uses is still open on Discord, so bundles must pass both):
  same   all calls in one Python process (module-level caches persist)
  fresh  a new Python process per case (nothing persists between calls)

Usage: python tools/emulate_per_case.py <input_dir with cases.jsonl> <output_dir> <unzipped submission dir> --track main|small [--mode same|fresh]
"""
from __future__ import annotations
import argparse, importlib.util, json, pathlib, subprocess, sys, time

def track_ok(sub: pathlib.Path, track: str) -> bool:
    marker = any(p.name.lower() == "small.txt" for p in sub.iterdir() if p.is_file())
    return marker == (track == "small")

def call_one(sub: pathlib.Path, case: dict) -> tuple[bool, bool]:
    """Import the solution (cached by Python after the first import) and run one case. Returns (prediction, valid)."""
    if "solution" not in sys.modules:
        sys.path.insert(0, str(sub))
        spec = importlib.util.spec_from_file_location("solution", sub / "solution.py")
        mod = importlib.util.module_from_spec(spec); sys.modules["solution"] = mod; spec.loader.exec_module(mod)
    mod = sys.modules["solution"]
    try:
        out = mod.are_robust(case["model_id"], reasoning_effort=case.get("reasoning_effort", "default"), problems=[case["problem"]])
        if type(out) is list and len(out) == 1 and type(out[0]) is bool:
            return out[0], True
    except Exception:  # noqa: BLE001 - mirrors the official ingestion: details are not copied
        pass
    return False, False

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("input_dir", type=pathlib.Path); ap.add_argument("output_dir", type=pathlib.Path)
    ap.add_argument("submission_dir", type=pathlib.Path); ap.add_argument("--track", choices=["main", "small"], required=True)
    ap.add_argument("--mode", choices=["same", "fresh"], default="same")
    ap.add_argument("--one-case", default=None, help=argparse.SUPPRESS)   # internal: fresh-mode worker
    a = ap.parse_args(); sub = a.submission_dir.resolve()
    if a.one_case is not None:
        pred, valid = call_one(sub, json.loads(a.one_case)); print(json.dumps({"is_robust": pred, "valid": valid})); return
    if not track_ok(sub, a.track):
        sys.exit(f"track marker does not match --track {a.track}")
    cases = [json.loads(l) for l in (a.input_dir / "cases.jsonl").read_text().splitlines() if l.strip()]
    a.output_dir.mkdir(parents=True, exist_ok=True); preds, timing = [], []
    for case in cases:
        t = time.time()
        if a.mode == "same":
            pred, valid = call_one(sub, case)
        else:
            r = subprocess.run([sys.executable, __file__, str(a.input_dir), str(a.output_dir), str(sub), "--track", a.track, "--one-case", json.dumps(case)],
                               capture_output=True, text=True)
            try:
                d = json.loads(r.stdout.strip().splitlines()[-1]); pred, valid = bool(d["is_robust"]), bool(d["valid"])
            except Exception:  # noqa: BLE001 - a crashed worker is an invalid case
                pred, valid = False, False
        preds.append({"id": case["id"], "is_robust": pred if valid else False, "valid": valid})
        timing.append({"id": case["id"], "model_id": case["model_id"], "seconds": round(time.time() - t, 3)})
    with (a.output_dir / "predictions.jsonl").open("w") as f:
        for p in preds: f.write(json.dumps(p, sort_keys=True, separators=(",", ":")) + "\n")
    (a.output_dir / "ingestion_summary.json").write_text(json.dumps({"cases": len(cases), "invalid_predictions": sum(not p["valid"] for p in preds)}, sort_keys=True) + "\n")
    (a.output_dir / "timing.jsonl").write_text("".join(json.dumps(t) + "\n" for t in timing))

if __name__ == "__main__":
    main()
