"""Input and output paths of the analysis scripts. A missing input stops the script with what the file is and where it comes
from, instead of a traceback; RESULTS (notes/results/) is created on import so every script can write on a fresh checkout.

Usage: from datapaths import ROOT, RESULTS, need;  need(ROOT / "data/labels/case_labels.csv")
"""
from __future__ import annotations
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "notes/results"
RESULTS.mkdir(parents=True, exist_ok=True)

# path prefix (relative to ROOT) -> (what it is, where it comes from); the longest matching prefix wins
SOURCES = {
    "data/labels/case_labels.csv": ("robustness labels of the stage-2 cases (byeolki + train-main-v2)",
                                    "python tools/fetch_hf_public.py && python tools/build_case_labels.py"),
    "data/seed/": ("byeolki labels and base accuracies (MIT, Haeyoung Park)",
                   "shipped in data/seed/; rebuilt by analysis/p1_byeolki_reuse.py from github.com/byeolki/aimo-interp"),
    "data/hf/": ("official public datasets of the challenge", "python tools/fetch_hf_public.py"),
    "data/official/": ("official problems with stable ids", "python tools/build_official_problem_files.py after tools/fetch_hf_public.py"),
    "data/pool/": ("problem pools (third-party problem texts, not redistributed)",
                   "tools/clone_external.sh, then tools/build_problem_pool.py (problems.jsonl); tt_wave_problems.jsonl is the test-time wave of tools/kaggle_push_gen.py"),
    "runs/val_logs/": ("stderr logs of the validation server ('Full Log' download of each submission, not redistributed)",
                       "only the submitting team can download them; results in notes/results/val_board_v1.md and runtime_val.md"),
    "runs/": ("logged generations and residual states (several GB, not shipped)",
              "regenerate with src/selflabel/ and the Kaggle kernels of tools/kaggle_push_gen.py; derived results are in notes/results/"),
    "external/": ("clones of the organizers' and third-party repositories", "bash tools/clone_external.sh"),
    "notes/results/": ("result file written by another analysis script", "shipped in notes/results/, or rerun the script that writes it"),
}


def need(*paths) -> None:
    """Exit with a '[missing input]' message for the first path that does not exist (an .npz is accepted for a states stem)."""
    for p in map(pathlib.Path, paths):
        if p.exists() or pathlib.Path(str(p) + ".npz").exists():
            continue
        rel = p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)
        key = max((k for k in SOURCES if rel.startswith(k)), key=len, default=None)
        what, src = SOURCES.get(key, ("input file", "see README.md"))
        sys.exit(f"[missing input] {rel}\n  what: {what}\n  where from: {src}")
