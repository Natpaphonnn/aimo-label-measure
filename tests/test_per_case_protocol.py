"""Per-problem protocol (organizers, Discord 2026-10-16): every case is its own are_robust call with a one-item list, and it is
not yet known whether calls share one process. Runs bundles through tools/emulate_per_case.py in both modes, scores with the
OFFICIAL scoring program, and asserts full coverage and the expected constant-prior accuracy."""
import json, os, pathlib, shutil, subprocess, sys, tempfile, unittest, zipfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_submission_contract import GS, make_cases  # noqa: E402

def solve_rate_dir(dst: pathlib.Path, small: bool, name: str = "solve-rate-v1") -> pathlib.Path:
    shutil.copytree(ROOT / "submissions" / name, dst, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "src/runtime", dst / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
    if small: (dst / "small.txt").write_text("small\n")
    return dst

def run(sub: pathlib.Path, root: pathlib.Path, track: str, mode: str) -> tuple[dict, list]:
    res = root / f"res_{mode}"; sc = root / f"scoring_{mode}"
    subprocess.run([sys.executable, str(ROOT / "tools/emulate_per_case.py"), str(root / "input"), str(res), str(sub), "--track", track, "--mode", mode],
                   check=True, capture_output=True, env={**os.environ, "TMPDIR": str(root)})   # bundles may write temp files
    (sc / "ref").parent.mkdir(parents=True); shutil.copytree(root / "reference", sc / "ref"); shutil.copytree(res, sc / "res")
    subprocess.run([sys.executable, str(GS / "components/scoring_program/scoring.py"), str(sc), str(root / f"out_{mode}"), "--track", track], check=True, capture_output=True)
    timing = [json.loads(l) for l in (res / "timing.jsonl").read_text().splitlines()]
    return json.loads((root / f"out_{mode}/scores.json").read_text()), timing

@unittest.skipUnless((GS / "components").exists(), "official getting-started clone missing (tools/clone_external.sh)")
class PerCaseProtocol(unittest.TestCase):
    def check(self, sub, root, track, n_robust, n_total, modes=("same", "fresh")):
        for mode in modes:
            s, timing = run(sub, root, track, mode)
            self.assertEqual(s["coverage"], 1.0, mode); self.assertEqual(s["invalid_predictions"], 0, mode)
            self.assertAlmostEqual(s["accuracy"], n_robust / n_total, msg=mode)   # every prior >= 0.5 -> constant True
            self.assertEqual(len(timing), n_total)

    def test_prior_only_zip_main(self):
        z = ROOT / "dist/prior-only-v1-main.zip"
        if not z.exists(): self.skipTest("build dist/prior-only-v1-main.zip first")
        with tempfile.TemporaryDirectory() as t:
            root = pathlib.Path(t); make_cases(root); zipfile.ZipFile(z).extractall(root / "sub")
            self.check(root / "sub", root, "main", 10, 21)

    def test_solve_rate_skeleton_both_tracks(self):
        for track, (nr, nt) in {"main": (10, 21), "small": (7, 14)}.items():
            with tempfile.TemporaryDirectory() as t:
                root = pathlib.Path(t); make_cases(root, n_robust=nr, n_total=nt)
                self.check(solve_rate_dir(root / "sub", track == "small"), root, track, nr, nt)

    def test_timing_canary_main_same_process(self):
        # no weights in the local HF cache: every measurement fails inside the bundle and the prior is still returned
        with tempfile.TemporaryDirectory() as t:
            root = pathlib.Path(t); make_cases(root)
            self.check(solve_rate_dir(root / "sub", False, "timing-canary"), root, "main", 10, 21, modes=("same",))

    def test_timing_canary_v2_main_same_process(self):
        with tempfile.TemporaryDirectory() as t:
            root = pathlib.Path(t); make_cases(root)
            self.check(solve_rate_dir(root / "sub", False, "timing-canary-v2"), root, "main", 10, 21, modes=("same",))

    def test_timing_canary_v3_main_both_modes(self):
        with tempfile.TemporaryDirectory() as t:
            root = pathlib.Path(t); make_cases(root)
            self.check(solve_rate_dir(root / "sub", False, "timing-canary-v3"), root, "main", 10, 21)
            # one direct call must emit a [canary3] JSON line with the probes
            r = subprocess.run([sys.executable, "-c", "import sys, json; sys.path.insert(0, sys.argv[1]); import solution; print(solution.are_robust('Qwen/Qwen3.5-4B', 'default', ['x']))", str(root / "sub")],
                               capture_output=True, text=True, env={**os.environ, "TMPDIR": str(root)})
            self.assertIn("[True]", r.stdout); line = next(l for l in r.stderr.splitlines() if l.startswith("[canary3] "))
            rec = json.loads(line[len("[canary3] "):]); self.assertIn("probes", rec); self.assertTrue(any(pr.get("writable") for pr in rec["probes"]))

if __name__ == "__main__":
    unittest.main()
