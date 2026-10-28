"""Run a built submission ZIP through the OFFICIAL ingestion and scoring programs on a synthetic Main-track-like case set
(all five models, gpt-oss at low/medium/high plus an unexpected effort string, balanced labels) and assert full coverage,
no invalid predictions, and that a constant-True bundle scores exactly the robust share. Reproduces the 2026-10-16 check of
submission 991595 (Main 0.48 = 10/21 with coverage 1.0): the drop from the expected 0.571 was the label balance, not the bundle."""
import json, pathlib, subprocess, sys, tempfile, unittest, zipfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
GS = ROOT / "external" / "aimo-interp_getting-started"
CONFIGS = [("Qwen/Qwen3.5-4B", None), ("Skywork/Skywork-OR1-Math-7B", None), ("allenai/Olmo-3-7B-Think", None),
           ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", None), ("openai/gpt-oss-120b", "low"), ("openai/gpt-oss-120b", "medium"),
           ("openai/gpt-oss-120b", "high"), ("openai/gpt-oss-120b", "xhigh")]

def make_cases(root: pathlib.Path, n_robust: int = 10, n_total: int = 21):
    cases, labels = [], []
    for i in range(n_total):
        m, e = CONFIGS[i % len(CONFIGS)]
        c = {"id": f"case-{i:03d}", "model_id": m, "problem": f"Problem {i // 3}: compute {i} + 1."}
        if e: c["reasoning_effort"] = e
        cases.append(c); labels.append({"id": c["id"], "is_robust": i < n_robust})
    (root / "input").mkdir(); (root / "reference").mkdir()
    (root / "input/cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    (root / "reference/labels.jsonl").write_text("".join(json.dumps(l) + "\n" for l in labels))

def run_official(zip_path: pathlib.Path, small: bool, root: pathlib.Path) -> dict:
    sub = root / "sub"; zipfile.ZipFile(zip_path).extractall(sub); res = root / "scoring_in" / "res"; ref = root / "scoring_in" / "ref"
    ref.parent.mkdir(parents=True); subprocess.run(["cp", "-r", str(root / "reference"), str(ref)], check=True)
    track = "small" if small else "main"
    subprocess.run([sys.executable, str(GS / "components/ingestion_program/ingestion.py"), str(root / "input"), str(res), str(sub), "--track", track], check=True, capture_output=True)
    subprocess.run([sys.executable, str(GS / "components/scoring_program/scoring.py"), str(root / "scoring_in"), str(root / "out"), "--track", track], check=True)
    return json.loads((root / "out/scores.json").read_text())

@unittest.skipUnless((GS / "components").exists(), "official getting-started clone missing (tools/clone_external.sh)")
class SubmissionContract(unittest.TestCase):
    def test_prior_only_main_full_coverage_and_constant_true(self):
        z = ROOT / "dist/prior-only-v1-main.zip"
        if not z.exists(): self.skipTest("build dist/prior-only-v1-main.zip first")
        with tempfile.TemporaryDirectory() as t:
            root = pathlib.Path(t); make_cases(root)
            s = run_official(z, False, root)
        self.assertEqual(s["coverage"], 1.0); self.assertEqual(s["invalid_predictions"], 0)
        self.assertAlmostEqual(s["accuracy"], 10 / 21)   # constant True on 10 robust of 21 -> 0.476, shown as 0.48

    def test_prior_only_small(self):
        z = ROOT / "dist/prior-only-v1-small.zip"
        if not z.exists(): self.skipTest("build dist/prior-only-v1-small.zip first")
        with tempfile.TemporaryDirectory() as t:
            root = pathlib.Path(t); make_cases(root, n_robust=7, n_total=14)
            s = run_official(z, True, root)
        self.assertEqual(s["coverage"], 1.0); self.assertEqual(s["invalid_predictions"], 0); self.assertAlmostEqual(s["accuracy"], 0.5)

if __name__ == "__main__":
    unittest.main()
