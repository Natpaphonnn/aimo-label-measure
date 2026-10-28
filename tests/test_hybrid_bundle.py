"""hybrid-v1 bundle: local copies match src, the prior-only path passes the per-case protocol on both tracks, and (slow,
opt-in) the sampling + hook pipeline runs on a local model."""
import json, os, pathlib, shutil, sys, tempfile, time, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_submission_contract import GS, make_cases  # noqa: E402
from test_per_case_protocol import run  # noqa: E402
B = ROOT / "submissions/hybrid-v1"

def bundle_dir(dst: pathlib.Path, small: bool) -> pathlib.Path:
    shutil.copytree(B, dst, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "src/runtime", dst / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
    if small: (dst / "small.txt").write_text("small\n")
    return dst

class HybridBundle(unittest.TestCase):
    def test_copies_match_src(self):
        self.assertEqual((B / "hooks.py").read_bytes(), (ROOT / "src/features/hooks.py").read_bytes())
        self.assertEqual((B / "answers.py").read_bytes(), (ROOT / "src/selflabel/answers.py").read_bytes())

    @unittest.skipUnless((GS / "components").exists(), "official getting-started clone missing")
    def test_prior_path_both_tracks(self):
        for track, n_robust, n_total in (("main", 10, 21), ("small", 7, 14)):
            with tempfile.TemporaryDirectory() as t:
                root = pathlib.Path(t); make_cases(root, n_robust=n_robust, n_total=n_total); sub = bundle_dir(root / "sub", track == "small")
                for mode in ("same", "fresh"):
                    s, timing = run(sub, root, track, mode)
                    self.assertEqual(s["coverage"], 1.0); self.assertEqual(s["invalid_predictions"], 0)
                    self.assertAlmostEqual(s["accuracy"], n_robust / n_total); self.assertEqual(len(timing), n_total)

    @unittest.skipUnless(os.environ.get("AIMO_RUN_SLOW") and (ROOT / "data/models/Qwen3.5-4B").exists(), "set AIMO_RUN_SLOW=1 with a local Qwen3.5-4B")
    def test_pipeline_on_local_model(self):
        with tempfile.TemporaryDirectory() as t:
            sub = bundle_dir(pathlib.Path(t) / "sub", False); sys.path.insert(0, str(sub))
            from runtime import ModelCache  # noqa
            from features import compute_case  # noqa
            import solution  # noqa
            f = compute_case(ModelCache(), str(ROOT / "data/models/Qwen3.5-4B"), "default", "What is 17 * 23?", time.time() + 600, {"k": 2, "cut": 48, "min_cut": 48})
            self.assertTrue(f, "compute_case returned {}"); self.assertEqual(f["agreement"]["k"], 2.0); self.assertEqual(len(f["layers"]), 3)
            for l in f["layers"]:
                self.assertEqual(f["wb"][l]["trace_mean"].shape, (2560,)); self.assertEqual(f["wb"][l]["answer_mean"].shape, (2560,))
            self.assertGreater(f["n_steps"], 0); self.assertLessEqual(f["n_steps"], 47)
            # decision path with a synthetic artifact: pcs from a random projection, zero coefficients -> p = sigmoid(intercept)
            import numpy as np; rng = np.random.default_rng(0); l0 = f["layers"][1]
            orders = {"hybrid": [f"pc{i}" for i in range(4)] + [f"disp_L{l}" for l in f["layers"]] + ["agree_top", "share_finished", "log_steps"],
                      "probe": [f"pc{i}" for i in range(4)] + [f"disp_L{l}" for l in f["layers"]] + ["log_steps"], "agreement": ["agree_top", "share_finished", "log_steps"]}
            for arm, order in orders.items():
                cfg = {"layer": l0, "kind": "trace", "pca_mean": [0.0] * 2560, "pca_components": rng.normal(size=(4, 2560)).tolist(), "feature_order": order,
                       "scaler_mean": [0.0] * len(order), "scaler_scale": [1.0] * len(order), "coef": [0.0] * len(order), "intercept": 1.0, "threshold": 0.5}
                self.assertTrue(solution._decide(cfg, f, arm, 0.3), arm)
                cfg["intercept"] = -1.0; self.assertFalse(solution._decide(cfg, f, arm, 0.9), arm)
                self.assertTrue(solution._decide(cfg, {}, arm, 0.9), arm)   # empty features -> prior
            bad = {**cfg, "feature_order": ["pc0", "agree_top"]}; self.assertTrue(solution._decide(bad, f, "probe", 0.9))   # missing feature -> prior
            print(json.dumps({k: v for k, v in f["agreement"].items()}), f["answers"], f["n_steps"])

if __name__ == "__main__":
    unittest.main()
