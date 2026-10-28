import sys, pathlib, unittest, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from runtime import Budget, CaseClock, prior_for, safe_batch, native_bools

class T(unittest.TestCase):
    def test_budget_allot(self):
        b = Budget(total_s=100, safety_s=10)
        self.assertAlmostEqual(b.allot(1.0, 3.0), 90 * 0.25, delta=0.5)
        self.assertAlmostEqual(b.allot(1.0, 0.0), 90, delta=0.5)
    def test_case_clock_banks_and_repays(self):
        now = [0.0]; c = CaseClock(per_case_s=100, max_bank_s=50, clock=lambda: now[0])
        self.assertEqual(c.start(), 100)            # fresh process: exactly one case worth
        now[0] = 10; self.assertEqual(c.start(), 150)   # 90 s banked but capped at 100 + 50
        now[0] = 300; self.assertEqual(c.start(), 0)    # behind schedule (300 s spent for 3 cases): prior only
        now[0] = 310; self.assertEqual(c.start(), 90)   # 4 cases earned 400 s, 310 spent
        self.assertEqual(c.start(n_cases=3), 350)      # a multi-item call (legacy protocol) gets 3 cases + bank, capped
    def test_prior(self):
        p = {"m|low": 0.7, "m|high": 0.9, "__global__": 0.6}
        self.assertEqual(prior_for(p, "m", "low"), 0.7); self.assertAlmostEqual(prior_for(p, "m", "medium"), 0.8); self.assertEqual(prior_for(p, "x", "default"), 0.6)
    def test_safe_batch(self):
        import numpy as np
        self.assertEqual(safe_batch(lambda: [np.bool_(True), 1, 0], 3, False), [True, True, False])
        self.assertEqual(safe_batch(lambda: [True], 3, True), [True, True, True])
        self.assertEqual(safe_batch(lambda: 1 / 0, 2, False), [False, False])
        out = safe_batch(lambda: [np.bool_(True)], 1, False); self.assertIs(type(out[0]), bool); self.assertIs(type(out), list)

class JobClockSim(unittest.TestCase):
    def test_never_exceeds_global_budget(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "analysis"))
        import random, sim_jobclock as S
        for name, (cases, track) in S.scenarios(random.Random(1)).items():
            r = S.run(cases, track); self.assertLess(r["used/budget"], 0.95, name)
            if name.startswith("val"): self.assertEqual(r["prior_fallbacks"], 0, name)
    def test_state_is_shared_and_expires(self):
        import tempfile
        from runtime import JobClock, STALE_S
        now = [1000.0]
        with tempfile.TemporaryDirectory() as d:
            a = JobClock(track="small", state_dir=d, clock=lambda: now[0]).start(["p"]); now[0] += 50
            b = JobClock(track="small", state_dir=d, clock=lambda: now[0]); b.start(["q"]); self.assertEqual(b.last["U"], 2)
            now[0] += STALE_S + 1
            c = JobClock(track="small", state_dir=d, clock=lambda: now[0]); c.start(["q"]); self.assertEqual(c.last["U"], 1)
            self.assertGreater(a, 0)

if __name__ == "__main__":
    unittest.main()
