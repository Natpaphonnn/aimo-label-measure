#!/usr/bin/env python3
"""Simulate runtime.JobClock under the global per-unique-problem budget (organizers, 2026-10-16): fake clock, fresh state per job,
every case pays process overhead before import, then uses its FULL allot (worst case) or answers its prior when allot < MIN_USEFUL_S.
Reports total time / true budget (must stay < 1), mean allot, and the share of prior fallbacks, for several case orderings.
True budget = 262.5 s per unique problem (val Main: 4200 s / 16 problems). Usage: python analysis/sim_jobclock.py"""
import random, sys, tempfile
sys.path.insert(0, "src")
from runtime import MIN_USEFUL_S, JobClock

TRUE_PER_PROBLEM = 4200 / 16

def run(cases, track="main", pre_s=7.0, post_s=1.0, seed=0, **kw):   # true overhead 7 + 1 (+1 for a prior answer) <= reserved 10
    t = [0.0]; allots = []; fallbacks = 0
    with tempfile.TemporaryDirectory() as d:
        for prob in cases:
            t[0] += pre_s
            clk = JobClock(track=track, state_dir=d, clock=lambda: t[0], **kw)   # fresh process: new object every case
            a = clk.start([prob]); allots.append(a)
            if a < MIN_USEFUL_S: fallbacks += 1; t[0] += 1.0
            else: t[0] += a
            t[0] += post_s; clk.touch()
    budget = TRUE_PER_PROBLEM * len(set(cases))
    return {"cases": len(cases), "problems": len(set(cases)), "used/budget": round(t[0] / budget, 3), "mean_allot": round(sum(allots) / len(allots), 1),
            "min_allot": round(min(allots), 1), "prior_fallbacks": fallbacks}

def scenarios(rng):
    out = {}
    v = [f"p{i}" for i in range(16)] + [f"p{i}" for i in rng.sample(range(16), 5)]; rng.shuffle(v); out["val Main 21 cases / 16 problems, random"] = (v, "main")
    v = [f"s{i}" for i in range(11)] + [f"s{i}" for i in rng.sample(range(11), 3)]; rng.shuffle(v); out["val Small 14 cases / 11 problems (assumed), random"] = (v, "small")
    for k, track in [(7, "main"), (4, "small"), (3, "main")]:
        probs = [f"q{i}" for i in range(40)]
        by_problem = [p for p in probs for _ in range(k)]
        by_config = [p for _ in range(k) for p in probs]
        rnd = by_problem[:]; rng.shuffle(rnd)
        out[f"test 40 problems x {k} cases ({track}), grouped by problem"] = (by_problem, track)
        out[f"test 40 problems x {k} cases ({track}), grouped by config"] = (by_config, track)
        out[f"test 40 problems x {k} cases ({track}), random"] = (rnd, track)
    var = [p for i in range(40) for p in [f"r{i}"] * rng.randint(1, 7)]; rng.shuffle(var); out["test 40 problems x 1-7 cases (main), random"] = (var, "main")
    return out

if __name__ == "__main__":
    rng = random.Random(0)
    print("| scenario | cases | used / budget | mean allot s | min allot s | prior fallbacks |")
    print("|---|---|---|---|---|---|")
    worst = 0
    for name, (cases, track) in scenarios(rng).items():
        r = run(cases, track); worst = max(worst, r["used/budget"])
        print(f"| {name} | {r['cases']} | {r['used/budget']} | {r['mean_allot']} | {r['min_allot']} | {r['prior_fallbacks']} |")
    st = run(scenarios(random.Random(0))["test 40 problems x 7 cases (main), grouped by config"][0], "main", stateless=True)
    print(f"| stateless fallback, 40 x 7 main | {st['cases']} | {st['used/budget']} | {st['mean_allot']} | {st['min_allot']} | {st['prior_fallbacks']} |")
    print("worst used/budget:", worst)
