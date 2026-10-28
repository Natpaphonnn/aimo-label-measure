"""Shared test-time runtime for Codabench submissions: time budget, model cache, safe batches, priors.

Copied verbatim into each submission bundle (see tools/build_submission.py --include src/runtime).
No third-party imports beyond torch/transformers, which exist in the evaluation image.
"""
from __future__ import annotations
import gc, hashlib, json, os, pathlib, sys, tempfile, time
from typing import Callable

TOTAL_BUDGET_S = 3600.0
SAFETY_S = 420.0          # reserve for import, final writes, and shared-GPU jitter
MODEL_LOAD_S = {"openai/gpt-oss-120b": 30.0}   # expected load seconds; default below
DEFAULT_LOAD_S = 10.0
# Per-problem protocol (organizers, Discord 2026-10-16): are_robust is called once per case with a one-item list, each case in a
# FRESH Python process (timing canary 991764), under one shared job budget (val Main: 4200 s for 21 cases with a 180 s safety
# margin = 191 s of wall clock per case, including process start and imports that happen before this module is loaded).
PER_CASE_S = float(os.environ.get("AIMO_PER_CASE_S", "160"))   # what one call may spend after import; ~30 s per case of buffer
MIN_USEFUL_S = 15.0                                             # below this a call returns its prior without loading anything
# Static per-case allotment used when no state persists between the fresh case processes (canary v2 995120: the temp dir does not).
# Rule: the job cap is 4200 s for the 21 val Main cases (262.5 s per unique problem, 200 s per case); organizers rescale the test cap
# proportionally and say passing val means fitting the test. 110 s = 0.55 x 200 s leaves room for a test with up to ~1.8x the val
# cases-per-problem ratio (val 1.31; train-main-v2 2.28 would overrun by ~25% at the tail). Override with AIMO_STATIC_ALLOT_S.
STATIC_ALLOT_S = float(os.environ.get("AIMO_STATIC_ALLOT_S", "110"))

def log(msg: str) -> None:
    print(f"[aimo {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

class Budget:
    """Wall-clock budget across all are_robust calls. Allot time per batch in proportion to a cost weight."""
    def __init__(self, total_s: float = TOTAL_BUDGET_S, safety_s: float = SAFETY_S):
        self.t0 = time.time(); self.total = total_s; self.safety = safety_s; self.seen: list[tuple[str, int]] = []
    def elapsed(self) -> float: return time.time() - self.t0
    def remaining(self) -> float: return self.total - self.safety - self.elapsed()
    def allot(self, weight: float, expected_future_weight: float) -> float:
        """Seconds for the current batch given its weight and the expected weight of batches still to come."""
        return max(0.0, self.remaining() * weight / max(weight + expected_future_weight, 1e-9))
    def deadline(self, allot_s: float) -> float: return time.time() + allot_s

class CaseClock:
    """Per-case time budget that works whether all calls share one process or each call starts a fresh one.
    Call k (1-based) in a process may run until k * per_case_s seconds after import, capped at per_case_s + max_bank_s for the
    call itself: unused time from fast calls (prior fallbacks, cached features) is banked, overruns are paid back by later calls.
    Time between calls counts as spent (it is wall clock the server also counts). A fresh process gets exactly per_case_s."""
    def __init__(self, per_case_s: float = PER_CASE_S, max_bank_s: float | None = None, clock: Callable[[], float] = time.time):
        self.per_case = per_case_s; self.max_bank = per_case_s if max_bank_s is None else max_bank_s
        self.clock = clock; self.t0 = clock(); self.cases = 0
    def start(self, n_cases: int = 1) -> float:
        """Register a call covering n_cases and return the seconds it may use (0 when the process is behind schedule)."""
        self.cases += max(1, n_cases)
        earned = self.cases * self.per_case - (self.clock() - self.t0)
        return max(0.0, min(earned, max(1, n_cases) * self.per_case + self.max_bank))

# Global budget (organizers, #qa 2026-10-16 15:30): one cap for the whole evaluation, scaled by the number of UNIQUE problems
# (val Main: 21 cases, 16 problems, 4200 s = 262.5 s per problem); no per-call limit; cases per problem vary on the test.
SEC_PER_PROBLEM = float(os.environ.get("AIMO_SEC_PER_PROBLEM", "240"))   # earned per new problem; 22.5 s/problem below val's rate
CASE_OVERHEAD_S = float(os.environ.get("AIMO_CASE_OVERHEAD_S", "10"))     # process start + imports + prior answer, per future case
CASE_CAP_S = float(os.environ.get("AIMO_CASE_CAP_S", "200"))              # never more than this for one case
STALE_S = 900.0                                                           # state older than this belongs to an earlier job

class JobClock:
    """Budget across cases that each run in a FRESH process (canary 991764), shared through a small state file in the temp dir.

    Accounting: every newly seen unique problem earns `sec_per_problem`; time spent is wall clock since the first case of the job
    (minus nothing: the overhead between cases is real time). Cases of problems already seen are expected to keep coming:
    c_hat = max(c0, most cases seen for one problem); each seen problem p still expects max(c_hat - n_p, 0) cases. The current
    case gets an equal share of what is left after reserving `overhead_s` for every POSSIBLE future case (up to c_max per
    problem; the overhead of a prior-only case is unavoidable, so it is always reserved):
        bank  = earned - spent - safety - overhead * future_max
        allot = clamp(bank / (1 + future_expected), 0, cap)
    Future problems are ignored (they bring their own earnings), which keeps the rule conservative. When `allot` is small the
    caller answers its prior. The state holds only hashes of problem texts, never the texts.
    Canary v2 (995120, 2026-10-17) showed that the temp dir does NOT persist between cases, so the default is now the STATIC
    allotment `static_allot_s` (STATIC_ALLOT_S) per case; the shared mode is used only when a persistent `state_dir` is given
    (canary v3 looks for one) or AIMO_STATELESS=0 is set."""
    def __init__(self, track: str = "main", sec_per_problem: float = SEC_PER_PROBLEM, c0: float | None = None, c_max: int | None = None,
                 overhead_s: float = CASE_OVERHEAD_S, safety_s: float = 30.0, cap_s: float = CASE_CAP_S, state_dir: str | None = None,
                 key: str = "", clock: Callable[[], float] = time.time, stateless: bool | None = None, static_allot_s: float = STATIC_ALLOT_S):
        # c_max = number of configs (Main: 4 small models + gpt-oss x 3 efforts; Small: 4). c0 = cases per problem assumed until
        # more are seen: 3 on Main, 4 (= c_max) on Small (analysis/sim_jobclock.py; DECISIONS #29).
        self.c_max = c_max or (7 if track == "main" else 4); self.c0 = c0 if c0 is not None else (3.0 if track == "main" else 4.0)
        self.B = sec_per_problem
        self.overhead = overhead_s; self.safety = safety_s; self.cap = cap_s; self.clock = clock
        self.static = static_allot_s
        if stateless is None:
            env = os.environ.get("AIMO_STATELESS")
            stateless = (env == "1") if env in ("0", "1") else (state_dir is None)   # no persistent dir known -> static
        self.stateless = stateless
        tag = hashlib.sha256(f"{os.getppid()}|{key}".encode()).hexdigest()[:12]
        self.path = pathlib.Path(state_dir or tempfile.gettempdir()) / f"aimo_jobclock_{tag}.json"
        self.last: dict = {}

    def _load(self, now: float) -> dict:
        try:
            st = json.loads(self.path.read_text())
            if now - float(st.get("last", 0)) <= STALE_S:
                return st
        except Exception:  # noqa: BLE001 - missing or corrupt state means a new job
            pass
        return {"job_start": now, "last": now, "problems": {}, "cases": 0}

    def _save(self, st: dict) -> None:
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp"); tmp.write_text(json.dumps(st)); os.replace(tmp, self.path)

    def start(self, problems: list[str]) -> float:
        n = max(1, len(problems)); now = self.clock()
        if self.stateless:
            a = max(0.0, n * self.static); self.last = {"mode": "static", "allot": a}; return min(a, n * self.cap)
        try:
            st = self._load(now)
            for p in problems or [""]:
                h = hashlib.sha256(p.encode("utf-8", "ignore")).hexdigest()[:16]
                st["problems"][h] = int(st["problems"].get(h, 0)) + 1
            st["cases"] = int(st.get("cases", 0)) + n; st["last"] = now
            U = len(st["problems"]); c_hat = min(max(self.c0, max(st["problems"].values())), self.c_max)
            future = sum(max(c_hat - k, 0.0) for k in st["problems"].values())            # cases expected: shares compute
            future_max = sum(max(self.c_max - k, 0.0) for k in st["problems"].values())   # cases possible: reserves overhead
            spent = now - float(st["job_start"])
            bank = self.B * U - spent - self.safety - self.overhead * future_max
            allot = max(0.0, min(n * bank / (n + future), n * self.cap))
            st["last_allot"] = allot; self._save(st)
            self.last = {"mode": "shared", "U": U, "cases": st["cases"], "c_hat": c_hat, "future": round(future, 1),
                         "spent": round(spent, 1), "bank": round(bank, 1), "allot": round(allot, 1)}
            return allot
        except Exception as exc:  # noqa: BLE001 - never fail a case because of the clock
            a = max(0.0, n * self.static); self.last = {"mode": f"error {type(exc).__name__}", "allot": a}
            return a

    def touch(self) -> None:
        """Mark the end of a case (keeps the state fresh for the STALE_S check)."""
        try:
            st = self._load(self.clock()); st["last"] = self.clock(); self._save(st)
        except Exception:  # noqa: BLE001
            pass

class ModelCache:
    """Load one model at a time from the offline HF cache; free VRAM between models."""
    def __init__(self):
        self.model_id: str | None = None; self.model = None; self.tokenizer = None
    def get(self, model_id: str):
        if self.model_id == model_id and self.model is not None:
            return self.tokenizer, self.model
        self.release()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t = time.time()
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        kw = {"local_files_only": True}
        mps = (not torch.cuda.is_available()) and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if torch.cuda.is_available():
            kw["device_map"] = "auto"
            if "gpt-oss" not in model_id:  # gpt-oss stays in its MXFP4 form (62 GiB); others in bf16
                kw["dtype"] = torch.bfloat16
        elif mps:                          # local functional tests on a Mac only; the evaluation image has CUDA
            kw["dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw); model.eval()
        if mps: model.to("mps")
        self.model_id, self.model, self.tokenizer = model_id, model, tok
        log(f"loaded {model_id} in {time.time() - t:.1f}s")
        return tok, model
    def release(self) -> None:
        if self.model is not None:
            log(f"releasing {self.model_id}")
        self.model = None; self.tokenizer = None; self.model_id = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.synchronize()
        except Exception:
            pass

def load_priors(path: pathlib.Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}

def prior_for(priors: dict, model_id: str, effort: str) -> float:
    """P(robust) for a configuration; falls back to model-level then global prior."""
    key = f"{model_id}|{effort}"
    if key in priors: return float(priors[key])
    same = [v for k, v in priors.items() if k.split("|")[0] == model_id and k != "__global__"]
    if same: return float(sum(same) / len(same))
    return float(priors.get("__global__", 0.6))

def native_bools(values) -> list[bool]:
    return [bool(v) for v in values]

def safe_batch(fn: Callable[[], list], n: int, fallback_value: bool, name: str = "batch") -> list[bool]:
    """Run fn(); on any failure or malformed output, return the fallback for every problem. Output is a real list of bools."""
    try:
        out = native_bools(fn())
        if len(out) != n:
            log(f"{name}: wrong length {len(out)} != {n}; falling back"); return [fallback_value] * n
        return out
    except Exception as exc:  # noqa: BLE001 - any failure must still yield a valid batch
        log(f"{name}: failed with {type(exc).__name__}: {str(exc)[:300]}; falling back")
        return [fallback_value] * n
