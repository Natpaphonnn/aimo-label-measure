"""hybrid-v1: robustness from K test-time reasoning samples (agreement) + the residual stream recorded during those samples.

Design (notes/DECISIONS.md #17, #21, #26, #29, #32-34):
  * per-case protocol: one are_robust call per case, fresh process, one global budget shared through runtime.JobClock;
  * features.compute_case samples K traces at the official settings with hooks.ResidualRecorder attached; the same bundle code
    serves three pre-registered arms, chosen by the artifact: "hybrid" (agreement + PCA(states) + dispersion), "agreement"
    (black-box only) and "probe" (states only). Only the artifact differs between the nominated bundles;
  * any failure, missing artifact or short allotment returns the per-config prior (native bools, correct length);
  * artifacts/hybrid_model.json format: {"arm": ..., "run": {k, cut, ...}, "configs": {"<model_id>": {"layer": int, "kind": "trace|answer",
    "pca_mean": [H], "pca_components": [[H]...], "feature_order": [...], "scaler_mean": [...], "scaler_scale": [...], "coef": [...],
    "intercept": float, "threshold": 0.5, "run": {...overrides}}, "default": {...}} (written by analysis/fit_hybrid_artifact.py).
"""
from __future__ import annotations
import json, math, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runtime import MIN_USEFUL_S, JobClock, ModelCache, load_priors, log, prior_for, safe_batch  # noqa: E402
import twostage  # noqa: E402

PRIORS = load_priors(HERE / "artifacts" / "priors.json")
ARTIFACT = HERE / "artifacts" / "hybrid_model.json"
TRACK = "small" if any(q.name.lower() == "small.txt" for q in HERE.iterdir()) else "main"
CLOCK = JobClock(track=TRACK, key=str(HERE))
CACHE = ModelCache()
CALLS: list[dict] = []


def _artifact() -> dict | None:
    return json.loads(ARTIFACT.read_text()) if ARTIFACT.exists() else None


def _config(art: dict, model_id: str, effort: str) -> dict | None:
    if model_id in art.get("ablate_models", []): return None   # DECISIONS #53: white-box ablation, these models answer the prior
    if art.get("format") == "two-stage":   # DECISIONS #44: decided in twostage.py; sampling runs only for models stage 2 knows
        known = {m for s2 in (art.get("stage2") or {}).values() for m in s2.get("models", [])}
        return {"two_stage": True} if (model_id in known or "bb_any" in (art.get("stage2") or {})) else None
    c = art.get("configs", {}); return c.get(f"{model_id}|{effort}") or c.get(model_id) or art.get("default")


def _vector(cfg: dict, feats: dict, arm: str) -> list[float] | None:
    """Assemble the feature vector in cfg['feature_order']: pc<i> (PCA of the chosen state block), disp_L<l>, agreement names,
    and n_steps/cut terms. Returns None when a required block is missing (then the caller falls back)."""
    import numpy as np
    vals = {}
    if arm in ("hybrid", "probe"):
        wb = feats.get("wb") or {}; layer = int(cfg["layer"]); kind = cfg.get("kind", "trace")
        if layer not in wb: return None
        x = np.asarray(wb[layer][f"{kind}_mean"], dtype=np.float64)
        z = (x - np.asarray(cfg["pca_mean"])) @ np.asarray(cfg["pca_components"]).T
        vals.update({f"pc{i}": float(v) for i, v in enumerate(z)})
        vals.update({f"disp_L{l}": float(wb[l]["dispersion"]) for l in wb})
    if arm in ("hybrid", "agreement"):
        vals.update({k: float(v) for k, v in (feats.get("agreement") or {}).items()})
    vals["n_steps"] = float(feats.get("n_steps", 0)); vals["log_steps"] = math.log1p(vals["n_steps"])
    try:
        return [vals[name] for name in cfg["feature_order"]]
    except KeyError as exc:
        log(f"feature {exc} missing for arm {arm}"); return None


def _decide(cfg: dict, feats: dict, arm: str, prior: float, art: dict | None = None, model_id: str = "") -> bool:
    if cfg.get("two_stage"):
        if not feats: return prior >= 0.5
        d, info = twostage.decide(art, model_id, feats.get("per_sample"), feats.get("answers") or [], prior)
        log(f"two-stage: {info}"); return d
    v = _vector(cfg, feats, arm) if feats else None
    if v is None:
        return prior >= 0.5
    z = sum(c * (x - m) / (s if s else 1.0) for c, x, m, s in zip(cfg["coef"], v, cfg["scaler_mean"], cfg["scaler_scale"])) + cfg["intercept"]
    p = 1 / (1 + math.exp(-max(min(z, 50), -50)))
    log(f"p_robust={p:.3f} threshold={cfg.get('threshold', 0.5)}")
    return p >= cfg.get("threshold", 0.5)


def are_robust(model_id: str, reasoning_effort: str = "default", problems: list[str] | None = None) -> list[bool]:
    problems = list(problems or []); effort = str(reasoning_effort or "default"); n = len(problems)
    prior = prior_for(PRIORS, model_id, effort); fallback = prior >= 0.5
    allot = CLOCK.start(problems); deadline = time.time() + allot
    art = _artifact(); cfg = _config(art, model_id, effort) if art else None; arm = (art or {}).get("arm", "hybrid")
    log(f"call {len(CALLS)+1}: {model_id} effort={effort} n={n} prior={prior:.2f} allot={allot:.0f}s arm={arm} model={'yes' if cfg else 'none'}")
    t0 = time.time()
    if cfg is None or allot < MIN_USEFUL_S:
        out = [fallback] * n
    else:
        from features import compute_case  # lazy: a broken torch never breaks the import
        run = {**art.get("run", {}), **cfg.get("run", {})}
        per = allot / max(n, 1)
        def _all():
            res = []
            for i, p in enumerate(problems):
                d = min(deadline, time.time() + per) if n > 1 else deadline
                res.append(_decide(cfg, compute_case(CACHE, model_id, effort, p, d, run), arm, prior, art, model_id))
            return res
        out = safe_batch(_all, n, fallback, name=f"{model_id}|{effort}")
    CALLS.append({"model_id": model_id, "effort": effort, "n": n, "seconds": round(time.time() - t0, 1), "positives": sum(out)})
    CLOCK.touch()
    log(f"done {model_id} effort={effort}: {sum(out)}/{n} robust in {time.time()-t0:.1f}s clock={CLOCK.last}")
    return out
