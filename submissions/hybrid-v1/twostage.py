"""Two-stage white-box decision (DECISIONS #44), numpy only; the same arithmetic as analysis/wb_two_stage.py.

Stage 1 (per model): p_i = sigmoid(coef . PCA(standardise(state_i)) + intercept) for each of the K samples, where state_i is the
residual-stream vector of block (kind, layer) read at the sample's own end (hooks.ResidualRecorder.features(lengths)).
Black-box twin: s_i = share of the other K - 1 samples whose final answer equals sample i's (0 if it has none).
Stage 2 (shared): logistic regression on [wb_mean, wb_sd, bb_mean, bb_sd] (the arm's subset) + model one-hot.
Artifact (artifacts/hybrid_model.json, "format": "two-stage"):
  {"format": "two-stage", "arm": "wb|bb|hybrid", "run": {...},
   "stage1": {"<model_id>": {"kind": "trace|answer", "layer": int, "scaler_mean": [H], "scaler_scale": [H], "pca_mean": [H],
                             "pca_components": [[n x H]], "coef": [n], "intercept": float}},
   "stage2": {"<arm>": {"features": [...], "models": [...], "scaler_mean": [...], "scaler_scale": [...], "coef": [...],
                        "intercept": float, "threshold": 0.5}}}
A model without a stage-1 probe uses the "bb" stage 2 when the artifact has one, else the prior; a model absent from stage 2's
model list (Olmo, gpt-oss-120b: no white-box training data) uses "bb_any" (agreement only, no model offset) when present, else the prior.
"""
from __future__ import annotations
import math
import numpy as np

ARM_FEATURES = {"bb": ["bb_mean", "bb_sd"], "wb": ["wb_mean", "wb_sd"], "hybrid": ["wb_mean", "wb_sd", "bb_mean", "bb_sd"]}


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))


def twin(answers: list) -> list[float]:
    k = len(answers); a = [None if x in (None, "None") else str(x) for x in answers]
    if k <= 1: return [0.0] * k
    return [0.0 if ai is None else (sum(1 for aj in a if aj == ai) - 1) / (k - 1) for ai in a]


def probe(s1: dict, X: np.ndarray) -> np.ndarray:
    """X: (K, H) states of the probe's block -> (K,) p_correct."""
    Z = (np.asarray(X, np.float64) - np.asarray(s1["scaler_mean"])) / np.where(np.asarray(s1["scaler_scale"]) == 0, 1.0, np.asarray(s1["scaler_scale"]))
    W = (Z - np.asarray(s1["pca_mean"])) @ np.asarray(s1["pca_components"]).T
    return np.array([_sigmoid(float(w @ np.asarray(s1["coef"]) + s1["intercept"])) for w in W])


def case_scores(art: dict, model_id: str, per_sample: dict | None, answers: list) -> dict:
    tw = np.array(twin(answers)); out = {"bb_mean": float(tw.mean()) if len(tw) else 0.0, "bb_sd": float(tw.std()) if len(tw) else 0.0}
    s1 = (art.get("stage1") or {}).get(model_id)
    if s1 is not None and per_sample:
        blk = per_sample.get(int(s1["layer"])) or per_sample.get(str(s1["layer"]))
        if blk is not None:
            p = probe(s1, blk[s1["kind"]]); out.update({"wb_mean": float(p.mean()), "wb_sd": float(p.std()), "p": [round(float(x), 4) for x in p]})
    return out


def decide(art: dict, model_id: str, per_sample: dict | None, answers: list, prior: float) -> tuple[bool, dict]:
    sc = case_scores(art, model_id, per_sample, answers); arm = art.get("arm", "wb")
    if arm in ("wb", "hybrid") and "wb_mean" not in sc: arm = "bb"
    s2 = (art.get("stage2") or {}).get(arm)
    if (s2 is None or model_id not in s2["models"]) and "bb_any" in (art.get("stage2") or {}):
        s2, arm = art["stage2"]["bb_any"], "bb_any"   # unlabelled model: agreement-only stage 2 without model offsets
    if s2 is None or (arm != "bb_any" and model_id not in s2["models"]):
        return prior >= 0.5, {**sc, "arm_used": "prior"}
    x = [sc[f] for f in s2["features"]] + [1.0 if model_id == m else 0.0 for m in s2["models"]]
    z = sum(c * (v - m) / (s if s else 1.0) for c, v, m, s in zip(s2["coef"], x, s2["scaler_mean"], s2["scaler_scale"])) + s2["intercept"]
    p = _sigmoid(z); return p >= s2.get("threshold", 0.5), {**sc, "arm_used": arm, "p_robust": round(p, 4)}
