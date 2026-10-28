"""hybrid-v1 features: K sampled reasoning traces per case under the official sampling settings, with the residual stream
recorded during the SAME generation (hooks.py; DECISIONS #32-33). Returns black-box agreement statistics and white-box
per-layer aggregates (mean trace state, mean answer state, cross-sample dispersion) and the per-sample states (two-stage artifacts).

compute_case(cache, model_id, effort, problem, deadline, cfg) -> dict | {}   (never raises; {} on failure)
  cfg keys (artifact "run" block, all optional): k (8), cut (2048), layers_frac ([.25,.5,.75]), window (64), force_tokens (16),
  reserve_s (12), steps_per_s (per-model estimate used to shrink the cut when the allotment is short).
Determinism: torch.manual_seed from a hash of (model_id, effort, problem); sampling is still GPU-kernel dependent, so the
decision rule is trained to be insensitive to single-sample flips (K samples, aggregates only).
"""
from __future__ import annotations
import hashlib, time
import numpy as np
from runtime import log
from answers import parse_answer
from hooks import ResidualRecorder, case_features, decoder_layers, layer_ids

SYSTEM = ("You are a precise math problem solver. Solve the given problem step by step, then output your final answer on the last line in the exact format:\n"
          "ANSWER: <your answer>\nThis result should be an integer value or NaN if the given problem does not have a solution.")
FORCE_SUFFIX = "\n</think>\n\nTime is up. Based on the reasoning so far, the final answer in the required format is:\nANSWER:"
DEFAULTS = {"k": 8, "cut": 2048, "layers_frac": [0.25, 0.5, 0.75], "window": 64, "force_tokens": 16, "reserve_s": 12.0, "steps_per_s": 40.0, "min_cut": 256}
# Per-model K / cut / decode speed from the server (canary v2 995120, 4k-token sampled decode, RTX PRO 6000): Qwen3.5-4B b8 flat 52
# steps/s; Skywork b4 74->67; Olmo b8 55->21 (strong long-context slowdown); DeepSeek b16 55->28; gpt-oss b4 flat ~25. steps_per_s
# below are conservative averages over the planned cut so that cut x K fits ~85 s of generation inside a 110 s static allotment.
MODEL_RUN = {"Qwen/Qwen3.5-4B": {"k": 8, "cut": 3072, "steps_per_s": 50.0},
             "Skywork/Skywork-OR1-Math-7B": {"k": 8, "cut": 3072, "steps_per_s": 60.0},
             "allenai/Olmo-3-7B-Think": {"k": 6, "cut": 2048, "steps_per_s": 40.0},
             "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": {"k": 8, "cut": 2560, "steps_per_s": 45.0},
             "openai/gpt-oss-120b": {"k": 4, "cut": 1536, "steps_per_s": 24.0}}


def _prompt(tok, model_id: str, effort: str, problem: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": problem}]
    extra = {"reasoning_effort": effort} if "gpt-oss" in model_id and effort not in ("default", "") else {"enable_thinking": True}
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **extra)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def _eos_ids(tok, model) -> set[int]:
    eos = set(x for x in [tok.eos_token_id] if x is not None)
    gc = getattr(model, "generation_config", None)
    if gc is not None and gc.eos_token_id is not None:
        eos |= set(gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id])
    return eos


def _agreement(final: list, natural: list, finished: list[bool], lengths: list[int], cut: int) -> dict:
    k = len(final); answered = [a for a in final if a is not None]
    def top_share(xs):
        if not xs: return 0.0
        vals, counts = np.unique([str(x) for x in xs], return_counts=True); return float(counts.max()) / k
    return {"k": float(k), "agree_top": top_share(answered), "agree_natural": top_share([a for a in natural if a is not None]),
            "n_distinct": float(len(set(str(a) for a in answered))), "share_answered": len(answered) / k,
            "share_finished": float(np.mean(finished)) if k else 0.0, "share_nan": sum(1 for a in answered if a == "NaN") / k,
            "mean_len_frac": float(np.mean(lengths)) / max(cut, 1), "cut": float(cut)}


def compute_case(cache, model_id: str, effort: str, problem: str, deadline: float, cfg: dict | None = None) -> dict:
    cfg = {**DEFAULTS, **MODEL_RUN.get(model_id, {}), **(cfg or {})}
    try:
        import torch
        t0 = time.time(); tok, model = cache.get(model_id); load_s = time.time() - t0
        text = _prompt(tok, model_id, effort, problem[:6000])
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        k = int(cfg["k"]); ids = {kk: v.to(model.device).repeat(k, 1) for kk, v in enc.items()}; P = int(ids["input_ids"].shape[1])
        avail = deadline - time.time() - float(cfg["reserve_s"])
        cut = int(min(cfg["cut"], max(cfg["min_cut"], avail * float(cfg["steps_per_s"]) * 0.85)))
        if avail < 5: log(f"no time for sampling (avail {avail:.0f}s)"); return {}
        n_layers = len(decoder_layers(model)); layers = layer_ids(n_layers, tuple(cfg["layers_frac"]))
        seed = int(hashlib.sha256(f"{model_id}|{effort}|{problem}".encode()).hexdigest()[:8], 16); torch.manual_seed(seed)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        rec = ResidualRecorder(model, layers, window=int(cfg["window"]))
        t1 = time.time()
        with torch.inference_mode(), rec:
            out = model.generate(**ids, do_sample=True, temperature=1.0, top_k=40, top_p=0.95, max_new_tokens=cut, max_time=max(5.0, avail), pad_token_id=pad)
        gen_s = time.time() - t1; eos = _eos_ids(tok, model)
        gens = [out[i, P:].tolist() for i in range(out.shape[0])]
        lengths, finished, texts, natural = [], [], [], []
        for g in gens:
            L = next((j for j, x in enumerate(g) if x in eos), len(g)); lengths.append(L); finished.append(L < len(g))
            txt = tok.decode(g[:L], skip_special_tokens=True); texts.append(txt); natural.append(parse_answer(txt))
        # per-sample ends: tokens before EOS (finished) or the cut, the same positions trace_states.py uses for saved token_ids (no EOS)
        raw = rec.features(lengths) if rec.n_steps > 0 else {}; wb = case_features(raw) if raw else {}
        del out; rec.ring.clear()
        final = list(natural); forced_n = 0
        need = [i for i in range(k) if not finished[i]]          # label convention: forced answers replace the natural answer of truncated samples
        if need and deadline - time.time() > 6:
            try:
                fb = tok([text + texts[i] + FORCE_SUFFIX for i in need], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                with torch.inference_mode():
                    fg = model.generate(**fb, do_sample=False, max_new_tokens=int(cfg["force_tokens"]), pad_token_id=pad)
                for row, i in enumerate(need):
                    fa = parse_answer("</think>\nANSWER:" + tok.decode(fg[row, fb["input_ids"].shape[1]:], skip_special_tokens=True))
                    if fa is not None: final[i] = fa; forced_n += 1
                del fb, fg
            except Exception as exc:  # noqa: BLE001
                log(f"forced answers failed: {type(exc).__name__}: {str(exc)[:120]}")
        agree = _agreement(final, natural, finished, lengths, cut); agree["forced_share"] = forced_n / k
        log(f"case: load {load_s:.1f}s gen {gen_s:.1f}s steps {rec.n_steps} cut {cut} k {k} agree_top {agree['agree_top']:.2f} finished {sum(finished)}/{k} layers {layers}")
        per_sample = {l: {"trace": raw["trace"][l], "answer": raw["answer"][l]} for l in raw.get("trace", {})} if raw else {}
        return {"agreement": agree, "wb": wb, "per_sample": per_sample, "layers": layers, "n_steps": int(rec.n_steps), "gen_s": gen_s, "load_s": load_s, "answers": [str(a) for a in final]}
    except Exception as exc:  # noqa: BLE001
        log(f"compute_case failed: {type(exc).__name__}: {str(exc)[:200]}"); return {}
