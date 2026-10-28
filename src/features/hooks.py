"""Test-time white-box capture during HF `generate` (DECISIONS #33): forward hooks on chosen decoder layers record the residual
stream at every decode step at no extra generation cost. Per sequence and layer the recorder keeps
  * a running mean over the last `window` generated tokens (ring buffer of per-step states), and
  * the state at the most recent position (= the answer position once generation stops).
Both are read after generate() returns; pass `lengths` (generated tokens per sequence, EOS included) so that sequences that
stopped early are read at their own end, not at the pad steps HF keeps decoding for them (2026-10-22 fix: before it, the
trace/answer states of finished samples were pad states and disagreed with trace_states.py). Prompt positions are skipped (N2: prompt-only states read the topic), the prefill call
is recognised as the step whose sequence length is > 1. Convention: with n generated tokens there are n - 1 decode steps, so the
recorded states belong to generated tokens 1..n-1; the final token's own state is never computed during generation. The "answer"
state is therefore the state of the token that EMITTED the last token. trace_states.py uses the same positions offline.

Usage:
    rec = ResidualRecorder(model, layers=[7, 15, 23], window=64)
    with rec: out = model.generate(**inputs, ...)
    feats = rec.features(lengths)   # {"trace": {layer: (B, H) np.float32}, "answer": {layer: (B, H)}, "n_steps": int}
`decoder_layers(model)` finds the layer list for Qwen3.5 / Qwen3 / Llama / Olmo-style models and gpt-oss.
No third-party imports beyond torch (the evaluation image has it).
"""
from __future__ import annotations
import numpy as np, torch


def decoder_layers(model) -> list[torch.nn.Module]:
    for path in ("model.layers", "model.language_model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
                return list(obj)
        except AttributeError:
            continue
    raise AttributeError("no decoder layer list found")


def layer_ids(n_layers: int, fracs=(0.25, 0.5, 0.75)) -> list[int]:
    return sorted({min(n_layers - 1, max(0, round(f * n_layers) - 1)) for f in fracs})


class ResidualRecorder:
    def __init__(self, model, layers: list[int], window: int = 64):
        self.model = model; self.layers = list(layers); self.window = window
        self.mods = decoder_layers(model); self.handles = []
        self.ring: dict[int, list[torch.Tensor]] = {l: [] for l in self.layers}   # per layer: every (B, H) step state, fp16
        # (all steps are kept: 3072 steps x 8 x 4096 x 3 layers x 2 B = 0.6 GB at most for the competition models)
        self.n_steps = 0; self._counted: set[int] = set()

    def _hook(self, layer: int):
        def fn(module, args, output):
            h = output[0] if isinstance(output, (tuple, list)) else output          # (B, T, H)
            if h.dim() != 3 or h.shape[1] != 1:                                       # prefill (T > 1) or unexpected: skip
                return
            self.ring[layer].append(h[:, -1, :].detach().to(torch.float16))
            if layer == self.layers[0]:
                self.n_steps += 1
        return fn

    def __enter__(self):
        for l in self.layers:
            self.handles.append(self.mods[l].register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def features(self, lengths=None) -> dict:
        """lengths[b] = generated tokens of sequence b (EOS included, at most the cut). With m tokens the states of tokens
        0..m-2 exist (decode steps 1..m-1), so sequence b is read at step min(m - 1, n_steps); None = every sequence ran to the
        last step (the old behaviour)."""
        trace, answer = {}, {}
        for l in self.layers:
            buf = self.ring[l]
            if not buf:
                continue
            S = torch.stack(buf)                                        # (steps, B, H)
            B = S.shape[1]; ends = [S.shape[0]] * B if lengths is None else [max(1, min(int(m) - 1, S.shape[0])) for m in lengths]
            trace[l] = torch.stack([S[max(0, e - self.window): e, b].float().mean(0) for b, e in enumerate(ends)]).cpu().numpy()
            answer[l] = torch.stack([S[e - 1, b].float() for b, e in enumerate(ends)]).cpu().numpy()
        return {"trace": trace, "answer": answer, "n_steps": self.n_steps}


def case_features(feats: dict, finished_mask=None) -> dict:
    """Per-case aggregates over the K sequences (DECISIONS #33): mean of trace and answer states, and the dispersion of the trace
    states (mean pairwise cosine distance). Returns {layer: {"trace_mean", "answer_mean", "dispersion"}}."""
    out = {}
    for l, X in feats["trace"].items():
        A = feats["answer"][l]
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
        S = Xn @ Xn.T; K = X.shape[0]
        disp = float((1 - S)[~np.eye(K, dtype=bool)].mean()) if K > 1 else 0.0
        out[l] = {"trace_mean": X.mean(0), "answer_mean": A.mean(0), "dispersion": disp}
    return out
