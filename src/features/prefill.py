#!/usr/bin/env python3
"""Prefill-only features for a (model, problem) pair: one forward pass, no generation.

Outputs per problem:
  - hidden states at the last prompt token for every layer (float16), and mean-pooled over the
    problem tokens for every layer -> features/<model>/<tag>_hidden.npz
  - memorization / fluency statistics of the problem text under the model (teacher forcing of the
    problem tokens inside the chat prompt): mean log-prob, perplexity, Min-K% (k=10,20,40) mean
    log-prob, Min-K%++ (z-scored), token count, zlib ratio, first/last-quarter mean log-prob
  - next-token distribution at the generation position: entropy, top-1 prob, top-5 mass,
    P(first token starts a '<think>' block or a digit)
Runs with the pinned competition venv (transformers 5.13 / torch 2.12) on CUDA, MPS or CPU.

Usage: python src/features/prefill.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --problems data/pool/problems.jsonl --out runs/features/<model>/pool
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys, time, zlib
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from selflabel.prompts import chat_messages

def pick_device(force: str | None = None) -> tuple[str, torch.dtype]:
    if force:
        d = force
    elif torch.cuda.is_available(): d = "cuda"
    elif torch.backends.mps.is_available(): d = "mps"
    else: d = "cpu"
    if d == "cuda":
        # native bf16 needs compute capability >= 8.0; torch.cuda.is_bf16_supported() also returns True on a T4 (sm75) through
        # emulation, which made the Kaggle states kernels of 2026-10-22 crawl (no progress in 2 h): use fp16 there
        return d, (torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)
    return d, (torch.bfloat16 if d == "mps" else torch.float32)

def load(model_id: str, device: str, dtype: torch.dtype, local_only: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
    kw = dict(dtype=dtype, local_files_only=local_only)
    if device == "cuda":
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if device != "cuda":
        model.to(device)
    model.eval()
    return tok, model

def problem_token_span(tok, messages: list[dict], problem: str) -> tuple[list[int], int, int]:
    """Token ids of the full chat prompt and the [start, end) span covering the problem text."""
    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    offs = ids["offset_mapping"]; s = full.find(problem); e = s + len(problem)
    start = next((i for i, (a, b) in enumerate(offs) if b > s), 0)
    end = next((i for i, (a, b) in enumerate(offs) if a >= e), len(offs))
    return ids["input_ids"], start, end

@torch.inference_mode()
def featurize(tok, model, problem: str, device: str) -> tuple[dict, np.ndarray, np.ndarray]:
    ids, s, e = problem_token_span(tok, chat_messages(problem), problem)
    x = torch.tensor([ids], device=device)
    out = model(input_ids=x, output_hidden_states=True, use_cache=False, return_dict=True)
    logits = out.logits[0].float()                       # (T, V)
    logp = torch.log_softmax(logits, dim=-1)
    tgt = x[0, 1:]
    tok_lp = logp[:-1].gather(1, tgt[:, None])[:, 0]     # log p(token_t | <t) for t=1..T-1
    span = tok_lp[max(s - 1, 0): e - 1]                  # problem tokens
    n = int(span.numel())
    lp = span.cpu().numpy() if n else np.zeros(1)
    srt = np.sort(lp)
    def mink(k): return float(srt[: max(1, int(math.ceil(k * n)))].mean())
    # Min-K%++: z-score each token log-prob under the model's own next-token distribution
    mu = (logp[:-1].exp() * logp[:-1]).sum(-1)[max(s - 1, 0): e - 1]
    var = (logp[:-1].exp() * logp[:-1] ** 2).sum(-1)[max(s - 1, 0): e - 1] - mu ** 2
    z = ((span - mu) / var.clamp_min(1e-6).sqrt()).cpu().numpy() if n else np.zeros(1)
    zs = np.sort(z)
    last = logp[-1]; p = last.exp()
    top = torch.topk(p, 5)
    feats = {
        "n_prompt_tokens": len(ids), "n_problem_tokens": n,
        "mean_logprob": float(lp.mean()), "ppl": float(math.exp(-lp.mean())), "min_logprob": float(lp.min()),
        "mink10": mink(0.10), "mink20": mink(0.20), "mink40": mink(0.40),
        "minkpp20": float(zs[: max(1, int(math.ceil(0.2 * n)))].mean()),
        "first_quarter_logprob": float(lp[: max(1, n // 4)].mean()), "last_quarter_logprob": float(lp[-max(1, n // 4):].mean()),
        "zlib_ratio": len(zlib.compress(problem.encode())) / max(1, len(problem.encode())),
        "chars": len(problem), "gen_pos_entropy": float(-(p * last).sum()), "gen_pos_top1": float(top.values[0]),
        "gen_pos_top5_mass": float(top.values.sum()),
    }
    hs = out.hidden_states                                 # tuple(L+1) of (1, T, H)
    last_tok = np.stack([h[0, -1].float().cpu().numpy() for h in hs]).astype(np.float16)
    mean_prob = np.stack([h[0, s:e].float().mean(0).cpu().numpy() for h in hs]).astype(np.float16)
    return feats, last_tok, mean_prob

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--problems", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True); ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--max-chars", type=int, default=6000)
    a = ap.parse_args()
    device, dtype = pick_device(a.device)
    rows = [json.loads(l) for l in a.problems.read_text().splitlines() if l.strip()][: a.limit]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); tok, model = load(a.model, device, dtype, a.local_only)
    print(f"[prefill] {a.model} on {device}/{dtype} loaded in {time.time()-t0:.0f}s; {len(rows)} problems", flush=True)
    feats, last, mean, ids = [], [], [], []
    for i, r in enumerate(rows):
        f, lt, mp = featurize(tok, model, r["problem"][: a.max_chars], device)
        f.update(problem_id=r["problem_id"], model_id=a.model); feats.append(f); last.append(lt); mean.append(mp); ids.append(r["problem_id"])
        if (i + 1) % 20 == 0:
            print(f"[prefill] {i+1}/{len(rows)} {(time.time()-t0)/(i+1):.2f}s per problem", flush=True)
    pathlib.Path(str(a.out) + "_feats.jsonl").write_text("".join(json.dumps(f) + "\n" for f in feats))
    np.savez_compressed(str(a.out) + "_hidden.npz", problem_id=np.array(ids), last_token=np.stack(last), mean_problem=np.stack(mean))
    print(f"[prefill] wrote {a.out}_feats.jsonl and _hidden.npz in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
