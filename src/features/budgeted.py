#!/usr/bin/env python3
"""Budgeted-sampling features in the TEST-TIME regime: k short samples per problem with a token cap, a greedy
forced answer at the cut, and per-problem statistics (agreement, finish rate, forced-answer commitment confidence,
answer-entropy proxies). Same code path is meant to run inside the Codabench bundle, so it uses only HF generate.

Per problem: k samples (temperature 1.0, top_k 40, top_p 0.95) up to --max-tokens; for every sample a forced
answer is read (greedy, FORCE_TOKENS) from the cut or from the natural end, with its mean top-1 probability.
Outputs jsonl rows: problem_id, model_id, k, max_tokens, answers, forced_answers, finished, n_tokens, agree,
agree_forced, finish_rate, mean_commit_conf, n_distinct, has_nan, mean_tokens, seconds.

Usage: python src/features/budgeted.py --model Skywork/Skywork-OR1-Math-7B --problems data/pool/problems.jsonl --out runs/features/<model>/pool_budgeted.jsonl --k 2 --max-tokens 3072 --batch 16
"""
from __future__ import annotations
import argparse, collections, json, pathlib, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from selflabel.answers import parse_answer, is_correct
from selflabel.prompts import FORCE_SUFFIX, FORCE_TOKENS, chat_messages

def load(model_id: str, local_only: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only); tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else (torch.float16 if torch.cuda.is_available() else torch.float32)
    kw = dict(dtype=dtype, local_files_only=local_only)
    if torch.cuda.is_available(): kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw); model.eval()
    return tok, model

@torch.inference_mode()
def generate(tok, model, prompts: list[str], max_new: int, sample: bool, seed: int) -> tuple[list[list[int]], list[bool]]:
    batch = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    torch.manual_seed(seed)
    kw = dict(do_sample=True, temperature=1.0, top_k=40, top_p=0.95) if sample else dict(do_sample=False)
    out = model.generate(**batch, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **kw)
    new = out[:, batch["input_ids"].shape[1]:]; eos = {x for x in [tok.eos_token_id, tok.pad_token_id] if x is not None}
    eos |= set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id or -1])
    ids, finished = [], []
    for row in new.tolist():
        cut = next((i for i, t in enumerate(row) if t in eos), None)
        ids.append(row[:cut] if cut is not None else row); finished.append(cut is not None)
    return ids, finished

@torch.inference_mode()
def forced_answers(tok, model, contexts: list[str]) -> tuple[list, list[float]]:
    """Greedy continuation after FORCE_SUFFIX; returns parsed answers and mean top-1 prob of the generated tokens."""
    batch = tok(contexts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**batch, max_new_tokens=FORCE_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
    seqs = out.sequences[:, batch["input_ids"].shape[1]:]; probs = torch.stack([torch.softmax(s.float(), -1).max(-1).values for s in out.scores], 1)
    answers, confs = [], []
    for i in range(seqs.shape[0]):
        text = tok.decode(seqs[i], skip_special_tokens=True); n = max(1, len([t for t in seqs[i].tolist() if t != tok.pad_token_id]))
        answers.append(parse_answer("</think>\nANSWER:" + text)); confs.append(float(probs[i, :n].mean()))
    return answers, confs

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--problems", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--k", type=int, default=2); ap.add_argument("--max-tokens", type=int, default=3072); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--local-only", action="store_true")
    a = ap.parse_args(); rows = [json.loads(l) for l in a.problems.read_text().splitlines() if l.strip()][: a.limit]
    a.out.parent.mkdir(parents=True, exist_ok=True); done = set()
    if a.out.exists(): done = {json.loads(l)["problem_id"] for l in a.out.read_text().splitlines() if l.strip()}
    rows = [r for r in rows if r["problem_id"] not in done]
    t0 = time.time(); tok, model = load(a.model, a.local_only); print(f"[budgeted] loaded {a.model} in {time.time()-t0:.0f}s; {len(rows)} problems to do", flush=True)
    per = max(1, a.batch // a.k)
    for s in range(0, len(rows), per):
        chunk = rows[s:s + per]; t = time.time()
        prompts = [tok.apply_chat_template(chat_messages(r["problem"]), tokenize=False, add_generation_prompt=True) for r in chunk]
        flat = [p for p in prompts for _ in range(a.k)]
        ids, fin = generate(tok, model, flat, a.max_tokens, True, a.seed + s)
        texts = [tok.decode(x, skip_special_tokens=True) for x in ids]
        ctx = [flat[i] + texts[i] + ("" if fin[i] and "</think>" in texts[i] else FORCE_SUFFIX) for i in range(len(flat))]
        fans, confs = forced_answers(tok, model, ctx)
        recs = []
        for j, r in enumerate(chunk):
            sl = slice(j * a.k, (j + 1) * a.k)
            nat = [parse_answer(x) for x in texts[sl]]; fa = fans[sl]; fn = fin[sl]; cf = confs[sl]; nt = [len(x) for x in ids[sl]]
            use = [n if n is not None else f for n, f in zip(nat, fa)]
            cnt = collections.Counter([u for u in use if u is not None]); top = cnt.most_common(1)[0][1] if cnt else 0
            cntf = collections.Counter([u for u in fa if u is not None]); topf = cntf.most_common(1)[0][1] if cntf else 0
            gold = r.get("answer")
            recs.append({"problem_id": r["problem_id"], "model_id": a.model, "k": a.k, "max_tokens": a.max_tokens, "answers": use, "natural_answers": nat, "forced_answers": fa,
                         "finished": fn, "n_tokens": nt, "agree": top / a.k, "agree_forced": topf / a.k, "finish_rate": sum(fn) / a.k, "mean_commit_conf": sum(cf) / a.k,
                         "n_distinct": len(cnt), "has_nan": any(u == "NaN" for u in use), "mean_tokens": sum(nt) / a.k,
                         "correct_rate": (sum(is_correct(u, gold) for u in use) / a.k) if gold is not None else None, "seconds": round((time.time() - t) / len(chunk), 1)})
        with a.out.open("a") as f:
            for rec in recs: f.write(json.dumps(rec) + "\n")
        print(f"[budgeted] {s+len(chunk)}/{len(rows)} | {sum(sum(x) for x in [[len(i) for i in ids]])/(time.time()-t):.0f} tok/s | finish {sum(fin)}/{len(fin)}", flush=True)
    print(f"[budgeted] done in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
