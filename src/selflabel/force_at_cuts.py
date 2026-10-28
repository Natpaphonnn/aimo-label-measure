#!/usr/bin/env python3
"""Post-hoc forced answers at token cut points for already generated samples (mlx-lm).
For each record with token_ids and each cut c < n_tokens: context = chat prompt + first c generated tokens + FORCE_SUFFIX,
greedy decode FORCE_TOKENS, parse the answer. Samples that finished naturally before c keep their natural answer at that cut.
Adds `cuts: {c: {"answer", "correct", "natural": bool}}` per record; writes <out>. Lets one 8k-token run emulate 2k / 4k budgets.
Usage: python src/selflabel/force_at_cuts.py --model data/models/Qwen3.5-4B --records runs/gen/X.jsonl --problems data/pool/problems.jsonl --cuts 1024 2048 4096 --out runs/gen/X.cuts.jsonl"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from selflabel.answers import parse_answer, is_correct
from selflabel.prompts import FORCE_SUFFIX, FORCE_TOKENS, chat_messages

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--records", type=pathlib.Path, required=True)
    ap.add_argument("--problems", type=pathlib.Path, required=True); ap.add_argument("--cuts", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--out", type=pathlib.Path, required=True); ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    texts = {}
    for l in a.problems.read_text().splitlines():
        if l.strip():
            r = json.loads(l); texts[(r["problem_id"], r.get("family", "original"), int(r.get("variant", 0)))] = r["problem"]
    recs = [json.loads(l) for l in a.records.read_text().splitlines() if l.strip()]
    model, tok = load(a.model); force_ids = tok.encode(FORCE_SUFFIX, add_special_tokens=False)
    jobs = []
    for i, r in enumerate(recs):
        r["cuts"] = {}
        ids = r.get("token_ids")
        for c in a.cuts:
            if ids is None: continue
            if r["finish_reason"] != "length" and r["n_tokens"] <= c:
                r["cuts"][str(c)] = {"answer": r["answer"], "correct": r["correct"], "natural": True}
            elif r["n_tokens"] > c or r["finish_reason"] == "length":
                key = (r["problem_id"], r.get("family", "original"), int(r.get("variant", 0)))
                prompt = tok.apply_chat_template(chat_messages(texts[key]), add_generation_prompt=True, enable_thinking=True)
                jobs.append((i, c, prompt + ids[:c] + force_ids))
    print(f"[cuts] {len(recs)} records, {len(jobs)} forced continuations", flush=True); t0 = time.time()
    gen = BatchGenerator(model, max_tokens=FORCE_TOKENS, stop_tokens=[[t] for t in tok.eos_token_ids], sampler=make_sampler(temp=0.0),
                         completion_batch_size=a.batch, prefill_batch_size=2, prefill_step_size=2048)
    for s in range(0, len(jobs), a.batch):
        chunk = jobs[s:s + a.batch]; uids = gen.insert([j[2] for j in chunk], [FORCE_TOKENS] * len(chunk)); out = {u: [] for u in uids}; left = set(uids)
        while left:
            for resp in gen.next_generated():
                if resp.finish_reason != "stop" and resp.token is not None: out[resp.uid].append(resp.token)
                if resp.finish_reason is not None: left.discard(resp.uid)
        for (i, c, _), u in zip(chunk, uids):
            ans = parse_answer("</think>\nANSWER:" + tok.decode(out[u])); gold = recs[i]["gold"]
            recs[i]["cuts"][str(c)] = {"answer": ans, "correct": is_correct(ans, gold), "natural": False}
        if (s // a.batch) % 20 == 0: print(f"[cuts] {s+len(chunk)}/{len(jobs)} {(time.time()-t0)/(s+len(chunk)):.2f}s each", flush=True)
    gen.close()
    a.out.write_text("".join(json.dumps({k: v for k, v in r.items() if k not in ("token_ids", "text")}) + "\n" for r in recs))
    print(f"[cuts] wrote {a.out} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
