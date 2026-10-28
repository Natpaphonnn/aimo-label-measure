#!/usr/bin/env python3
"""Post-hoc forced answers at token cut points for saved generations: torch/transformers version of force_at_cuts.py (Kaggle
T4 / CUDA / MPS; the MLX one is Mac-only). Same output: adds `cuts: {c: {"answer", "correct", "natural"}}` per record.
For each record with token_ids and each cut c: if the sample finished naturally within c tokens the natural answer is kept
(natural=True); otherwise context = chat prompt + first c generated tokens + FORCE_SUFFIX, greedy FORCE_TOKENS, parsed.
Default path: one sequence at a time, prefill in 2048-token chunks through the KV cache, then greedy decode step by step.
Qwen3.5's torch linear-attention fallback copies the whole prefix to fp32 during prefill (4+ GiB for one 8k sequence), which OOMed a
T4 even at batch 1 (shards v3, 2026-10-17); chunking bounds that copy. --chunk-prefill 0 selects batched generate instead.
Usage: python src/selflabel/force_at_cuts_hf.py --model Qwen/Qwen3.5-4B --records a.jsonl b.jsonl --problems data/pool/night2_qwen35_amc.jsonl \
          --cuts 512 1024 2048 4096 --out runs/gen/X.cuts.jsonl [--batch 1] [--limit N]"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from selflabel.answers import parse_answer, is_correct
from selflabel.prompts import FORCE_SUFFIX, FORCE_TOKENS, chat_messages
from features.prefill import pick_device, load


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--records", type=pathlib.Path, nargs="+", required=True)
    ap.add_argument("--problems", type=pathlib.Path, required=True); ap.add_argument("--cuts", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--out", type=pathlib.Path, required=True); ap.add_argument("--batch", type=int, default=1); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None); ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--chunk-prefill", type=int, default=2048, help="prefill chunk (tokens) for the one-at-a-time cached path; 0 = batched generate")
    a = ap.parse_args()
    texts = {}
    for l in a.problems.read_text().splitlines():
        if l.strip():
            r = json.loads(l); texts[(r["problem_id"], r.get("family", "original"), int(r.get("variant", 0)))] = r["problem"]
    recs = [json.loads(l) for p in a.records for l in p.read_text().splitlines() if l.strip()][: a.limit]
    device, dtype = pick_device(a.device); t0 = time.time(); tok, model = load(a.model, device, dtype, a.local_only)
    force_ids = tok(FORCE_SUFFIX, add_special_tokens=False)["input_ids"]; pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"[cuts-hf] {a.model} on {device}/{dtype} loaded in {time.time()-t0:.0f}s; {len(recs)} records", flush=True)
    jobs = []
    for i, r in enumerate(recs):
        r["cuts"] = {}; ids = r.get("token_ids")
        if not ids: continue
        key = (r["problem_id"], r.get("family", "original"), int(r.get("variant", 0)))
        prompt_text = tok.apply_chat_template(chat_messages(texts[key]), add_generation_prompt=True, enable_thinking=True, tokenize=False)
        prompt_ids = list(tok(prompt_text, add_special_tokens=False)["input_ids"])
        for c in a.cuts:
            if r["finish_reason"] != "length" and r["n_tokens"] <= c:
                r["cuts"][str(c)] = {"answer": r["answer"], "correct": r["correct"], "natural": True}
            else:
                jobs.append((i, c, prompt_ids + list(ids[:c]) + list(force_ids)))
    print(f"[cuts-hf] {len(jobs)} forced continuations", flush=True); t1 = time.time()
    jobs.sort(key=lambda j: len(j[2]))                        # similar lengths together -> little padding
    step = int(a.chunk_prefill)

    @torch.inference_mode()
    def greedy_chunked(seq: list[int]) -> list[int]:
        """Prefill in chunks of `step` tokens with the model's own KV cache, then greedy-decode FORCE_TOKENS one step at a time.
        Avoids the whole-prefix fp32 copy of Qwen3.5's torch linear attention (4+ GiB for one 8k sequence: shards v3 OOMed at batch 1)."""
        x = torch.tensor([seq], device=model.device); cache = None; logits = None
        for s0 in range(0, len(seq), step):
            o = model(input_ids=x[:, s0:s0 + step], past_key_values=cache, use_cache=True, return_dict=True); cache = o.past_key_values; logits = o.logits[:, -1]
        out = []
        eos = set(x for x in [tok.eos_token_id] if x is not None)
        gc = getattr(model, "generation_config", None)
        if gc is not None and gc.eos_token_id is not None: eos |= set(gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id])
        for _ in range(FORCE_TOKENS):
            nxt = int(logits.argmax(-1)[0]); out.append(nxt)
            if nxt in eos: break
            o = model(input_ids=torch.tensor([[nxt]], device=model.device), past_key_values=cache, use_cache=True, return_dict=True); cache = o.past_key_values; logits = o.logits[:, -1]
        del cache
        return out

    for s in range(0, len(jobs), a.batch):
        chunk = jobs[s:s + a.batch]; L = max(len(j[2]) for j in chunk)
        try:
            if step > 0:                                              # chunked path (default): one sequence at a time
                for (i, c, seq) in chunk:
                    ans = parse_answer("</think>\nANSWER:" + tok.decode(greedy_chunked(seq), skip_special_tokens=True)); gold = recs[i]["gold"]
                    recs[i]["cuts"][str(c)] = {"answer": ans, "correct": is_correct(ans, gold), "natural": False}
            else:                                                     # batched generate (fine for standard-attention models)
                inp = torch.full((len(chunk), L), pad, dtype=torch.long); att = torch.zeros((len(chunk), L), dtype=torch.long)
                for row, (_, _, seq) in enumerate(chunk):
                    inp[row, L - len(seq):] = torch.tensor(seq); att[row, L - len(seq):] = 1      # left padding
                with torch.inference_mode():
                    out = model.generate(input_ids=inp.to(model.device), attention_mask=att.to(model.device), do_sample=False, max_new_tokens=FORCE_TOKENS, pad_token_id=pad)
                for row, (i, c, _) in enumerate(chunk):
                    ans = parse_answer("</think>\nANSWER:" + tok.decode(out[row, L:], skip_special_tokens=True)); gold = recs[i]["gold"]
                    recs[i]["cuts"][str(c)] = {"answer": ans, "correct": is_correct(ans, gold), "natural": False}
                del out
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache() if device == "cuda" else None
            for (i, c, _) in chunk: recs[i]["cuts"].setdefault(str(c), {"answer": None, "correct": False, "natural": False, "error": "oom"})
            print(f"[cuts-hf] OOM at batch {len(chunk)}, L={L}; marked", flush=True)
        if (s // a.batch) % 20 == 0: print(f"[cuts-hf] {s+len(chunk)}/{len(jobs)} {(time.time()-t1)/(s+len(chunk)):.2f}s each", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps({k: v for k, v in r.items() if k not in ("token_ids", "text")}) + "\n" for r in recs))
    print(f"[cuts-hf] wrote {a.out} ({len(recs)} records) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
