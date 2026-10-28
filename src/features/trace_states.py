#!/usr/bin/env python3
"""White-box features from saved generations (DECISIONS #33): residual-stream states of the model's OWN reasoning trace.

For every record with `token_ids` (gen_mlx.py output), one teacher-forced forward pass over chat prompt + generated tokens gives
the hidden states at every position; these equal the states the model had while generating those tokens (same tokens, same
weights), so this reproduces what forward hooks would capture at test time, but for all cut points at once. Per record, per
cut c in --cuts (and the natural end when the sample finished before c) and per layer in --layers:
  trace_c  = mean of the layer output over the last --window generated tokens before c      (float16, hidden)
  answer_c = the layer output at the token that emitted the last generated token before c  (float16, hidden; see hooks.py)
plus prompt-only states (last prompt token) so that the N2 prompt probe can be run as a control.
Output: <out>.npz with arrays [n_records, n_cuts, n_layers, hidden] and the record keys; <out>.jsonl with the per-record meta.
Memory: --max-len tokens per pass; longer traces are processed in chunks with the KV cache (DynamicCache), so a 9k-token
trace on MPS fits in a few GB.
Usage: python src/features/trace_states.py --model data/models/Qwen3.5-4B --records runs/gen/Qwen_Qwen3.5-4B/night1_originals_8k.jsonl \
          --problems data/pool/night1_qwen35.jsonl --cuts 512 1024 2048 4096 8192 --layers 0.25 0.5 0.75 --out runs/features/Qwen_Qwen3.5-4B/night1_states
"""
from __future__ import annotations
import argparse, gc, json, os, pathlib, signal, subprocess, sys, time
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from selflabel.prompts import chat_messages
from features.prefill import pick_device, load


def layer_indices(model, fracs: list[float]) -> list[int]:
    n = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else model.config.text_config.num_hidden_layers
    return sorted({min(n - 1, max(0, round(f * n) - 1)) for f in fracs})


@torch.inference_mode()
def states_for(model, ids: list[int], layers: list[int], positions: list[int], window: int, device: str, max_len: int) -> dict[int, np.ndarray]:
    """Returns {layer: array[len(positions), 2, hidden]}: [:, 0] = mean of the layer output over the window of tokens ending at the
    position (inclusive), [:, 1] = the layer output at the position. hidden_states[l + 1] is the output of decoder layer l.
    The sequence is processed in chunks of max_len with the KV cache so long traces fit in memory."""
    x = torch.tensor([ids], device=device); want = sorted(set(positions)); H = hidden_size(model)
    sums = {l: {p: torch.zeros(H, device=device) for p in want} for l in layers}; last = {l: {} for l in layers}; cnt = {p: 0 for p in want}
    cache = None; done = 0   # the model builds its own cache type (Qwen3.5's hybrid layers need a special one)
    while done < len(ids):
        chunk = x[:, done: done + max_len]; T = chunk.shape[1]
        if done + T >= len(ids) and T % 256:   # last chunk: pad to a multiple of 256 so MPS reuses compiled graphs (a new shape per
            chunk = torch.cat([chunk, chunk[:, -1:].repeat(1, 256 - T % 256)], 1)   # record grew the footprint to 15-20 GB); causal => states < T unchanged
        o = model(input_ids=chunk, past_key_values=cache, use_cache=True, output_hidden_states=True, return_dict=True); cache = o.past_key_values
        for l in layers:
            h = o.hidden_states[l + 1][0].float()
            for p in want:
                lo, hi = max(p - window + 1, 0), p + 1; a, b = max(lo, done), min(hi, done + T)
                if a < b:
                    sums[l][p] += h[a - done: b - done].sum(0)
                    if l == layers[0]: cnt[p] += b - a
                if done <= p < done + T: last[l][p] = h[p - done].cpu().numpy().astype(np.float16)
        done += T
    out = {}
    for l in layers:
        arr = np.zeros((len(positions), 2, H), np.float16)
        for i, p in enumerate(positions):
            arr[i, 0] = (sums[l][p] / max(cnt[p], 1)).cpu().numpy().astype(np.float16); arr[i, 1] = last[l][p]
        out[l] = arr
    return out


def hidden_size(model) -> int:
    c = model.config; return c.hidden_size if hasattr(c, "hidden_size") else c.text_config.hidden_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--records", type=pathlib.Path, nargs="+", required=True)
    ap.add_argument("--problems", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--cuts", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192]); ap.add_argument("--layers", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--window", type=int, default=64); ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--device", default=None); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--save-every", type=int, default=50, help="records between checkpoints (<out>.part.npz)")
    a = ap.parse_args()
    device, dtype = pick_device(a.device)
    texts = {}
    for l in a.problems.read_text().splitlines():
        if l.strip():
            r = json.loads(l); texts[(r["problem_id"], r.get("family", "original"), int(r.get("variant", 0)))] = r["problem"]
    recs = [json.loads(l) for p in a.records for l in p.read_text().splitlines() if l.strip()][: a.limit]
    recs = [r for r in recs if r.get("token_ids")]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); tok, model = load(a.model, device, dtype, a.local_only); layers = layer_indices(model, a.layers)
    print(f"[states] {a.model} on {device}/{dtype} loaded in {time.time()-t0:.0f}s; {len(recs)} records; layers {layers}; cuts {a.cuts}", flush=True)
    H = hidden_size(model)
    gcfg = getattr(model, "generation_config", None); eos = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    if gcfg is not None and gcfg.eos_token_id is not None: eos |= set(gcfg.eos_token_id if isinstance(gcfg.eos_token_id, list) else [gcfg.eos_token_id])
    trace = np.zeros((len(recs), len(a.cuts), len(layers), H), np.float16); answer = np.zeros_like(trace); prompt_last = np.zeros((len(recs), len(layers), H), np.float16)
    meta = []
    # Checkpoint/resume (2026-10-25): every --save-every records and on SIGTERM (mem_guard, the author's other jobs) the arrays so far go
    # to <out>.part.npz/.jsonl; a restart with the same records/cuts/layers continues from there instead of from zero.
    part = pathlib.Path(str(a.out) + ".part.npz"); part_meta = pathlib.Path(str(a.out) + ".part.jsonl"); start = 0
    if part.exists() and part_meta.exists():
        z = np.load(part); pm = [json.loads(l) for l in part_meta.read_text().splitlines() if l.strip()]
        same = list(z["cuts"]) == list(a.cuts) and list(z["layers"]) == list(layers) and len(pm) <= len(recs) and z["trace"].shape[1:] == trace.shape[1:] \
            and all((m["problem_id"], m["sample_idx"]) == (r["problem_id"], r["sample_idx"]) for m, r in zip(pm, recs))
        if same:
            start = len(pm); trace[:start] = z["trace"][:start]; answer[:start] = z["answer"][:start]; prompt_last[:start] = z["prompt_last"][:start]; meta = pm
            print(f"[states] resuming from {start}/{len(recs)} ({part.name})", flush=True)
        else:
            print(f"[states] ignoring {part.name}: different records/cuts/layers", flush=True)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    def save_part(n: int) -> None:
        tmp = pathlib.Path(str(a.out) + ".part.tmp.npz")
        np.savez(tmp, trace=trace[:n], answer=answer[:n], prompt_last=prompt_last[:n], layers=np.array(layers), cuts=np.array(a.cuts))
        part_meta.write_text("".join(json.dumps(m) + "\n" for m in meta[:n])); tmp.replace(part)
    t_run = time.time()
    for i, r in enumerate(recs):
        if i < start: continue
        key = (r["problem_id"], r.get("family", "original"), int(r.get("variant", 0)))
        prompt_text = tok.apply_chat_template(chat_messages(texts[key]), add_generation_prompt=True, enable_thinking=True, tokenize=False)
        prompt_ids = list(tok(prompt_text, add_special_tokens=False)["input_ids"])   # same ids gen_mlx.py fed the model
        gen = list(r["token_ids"])
        if gen and gen[-1] in eos: gen = gen[:-1]   # vLLM keeps the EOS id; HF/MLX records do not; the bundle reads finished samples before EOS
        P = len(prompt_ids); ids = prompt_ids + gen
        # hooks.py convention: with m = min(c, n_gen) generated tokens inside the cut, the last state computed during generation is
        # that of generated token m - 1 (index P + m - 2); the window ends there. m <= 1 falls back to the last prompt position.
        positions = [P - 1] + [max(P - 1, P + min(c, len(gen)) - 2) for c in a.cuts]
        ids = ids[: max(positions) + 1]   # causal model: tokens after the last needed position cannot change the states read
        st = states_for(model, ids, layers, positions, a.window, device, a.max_len)
        for j, l in enumerate(layers):
            prompt_last[i, j] = st[l][0, 1]
            for k in range(len(a.cuts)):
                trace[i, k, j] = st[l][k + 1, 0]; answer[i, k, j] = st[l][k + 1, 1]
        meta.append({"problem_id": r["problem_id"], "family": r.get("family", "original"), "variant": int(r.get("variant", 0)), "sample_idx": r["sample_idx"],
                     "n_gen": len(gen), "n_prompt": P, "cut_reached": [int(min(c, len(gen))) for c in a.cuts], "finish_reason": r.get("finish_reason")})
        del st; gc.collect()   # 2026-10-22: without releasing the allocator cache the MPS process grew to 20 GB over ~110 records
        if device == "mps": torch.mps.empty_cache()
        elif device == "cuda": torch.cuda.empty_cache()
        if stop["flag"]:
            save_part(i + 1); print(f"[states] SIGTERM: saved {i+1}/{len(recs)} to {part.name}; rerun the same command to resume", flush=True); sys.exit(3)
        if (i + 1) % a.save_every == 0: save_part(i + 1)
        if (i + 1) % 10 == 0 or i == start:
            mem = ""
            if device == "mps":
                fp = subprocess.run(["top", "-l", "1", "-pid", str(os.getpid()), "-stats", "mem"], capture_output=True, text=True).stdout.split()[-1:]
                mem = f" mps {torch.mps.driver_allocated_memory()/2**30:.1f} GB footprint {fp[0] if fp else '?'}"
            print(f"[states] {i+1}/{len(recs)} {(time.time()-t_run)/(i+1-start):.1f}s per record{mem}", flush=True)
    np.savez_compressed(str(a.out) + ".npz", trace=trace, answer=answer, prompt_last=prompt_last, layers=np.array(layers), cuts=np.array(a.cuts), window=a.window)
    pathlib.Path(str(a.out) + ".jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    part.unlink(missing_ok=True); part_meta.unlink(missing_ok=True)
    print(f"[states] wrote {a.out}.npz ({trace.nbytes*2/1e6:.0f} MB) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
