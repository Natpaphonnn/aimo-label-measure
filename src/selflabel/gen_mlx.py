#!/usr/bin/env python3
"""Sample model answers on a Mac with mlx-lm (continuous batching, resume, power/time guards).

Protocol: organizers' solve prompt as system message, chat template with thinking enabled,
temperature 1.0, top_k 40, top_p 0.95 (FAQ 12), a token cap (documented deviation from the
100k-token regime) and a greedy forced-answer continuation for capped samples.

Usage:
  python src/selflabel/gen_mlx.py --model Qwen/Qwen3.5-4B --problems data/pool/problems.jsonl \
      --out runs/gen/Qwen_Qwen3.5-4B/originals.jsonl --n-samples 10 --max-tokens 24576 --batch 8 \
      --require-ac --until 07:30 --stop-file runs/STOP

Input jsonl rows need: problem_id, problem (text), answer (int or "NaN"); optional family, variant.
Output jsonl rows: one per finished sample (full generated text included); resume skips done keys.
"""
from __future__ import annotations
import argparse, re, datetime as dt, json, pathlib, subprocess, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from selflabel.answers import parse_answer, is_correct
from selflabel.prompts import FORCE_SUFFIX, FORCE_TOKENS, chat_messages

def on_ac_power() -> bool:
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return "AC Power" in out

def battery_pct() -> int:
    m = re.search(r"(\d+)%", subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout)
    return int(m.group(1)) if m else 100

def parse_until(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    h, m = map(int, s.split(":"))
    now = dt.datetime.now(); t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > now else t + dt.timedelta(days=1)

def key(r: dict) -> tuple:
    return (r["problem_id"], r.get("family", "original"), int(r.get("variant", 0)), int(r["sample_idx"]))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--problems", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n-samples", type=int, default=10); ap.add_argument("--sample-start", type=int, default=0, help="first sample index (lets the Mac take indices Kaggle does not)")
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--batch", type=int, default=8, help="concurrent sequences")
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--limit", type=int, default=None, help="first N problems (smoke test)")
    ap.add_argument("--require-ac", action="store_true"); ap.add_argument("--min-battery", type=int, default=None, help="on battery, stop below this percent (the author 2026-10-22: battery runs allowed)"); ap.add_argument("--until", default=None, help="stop inserting new work at HH:MM local")
    ap.add_argument("--stop-file", type=pathlib.Path, default=pathlib.Path("runs/STOP"))
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    problems = [json.loads(l) for l in a.problems.read_text().splitlines() if l.strip()][: a.limit]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        for l in a.out.read_text().splitlines():
            if l.strip():
                done.add(key(json.loads(l)))
    # round-robin over sample index so partial runs cover every problem evenly
    jobs = [dict(problem_id=p["problem_id"], family=p.get("family", "original"), variant=int(p.get("variant", 0)),
                 sample_idx=s, text=p["problem"], answer=p["answer"])
            for s in range(a.sample_start, a.n_samples) for p in problems]
    jobs = [j for j in jobs if key(j) not in done]
    print(f"[gen_mlx] {len(problems)} problems, {len(jobs)} samples to do ({len(done)} already done)", flush=True)
    if not jobs:
        return

    t_load = time.time()
    model, tokenizer = load(a.model)
    # stop on the union of tokenizer EOS ids and the ids declared in (generation_)config.json
    eos_ids = set(tokenizer.eos_token_ids)
    for name in ("generation_config.json", "config.json"):
        cfg_path = pathlib.Path(a.model) / name
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text()); cfg = cfg.get("text_config", cfg) if name == "config.json" else cfg
            v = cfg.get("eos_token_id"); eos_ids |= set(v if isinstance(v, list) else [v]) if v is not None else set()
    eos_ids.discard(None)
    print(f"[gen_mlx] loaded {a.model} in {time.time() - t_load:.0f}s; eos ids {sorted(eos_ids)} = {[tokenizer.decode([t]) for t in sorted(eos_ids)]}", flush=True)
    mx.random.seed(a.seed)
    sampler = make_sampler(temp=1.0, top_p=0.95, top_k=40)
    greedy = make_sampler(temp=0.0)
    gen = BatchGenerator(model, max_tokens=a.max_tokens, stop_tokens=[[t] for t in sorted(eos_ids)],
                         sampler=sampler, completion_batch_size=a.batch, prefill_batch_size=2, prefill_step_size=1024)

    def encode(job: dict) -> list[int]:
        return tokenizer.apply_chat_template(chat_messages(job["text"]), add_generation_prompt=True, enable_thinking=True)

    force_ids = tokenizer.encode(FORCE_SUFFIX, add_special_tokens=False)
    until = parse_until(a.until); t0 = time.time()
    active: dict[int, dict] = {}      # uid -> job state (tokens list, started)
    forced: dict[int, dict] = {}      # uid of forced continuation -> job state
    queue = list(jobs); n_done = 0; n_tokens = 0; stop_reason = None; last_check = 0.0

    def should_stop() -> str | None:
        if a.stop_file.exists(): return "stop file"
        if a.require_ac and not on_ac_power(): return "not on AC power"
        if a.min_battery is not None and not on_ac_power() and battery_pct() < a.min_battery: return f"battery below {a.min_battery}%"
        if until and dt.datetime.now() >= until: return f"deadline {a.until}"
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes: return "max minutes"
        return None

    def fill() -> None:
        while queue and len(active) + len(forced) < a.batch and stop_reason is None:
            job = queue.pop(0)
            uid = gen.insert([encode(job)], [a.max_tokens])[0]
            active[uid] = {**job, "tokens": [], "t_start": time.time()}

    def write(job: dict) -> None:
        nonlocal n_done
        text = tokenizer.decode(job["tokens"])
        ans = parse_answer(text)
        rec = {"model_id": a.model, "problem_id": job["problem_id"], "family": job["family"], "variant": job["variant"],
               "sample_idx": job["sample_idx"], "gold": job["answer"], "n_tokens": len(job["tokens"]),
               "finish_reason": job["finish_reason"], "answer": ans, "correct": is_correct(ans, job["answer"]),
               "forced_answer": job.get("forced_answer"), "forced_correct": job.get("forced_correct"),
               "seconds": round(time.time() - job["t_start"], 1), "max_tokens": a.max_tokens, "seed": a.seed,
               "sampling": {"temperature": 1.0, "top_k": 40, "top_p": 0.95}, "backend": "mlx-lm", "text": text, "token_ids": list(job["tokens"])}
        with a.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_done += 1

    fill()
    try:
        while active or forced:
            if time.time() - last_check > 20:
                last_check = time.time(); stop_reason = stop_reason or should_stop()
                if stop_reason: queue.clear()
                el = time.time() - t0
                print(f"[gen_mlx] done {n_done} | active {len(active)}+{len(forced)} | queued {len(queue)} | {n_tokens/max(el,1):.0f} tok/s | {el/60:.1f} min" + (f" | stopping: {stop_reason}" if stop_reason else ""), flush=True)
            responses = gen.next_generated()
            if not responses and not active and not forced:
                break
            for r in responses:
                if r.uid in forced:
                    job = forced[r.uid]
                    if r.finish_reason != "stop" and r.token is not None:
                        job["ftokens"].append(r.token)
                    if r.finish_reason is not None:
                        ftext = tokenizer.decode(job["ftokens"])
                        fa = parse_answer("</think>\nANSWER:" + ftext)
                        job["forced_answer"] = fa; job["forced_correct"] = is_correct(fa, job["answer"])
                        del forced[r.uid]; write(job); fill()
                    continue
                job = active.get(r.uid)
                if job is None:
                    continue
                if r.finish_reason != "stop" and r.token is not None:
                    job["tokens"].append(r.token); n_tokens += 1
                if r.finish_reason is not None:
                    job["finish_reason"] = r.finish_reason
                    del active[r.uid]
                    if r.finish_reason == "length" and stop_reason != "abort":
                        # greedy forced answer from the cached context
                        job["ftokens"] = []
                        fuid = gen.insert([force_ids], [FORCE_TOKENS], caches=[r.prompt_cache], samplers=[greedy])[0]
                        forced[fuid] = job
                    else:
                        write(job); fill()
            if stop_reason and not queue and (len(active) + len(forced)) and time.time() - t0 > 0 and stop_reason != "abort":
                # a stop was requested: let running sequences finish (they are checkpointed on completion)
                pass
    except KeyboardInterrupt:
        stop_reason = "keyboard interrupt"
    finally:
        gen.close()
        el = time.time() - t0
        print(f"[gen_mlx] finished: {n_done} samples written, {n_tokens} tokens, {n_tokens/max(el,1):.0f} tok/s, {el/60:.1f} min, stop={stop_reason}", flush=True)

if __name__ == "__main__":
    main()
