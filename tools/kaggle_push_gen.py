#!/usr/bin/env python3
"""Package a generation job (model x problems x samples) into a self-contained Kaggle kernel and push it.

Usage: python tools/kaggle_push_gen.py --model Skywork/Skywork-OR1-Math-7B --problems data/pool/problems.jsonl \
          --n-samples 10 --max-tokens 16384 --slug kx-gen-skywork-s0 --shard 0/3 [--backend vllm|hf] [--push]
The problems file (optionally variants) is embedded gzipped+base64 in the script; outputs land in
/kaggle/working/<out_name>.jsonl and are fetched with `kaggle kernels output <slug> -p runs/kaggle/<slug>`.
Every push is appended to notes/KAGGLE_LOG.md by hand with the GPU hours used (see kernel log).
"""
import argparse, base64, gzip, json, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from selflabel.prompts import SOLVE_SYSTEM, FORCE_SUFFIX, FORCE_TOKENS

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--problems", type=pathlib.Path, required=True)
ap.add_argument("--n-samples", type=int, default=10); ap.add_argument("--max-tokens", type=int, default=16384)
ap.add_argument("--slug", required=True); ap.add_argument("--shard", default="0/1", help="i/n: take every n-th job starting at i")
ap.add_argument("--backend", default="hf", choices=["vllm", "hf"], help="vllm: the current wheel is CUDA-13 and breaks Kaggle's torch (smoke 2026-10-15); pin a version via --vllm-spec"); ap.add_argument("--dtype", default="float16")
ap.add_argument("--chunk", type=int, default=None); ap.add_argument("--budget-hours", type=float, default=3.0)
ap.add_argument("--vllm-spec", default="vllm", help="pip spec for the vllm backend, e.g. 'vllm==0.8.5'"); ap.add_argument("--seed", type=int, default=20261015); ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--timeout", type=int, default=4 * 3600); ap.add_argument("--push", action="store_true")
ap.add_argument("--samples", default=None, help="a:b -> sample indices a..b-1 (default 0..n_samples)")
ap.add_argument("--exclude-done", type=pathlib.Path, nargs="*", default=[], help="jsonl files whose (problem_id, family, variant, sample_idx) are skipped")
ap.add_argument("--single-worker", action="store_true", help="hf backend: one process with device_map=auto over all GPUs (7-8B fp16 models do not fit one T4); default is one worker per GPU")
ap.add_argument("--force-chunk", type=int, default=1, help="hf backend: truncated prefixes per forced-answer generate call (1 avoids the Qwen3.5 linear-attention fp32 OOM)")
a = ap.parse_args()

probs = [json.loads(l) for l in a.problems.read_text().splitlines() if l.strip()][: a.limit]
jobs = [dict(problem_id=p["problem_id"], family=p.get("family", "original"), variant=int(p.get("variant", 0)), sample_idx=s, text=p["problem"], answer=p["answer"])
        for s in (range(*map(int, a.samples.split(":"))) if a.samples else range(a.n_samples)) for p in probs]
i, n = map(int, a.shard.split("/")); jobs = jobs[i::n]
done = {(r["problem_id"], r.get("family", "original"), int(r.get("variant", 0)), int(r["sample_idx"])) for f in a.exclude_done for l in f.read_text().splitlines() if l.strip() for r in [json.loads(l)]}
jobs = [j for j in jobs if (j["problem_id"], j["family"], j["variant"], j["sample_idx"]) not in done]
cfg = {"model": a.model, "max_tokens": a.max_tokens, "backend": a.backend, "dtype": a.dtype, "seed": a.seed,
       "out_name": f"{a.slug}.jsonl", "system_prompt": SOLVE_SYSTEM, "force_suffix": FORCE_SUFFIX, "force_tokens": FORCE_TOKENS,
       "budget_seconds": int(a.budget_hours * 3600), "vllm_spec": a.vllm_spec, "chunk": a.chunk or (64 if a.backend == "vllm" else 8),
       "answers_src": (ROOT / "src/selflabel/answers.py").read_text(), "per_gpu_workers": not a.single_worker, "force_chunk": a.force_chunk}
tpl = (ROOT / "kaggle/gen_kernel_template.py").read_text()
script = tpl.replace("__CONFIG__", json.dumps(cfg).replace("\\", "\\\\").replace('"""', '\\"\\"\\"')).replace("__JOBS_B64__", base64.b64encode(gzip.compress(json.dumps(jobs).encode())).decode())
kdir = ROOT / "runs/kaggle" / a.slug; kdir.mkdir(parents=True, exist_ok=True)
(kdir / "kernel.py").write_text(script)
(kdir / "kernel-metadata.json").write_text(json.dumps({"id": f"<kaggle-user>/{a.slug}", "title": a.slug, "code_file": "kernel.py", "language": "python",
    "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "true"}, indent=1))
print(f"{len(jobs)} jobs ({len(probs)} problems x {a.n_samples} samples, shard {a.shard}) -> {kdir}/kernel.py ({len(script)//1024} KB), model {a.model}, cap {a.max_tokens}, backend {a.backend}")
if a.push:
    kaggle = pathlib.Path.home() / "Library/Application Support/pipx/venvs/kaggle/bin/kaggle"
    r = subprocess.run([str(kaggle) if kaggle.exists() else "kaggle", "kernels", "push", "-p", str(kdir), "-t", str(a.timeout)], capture_output=True, text=True); print((r.stdout + r.stderr).strip())
