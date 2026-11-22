#!/usr/bin/env python3
"""Regeneration manifest for the cached inputs of the report (revision of #21, R-Q4/M3) -> notes/results/cache/regeneration.json
and notes/results/cache/cases_158.csv / cases_139.csv.

For every (records, states) block that analysis/run_wb.py reads, the manifest records how it was produced: the generation
settings of the Kaggle kernel (tools/kaggle_push_gen.py CONFIG: model, token cap, backend, dtype, seed, samples per problem,
problem file) or of the Mac run, and the environment of the states job (tools/kaggle_pack.py running
kaggle/jobs/states_cuts_job.py, or src/features/trace_states.py on the Mac), plus the sample ids it contributes. Problem
texts are not included (third-party licences); problem ids are sha256 prefixes of the problem text, so rebuilding the pool
with tools/build_problem_pool.py and tools/build_official_problem_files.py gives the same ids.
Needs runs/ (the logged generations); the outputs are shipped in notes/results/cache/.
"""
import ast, json, pathlib, re, sys
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from datapaths import RESULTS, need  # noqa: E402
from run_wb import BLOCKS  # noqa: E402

KEEP = ("model", "max_tokens", "backend", "dtype", "seed", "n_samples", "out_name", "system_prompt", "force_suffix", "force_tokens", "vllm_spec")


def kernel_info(path: pathlib.Path) -> dict:
    k = path.parent.parent / "kernel.py" if path.parent.name == "output" else None
    if not k or not k.exists(): return {"produced_on": "Mac (Apple M-series, bf16)", "script": "src/selflabel/ (gen_mlx.py, force_at_cuts.py) and src/features/trace_states.py; see tools/supplementary_README.md"}
    t = k.read_text(); info = {"kaggle_kernel": k.parent.name, "gpu": "Kaggle T4 (fp16)"}
    m = re.search(r'CONFIG = json\.loads\("""(.*?)"""\)', t, re.S)
    if m:
        c = json.loads(ast.literal_eval('"""' + m.group(1) + '"""')); info["generation"] = {x: c[x] for x in KEEP if x in c}
        info["command"] = (f"python tools/kaggle_push_gen.py --model {c.get('model')} --problems <pool file> --max-tokens {c.get('max_tokens')} "
                           f"--backend {c.get('backend')} --dtype {c.get('dtype')} --seed {c.get('seed')} --slug {k.parent.name} --push")
    m = re.search(r"os\.environ\.update\((\{.*?\})\)", t)
    if m:
        env = json.loads(m.group(1)); info["states_job_env"] = env
        info["command"] = f"python tools/kaggle_pack.py ... (runs kaggle/jobs/states_cuts_job.py with env {json.dumps(env)})"
    return info


def main():
    out = {"blocks": []}
    for name, blocks in BLOCKS.items():
        if name == "olmo": continue
        for mid, rec, st, cuts, cut in blocks:
            recs = rec if isinstance(rec, list) else [rec]; sts = st if isinstance(st, list) else [st]
            out["blocks"].append({"model": mid, "cut": cut,
                                  "records": [{"file": str(r.relative_to(ROOT)), **kernel_info(r)} for r in recs],
                                  "states": [{"file": str(s.relative_to(ROOT)) + ".npz", **kernel_info(pathlib.Path(str(s)))} for s in sts],
                                  "cuts_file": str(cuts.relative_to(ROOT)) if cuts else None})
    cache = RESULTS / "cache"; cache.mkdir(parents=True, exist_ok=True)
    (cache / "regeneration.json").write_text(json.dumps(out, indent=1))
    need(cache / "samples.csv.gz", ROOT / "data/labels/case_labels.csv")
    s = pd.read_csv(cache / "samples.csv.gz")
    lab = pd.read_csv(ROOT / "data/labels/case_labels.csv").drop_duplicates(["model_id", "problem_id"])
    for K, name in ((4, "cases_158.csv"), (8, "cases_139.csv")):
        g = s.sort_values("sample_idx").groupby(["model_id", "problem_id"]).sample_idx.apply(lambda v: list(v)[:K])
        g = g[g.map(len) >= K].reset_index().rename(columns={"sample_idx": f"sample_ids_first_{K}"})
        c = g.merge(lab[["model_id", "problem_id", "y", "source"]], on=["model_id", "problem_id"])
        c[f"sample_ids_first_{K}"] = c[f"sample_ids_first_{K}"].map(lambda v: " ".join(map(str, v)))
        c.to_csv(cache / name, index=False); print(f"{name}: {len(c)} cases, {c.problem_id.nunique()} problems")
    print(f"regeneration.json: {len(out['blocks'])} blocks")
    write_tex(out, pd.read_csv(cache / "cases_158.csv"), pd.read_csv(cache / "cases_139.csv"))


SHORT = {"Qwen/Qwen3.5-4B": "Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B": "Skywork-OR1-Math-7B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-R1-0528-Qwen3-8B"}
esc = lambda s: (str(s).replace("{", r"\{").replace("}", r"\}").replace("_", r"\_").replace("#", r"\#").replace("%", r"\%")
                 .replace("/", r"/\allowbreak{}").replace(",", r",\allowbreak{}"))


def write_tex(man: dict, c158: pd.DataFrame, c139: pd.DataFrame) -> None:
    """PDF appendix (camera-ready of #21, M3/R-Q4): how every input was produced and the exact case/sample IDs."""
    L = [r"\section{Regeneration commands and case/sample IDs}\label{app:regen}",
         r"Generated by \texttt{tools/export\_regeneration\_manifest.py} from the logged runs (machine-readable copy:",
         r"\texttt{notes/results/cache/regeneration.json}, \texttt{cases\_158.csv}, \texttt{cases\_139.csv} in the code).",
         r"\paragraph{Commands.} Problem pool: \texttt{bash tools/clone\_external.sh; python tools/build\_problem\_pool.py;",
         r"python tools/build\_official\_problem\_files.py} (problem IDs are sha256 prefixes of the problem text). Generation on Kaggle:",
         r"\texttt{python tools/kaggle\_push\_gen.py -{}-model M -{}-problems P -{}-max-tokens T -{}-backend B -{}-dtype D -{}-seed S -{}-push} with the",
         r"settings below; on the laptop: \texttt{src/selflabel/gen\_mlx.py}. Residual states: \texttt{tools/kaggle\_pack.py} running",
         r"\texttt{kaggle/jobs/states\_cuts\_job.py} with the environment below (laptop: \texttt{src/features/trace\_states.py}).",
         r"Stage~1 and 2, nested: \texttt{python analysis/wb\_nested.py -{}-jobs 3}; from the cached scores without states:",
         r"\texttt{python analysis/wb\_nested.py -{}-from-cache}.", "",
         r"{\scriptsize\setlength{\tabcolsep}{2pt}\begin{longtable}{p{0.15\linewidth}p{0.35\linewidth}p{0.45\linewidth}}", r"\toprule",
         r"model, cut & records (generation) & states (extraction) \\", r"\midrule", r"\endhead"]
    def gen(r):
        g = r.get("generation")
        if g: return esc(f"{r.get('kaggle_kernel')} (T4 fp16): max_tokens {g.get('max_tokens')}, {g.get('backend')}, {g.get('dtype')}, seed {g.get('seed')}")
        if r.get("kaggle_kernel"): return esc(f"{r['kaggle_kernel']} (T4 fp16)")
        return "laptop (bf16)"
    for b in man["blocks"]:
        recs = "; ".join(esc(pathlib.Path(r["file"]).name) + ": " + gen(r) for r in b["records"])
        sts = "; ".join(esc(pathlib.Path(s["file"]).name) + ": " + (esc(f"{s['kaggle_kernel']} (T4 fp16), env " + json.dumps(s["states_job_env"])) if s.get("states_job_env") else gen(s)) for s in b["states"])
        L.append(f"{SHORT.get(b['model'], esc(b['model']))}, {b['cut']} & {recs} & {sts} \\\\")
    L += [r"\bottomrule", r"\end{longtable}}", r"\addtocounter{table}{-1}   % uncaptioned longtable: keep table numbers consecutive"]
    first = lambda c, K: c[f"sample_ids_first_{K}"].map(lambda v: [int(x) for x in str(v).split()])
    s4 = dict(zip(zip(c158.model_id, c158.problem_id), first(c158, 4))); s8 = dict(zip(zip(c139.model_id, c139.problem_id), first(c139, 8)))
    odd = {k for k, v in s4.items() if v != list(range(4))} | {k for k, v in s8.items() if v != list(range(8))}
    odd_ix = "; ".join(sorted({" ".join(map(str, s4[k])) for k in odd if k in s4}))
    L += [r"\paragraph{Case sets.} A case at $K$ uses its first $K$ logged samples by index, i.e.\ indices $0,\dots,K-1$ "
          + (f"(139-case set: $K=8$), except the cases marked $\\ddagger$, which have no sample 0 in our records and use indices {odd_ix} at $K\\le4$. " if odd else "(139-case set: $K=8$). ")
          + r"Problem ID (16 hex), label (R robust, N non-robust), $\ast$ = official \texttt{train-main-v2} label (else byeolki), $\dagger$ = not in the 139-case set (fewer than 8 samples)."]
    in139 = set(zip(c139.model_id, c139.problem_id))
    for m, h in c158.groupby("model_id"):
        items = [f"{pid}~{'R' if y else 'N'}{'$^\\ast$' if str(s).startswith('train-main') else ''}{'$^\\dagger$' if (m, pid) not in in139 else ''}{'$^\\ddagger$' if (m, pid) in odd else ''}"
                 for pid, y, s in zip(h.problem_id, h.y.astype(bool), h.source)]
        L.append(f"\\par\\noindent\\textbf{{{SHORT.get(m, m)}}} ({len(h)} cases): {{\\scriptsize\\ttfamily " + ", ".join(i.replace("_", "\\_") for i in items) + "}")
    (ROOT / "notes/report/latex/app_regen.tex").write_text("\n".join(L) + "\n"); print("app_regen.tex written")


if __name__ == "__main__":
    main()
