#!/usr/bin/env python3
"""Driver for the two-stage white-box analysis (DECISIONS #44/#45): collects every available (records, states, cuts) block per
model from the known output locations, runs analysis/wb_two_stage.py for both stage-1 targets, and writes
notes/results/wb_two_stage_<tag>.json + .md (stage-1 AUCs, stage-2 arms x K, paired statistics).

Usage: python analysis/run_wb.py --tag v1 [--models qwen skywork deepseek olmo]
Blocks are skipped silently when their files do not exist yet, so the same command works as data arrive.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from wb_two_stage import load_model_block, run  # noqa: E402
from datapaths import need  # noqa: E402

K_OUT = ROOT / "runs/kaggle"
# (model_id, records file(s), states stem(s), cuts file or None, cut)
BLOCKS = {
    "qwen": [("Qwen/Qwen3.5-4B", ROOT / "runs/pilot/qwen_amc_s01_cut3072.jsonl", ROOT / "runs/pilot/qwen_pilot_states", ROOT / "runs/pilot/qwen_pilot.cuts.jsonl", 3072),
             ("Qwen/Qwen3.5-4B", ROOT / "runs/pilot/qwen_amc_rest_cut3072.jsonl", ROOT / "runs/pilot/qwen_rest_states", ROOT / "runs/pilot/qwen_rest.cuts.jsonl", 3072),
             # test-time wave: 714 samples generated + states on the Mac (bf16), the other 158 on Kaggle (fp16; 24 Mac samples
             # recomputed there as a precision check, the Mac copy is used)
             ("Qwen/Qwen3.5-4B", [ROOT / "runs/gen/Qwen_Qwen3.5-4B/tt_wave_3072.jsonl", K_OUT / "kx-tt-qwen-rest/output/kx-tt-qwen-rest.jsonl"],
              [ROOT / "runs/features/Qwen_Qwen3.5-4B/tt_wave_states", K_OUT / "kx-states-tt-qwen-rest/output/qwen_tt_rest_states"], None, 3072)],
    "skywork": [("Skywork/Skywork-OR1-Math-7B", K_OUT / "kx-gen-skywork-amc/output/kx-gen-skywork-amc.jsonl", K_OUT / "kx-states-skywork-amc/output/skywork_amc_states", K_OUT / "kx-states-skywork-amc/output/skywork_amc.cuts.jsonl", 3072),
                ("Skywork/Skywork-OR1-Math-7B", K_OUT / "kx-tt-skywork/output/kx-tt-skywork.jsonl", K_OUT / "kx-states-tt-skywork/output/skywork_tt_states", None, 3072)],
    "deepseek": [("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", K_OUT / "kx-gen-deepseek-amc/output/kx-gen-deepseek-amc.jsonl", K_OUT / "kx-states-deepseek-amc/output/deepseek_amc_states", K_OUT / "kx-states-deepseek-amc/output/deepseek_amc.cuts.jsonl", 2560),
                 ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", K_OUT / "kx-tt-deepseek/output/kx-tt-deepseek.jsonl", K_OUT / "kx-states-tt-deepseek/output/deepseek_tt_states", None, 2560)],
    "olmo": [("allenai/Olmo-3-7B-Think", K_OUT / "kx-tt-olmo/output/kx-tt-olmo.jsonl", K_OUT / "kx-states-tt-olmo/output/olmo_tt_states", None, 2048)],
}


def collect(models: list[str]) -> pd.DataFrame:
    parts = []
    for name in models:
        for mid, rec, st, cuts, cut in BLOCKS[name]:
            rec = [r for r in (rec if isinstance(rec, list) else [rec]) if r.exists()]
            st = [s for s in (st if isinstance(st, list) else [st]) if pathlib.Path(str(s) + ".npz").exists()]
            if not (rec and st):
                print(f"[run_wb] skip a {name} block: records or states not there yet"); continue
            d = load_model_block(mid, [str(r) for r in rec], [str(s) for s in st], cut, [str(cuts)] if cuts and cuts.exists() else [])
            print(f"[run_wb] {name}: {len(d)} samples from {[r.name for r in rec]} / {[s.name for s in st]}, {d.problem_id.nunique()} problems"); parts.append(d)
    if not parts:   # nothing logged at all (e.g. the supplementary code without runs/): say what is missing
        mid, rec, st, cuts, cut = BLOCKS[models[0]][0]
        need(*(rec if isinstance(rec, list) else [rec]), *(st if isinstance(st, list) else [st]))
    return pd.concat(parts, ignore_index=True)


def table(res: dict) -> str:
    L = [f"stage-1 target: **{res['stage1']['target']}**", "", "| model | gold samples | share correct | AUC probe | AUC twin (K=8) |", "|---|---|---|---|---|"]
    for m, r in res["stage1"].items():
        if isinstance(r, dict) and "n_samples_gold" in r:
            f = lambda v: "n/a" if v is None else f"{v:.3f}"
            L.append(f"| {m} | {r['n_samples_gold']} | {f(r['share_correct'])} | {f(r['auc_probe'])} | {f(r['auc_twin'])} |")
    L += ["", "| K | n cases | problems | robust share | prior | bb | wb | hybrid | wb vs bb (delta, p, MDE80) | wb vs prior | hybrid vs bb |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res["stage2"].items():
        if not r.get("n"): L.append(f"| {k} | 0 | | | | | | | | | |"); continue
        ps = r["paired_last_seed"]; g = lambda a: f"{r[a]['acc_mean']:.3f} ({r[a]['acc_sd']:.3f})"
        st = lambda key: f"{ps[key]['delta']:+.3f}, p={ps[key]['mcnemar_p']:.3f}, {ps[key]['mde80']:.2f}"
        L.append(f"| {k} | {r['n']} | {r['n_problems']} | {r['robust_share']:.2f} | {r['prior_acc']:.3f} | {g('bb')} | {g('wb')} | {g('hybrid')} | {st('wb_vs_bb')} | {st('wb_vs_prior')} | {st('hybrid_vs_bb')} |")
    fam = [(k, c, r["paired_last_seed"][c]["mcnemar_p"]) for k, r in res["stage2"].items() if r.get("n") for c in ("wb_vs_bb", "wb_vs_prior")]
    if fam:   # Holm over the primary family (wb vs bb and wb vs prior at every K), the correction the report uses
        order = sorted(range(len(fam)), key=lambda i: fam[i][2]); adj = [0.0] * len(fam); run_max = 0.0
        for rank, i in enumerate(order):
            run_max = max(run_max, min(1.0, (len(fam) - rank) * fam[i][2])); adj[i] = run_max
        L += ["", f"Holm-adjusted p over the {len(fam)} primary comparisons: " + "; ".join(f"{k} {c} {p:.3f}" for (k, c, _), p in zip(fam, adj))]
    for k, r in res["stage2"].items():
        lc = r.get("label_curve")
        if not lc: continue
        ns = sorted({n for a in lc.values() for n in a}, key=int)
        L += ["", f"label-efficiency curve at {k} (held-out accuracy, stage 2 trained on n labelled cases, 20 draws):", "",
              "| arm | " + " | ".join(f"n={n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
        for a, v in lc.items():
            L.append(f"| {a} | " + " | ".join(f"{v[n][0]:.3f}" if n in v else "" for n in ns) + " |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--models", nargs="+", default=list(BLOCKS))
    ap.add_argument("--targets", nargs="+", default=["correct", "consistent"])
    a = ap.parse_args()
    df = collect(a.models)
    need(ROOT / "data/labels/case_labels.csv")
    labels = pd.read_csv(ROOT / "data/labels/case_labels.csv")
    out = {"tag": a.tag, "models": a.models, "n_samples": int(len(df)), "results": {}}
    md = [f"# Two-stage white-box analysis `{a.tag}` (analysis/run_wb.py)", "",
          f"Samples: {len(df)} over {df.groupby('model_id').problem_id.nunique().to_dict()} problems per model. Labels: data/labels/case_labels.csv.",
          "Stage 2 = grouped 5-fold CV by problem x 3 seeds; accuracy mean (sd over seeds); paired stats on the last seed.", ""]
    for t in a.targets:
        res = run(df, labels, None, t); out["results"][t] = res; md += [f"## target = {t}", "", table(res), ""]
    (ROOT / f"notes/results/wb_two_stage_{a.tag}.json").write_text(json.dumps(out, indent=1, default=str))
    (ROOT / f"notes/results/wb_two_stage_{a.tag}.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
