#!/usr/bin/env python3
"""Report figures (notes/report/latex/fig_*.pdf) from the result files; nothing is recomputed except the label simulation of Fig. 1.

Fig. 1  label instrument: P(sampled label = non-robust | no true degradation) vs true solve probability (analysis/label_noise_ceiling.py protocol).
Fig. 2  stage-1 probe AUC by layer and state kind, per model, with 8-sample agreement (notes/results/wb_diagnostics_<tag>.json).
Fig. 3  (a) balanced accuracy of stage 2 by K, nested CV (wb, bb, offsets; prior = 0.5; notes/results/wb_nested.json); (b) AUC of solve-rate-type
        predictors for the robustness label, monotone and band |x - 0.5|, with 95% CIs (notes/results/label_decomp_revision.json).
Usage: python analysis/make_figures.py --tag v2
"""
import argparse, json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "notes/report/latex"; OUT.mkdir(parents=True, exist_ok=True)
import sys; sys.path.insert(0, str(ROOT / "analysis"))
from datapaths import need  # noqa: E402
# reference palette (dataviz skill, light mode, first three categorical slots validate all-pairs); text and grid in neutral ink
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "prior": "#8a8984"}
MODEL_COLOR = {"Qwen/Qwen3.5-4B": C["blue"], "Skywork/Skywork-OR1-Math-7B": C["orange"], "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": C["aqua"]}
SHORT = {"Qwen/Qwen3.5-4B": "Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B": "Skywork-OR1-7B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-R1-8B"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "lines.linewidth": 1.6, "lines.markersize": 4.5})


def fig1() -> None:
    rng = np.random.default_rng(20261015); K, T, n = 10, 7, 40000; ps = np.linspace(0.5, 1.0, 26); nr, dr = [], []
    for p0 in ps:
        base = rng.binomial(K, p0, n) / K; acc = rng.binomial(K, p0, (n, T)) / K; md = (base[:, None] - acc).max(1)
        nr.append(np.mean(md >= 0.25 - 1e-9)); dr.append(np.mean((md > 0.10 + 1e-9) & (md < 0.25 - 1e-9)))
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    ax.plot(ps, nr, color=C["blue"], marker="o", markevery=5); ax.plot(ps, dr, color=C["prior"], marker="s", markevery=5)
    ax.text(0.52, nr[1] + 0.03, "labelled non-robust", color=C["ink"]); ax.text(0.52, dr[1] - 0.06, "dropped (between bands)", color=C["ink2"])
    i9 = int(np.argmin(abs(ps - 0.9))); ax.annotate(f"{nr[i9]:.0%} at p = 0.9", (ps[i9], nr[i9]), (0.8, 0.52), color=C["ink"],
                                                    arrowprops=dict(arrowstyle="-", color=C["ink2"], lw=0.6))
    ax.set_xlabel("true solve probability (no family truly degrades)"); ax.set_ylabel("share of sampled labels"); ax.set_ylim(0, 0.65)
    fig.tight_layout(); fig.savefig(OUT / "fig_label_noise.pdf"); plt.close(fig)


def fig2(tag: str) -> None:
    need(ROOT / f"notes/results/wb_diagnostics_{tag}.json", ROOT / f"notes/results/wb_balanced_{tag}.json")
    d = json.loads((ROOT / f"notes/results/wb_diagnostics_{tag}.json").read_text())["models"]
    fig, ax = plt.subplots(figsize=(3.3, 2.3)); xs = [0, 1, 2]
    for m, r in d.items():
        col = MODEL_COLOR.get(m, C["ink2"])
        ax.plot(xs, [r["blocks"][f"S:trace:{j}"] for j in xs], color=col, marker="o")
        ax.plot(xs, [r["blocks"][f"S:answer:{j}"] for j in xs], color=col, marker="o", ls=(0, (2, 2)), lw=1.0, mfc="white")
        ax.plot([3], [r["auc_twin8"]], color=col, marker="D", ls="none")
    ys = {m: r["blocks"]["S:trace:2"] for m, r in d.items()}; last = None
    for m in sorted(ys, key=ys.get, reverse=True):   # direct labels at the right end of the trace lines, kept 0.035 apart
        y = ys[m] if last is None else min(ys[m], last - 0.035); last = y
        ax.text(2.08, y, SHORT.get(m, m), color=C["ink"], va="center", fontsize=7)
    ax.set_xticks([0, 1, 2, 3], ["L/4", "L/2", "3L/4", "agreement\n(8 samples)"]); ax.set_xlim(-0.2, 3.3)
    ax.set_ylabel("AUC, correct vs incorrect"); ax.set_ylim(0.45, 1.0)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=C["ink2"], marker="o"), Line2D([], [], color=C["ink2"], ls=(0, (2, 2)), lw=1.0, marker="o", mfc="white")],
              ["trace state (last 64 tokens)", "last-token state"], loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_probe_layers.pdf"); plt.close(fig)


def fig3(tag: str) -> None:
    # revision of #21: (a) from the nested run (analysis/wb_nested.py), (b) from analysis/label_decomp_revision.py
    need(ROOT / "notes/results/wb_nested.json", ROOT / "notes/results/label_decomp_revision.json")
    bal = json.loads((ROOT / "notes/results/wb_nested.json").read_text())["per_K"]
    e = json.loads((ROOT / "notes/results/label_decomp_revision.json").read_text())["auc"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ks = sorted(bal, key=lambda k: int(k[1:])); x = [int(k[1:]) for k in ks]
    for arm, col, lab in [("wb", C["blue"], "white-box (probe)"), ("bb", C["orange"], "black-box (agreement)"), ("offsets", C["prior"], "per-model offsets")]:
        a.plot(x, [bal[k]["arms"][f"{arm}_bal"]["bacc"] for k in ks], color=col, marker="o", label=lab, ls="-" if arm != "offsets" else (0, (1, 1.5)))
    a.axhline(0.5, color=C["prior"], lw=1.2, ls=(0, (3, 2))); a.text(8, 0.505, "majority class", color=C["ink2"], ha="right", va="bottom", fontsize=7)
    a.set_xscale("log", base=2); a.set_xticks(x, [str(v) for v in x]); a.set_xlabel("samples per case K")
    a.set_ylabel("balanced accuracy (nested CV)"); a.set_ylim(0.40, 0.7); a.legend(fontsize=6.5, loc="lower left"); a.set_title("(a) stage 2 under a balanced label mix", fontsize=8, loc="left")
    names = [("base", "labellers' base acc. (8k)"), ("solve", "solve rate at our cut"), ("twin_mean", "agreement (8 samples)"), ("wb_mean", "probe mean (white-box)")]
    y = np.arange(len(names))
    for i, (k, lab) in enumerate(names):
        r = e[f"byeolki|{k}"]   # byeolki-protocol labels (R-S1); official cases are reported in the text
        for v, ci, col, dy, mk in ((r["auc"], r["ci"], C["blue"], 0.14, "o"), (r["band_auc"], r["band_ci"], C["orange"], -0.14, "s")):
            b.plot(ci, [i + dy] * 2, color=col, lw=1.2, alpha=0.6); b.plot([v], [i + dy], mk, color=col, ms=5)
            b.text(ci[1] + 0.01, i + dy, f"{v:.2f}", va="center", color=C["ink"], fontsize=6.5)
    b.plot([], [], "o", color=C["blue"], label="monotone x"); b.plot([], [], "s", color=C["orange"], label="band |x-0.5|"); b.legend(fontsize=6.5, loc="lower center", ncol=2)
    b.axvline(0.5, color=C["ink2"], lw=0.8); b.set_yticks(y, [l for _, l in names]); b.set_xlim(0.3, 0.9); b.set_ylim(-1.1, len(names) - 0.5); b.grid(axis="y", visible=False)
    b.set_xlabel("AUC for the robustness label, 95% CI (0.5 = chance)"); b.set_title("(b) what predicts the label", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_label_eff_decomp.pdf"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="v2"); a = ap.parse_args()
    fig1(); fig2(a.tag); fig3(a.tag); print("figures written to", OUT)
