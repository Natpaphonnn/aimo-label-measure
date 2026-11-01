# Code for "What Can the AIMO Robustness Labels Measure?" (anonymized supplementary material)

Open-source licence: MIT (`LICENSE`). Anonymized mirror: https://anonymous.4open.science/r/aimo-label-measure ; the public
GitHub repository will be released after review.

## Install (Python 3.12)

```bash
cd aimo-interp-code
python3.12 -m venv .venv && source .venv/bin/activate      # or: uv venv --python 3.12 .venv
pip install -r requirements.txt                             # torch/transformers are only needed for tests/test_hooks.py and sampling
python -m unittest discover -s tests -v                     # tests that need the organizers' getting-started repo or a model skip
```

## What runs from this folder alone

| report | command | needs |
|---|---|---|
| Sec 2 label simulation, Fig 1 right | `python analysis/label_noise_ceiling.py` | nothing |
| Sec 6 recommendations, App C | `python analysis/label_budget.py` | nothing |
| Figures 1-2 | `python analysis/make_figures.py --tag v3` | shipped `notes/results/*_v3.json` |
| stage-2 labels | `python tools/fetch_hf_public.py && python tools/build_case_labels.py` | network (public Hugging Face datasets, no token); shipped `data/seed/` |

## What needs the logged generations (not shipped)

These scripts read `runs/` (sampled solutions and residual states, several GB, produced on free Kaggle T4s and a laptop with
`src/selflabel/` and the kernels of `tools/kaggle_push_gen.py`). Without them they stop with a `[missing input] <file>` message
that says what the file is and where it comes from (`analysis/datapaths.py`). Their outputs are shipped in `notes/results/`.

| report | script | output in notes/results/ |
|---|---|---|
| Sec 5 stage 1 + stage 2, Table 1, Holm over the primary family | `analysis/run_wb.py --tag v3` | wb_two_stage_v3.{md,json} |
| Sec 5 balanced accuracy, Fig 2a | `analysis/wb_balanced.py --tag v3` | wb_balanced_v3.{md,json} |
| Sec 5 layers, controls, transfer, label decomposition, Fig 1 left, Fig 2b | `analysis/wb_diagnostics.py --tag v3` | wb_diagnostics_v3.{md,json} |
| Sec 5 white-box ablation, App B | `analysis/wb_vs_offsets.py` | wb_vs_offsets_v3.md |
| Sec 5 validation board, runtime, App A | `analysis/parse_val_logs.py`, `analysis/runtime_table.py` (server logs) | val_board_v1.md, runtime_val.md |

## Exploratory scripts

`analysis/p0_*`, `analysis/p1_*`, `analysis/m1_*`, `ablation_arms.py`, `eval_features.py`, `train_solve_probe.py`,
`fit_hybrid_artifact.py` and the `tools/*.sh` laptop schedulers are earlier explorations (Sec 8, negative results N1-N2) or
infrastructure; they are not needed for any table or figure. Each has `--selftest` or stops with a `[missing input]` / usage message.
The validation-server logs can be pointed to with `AIMO_VAL_LOGS=<dir>`.

## Other contents

- `bundles/`: the exact ZIPs submitted to Codabench: wb-K4 Main 1011261 / Small 1009859 (the two nominated
  for final evaluation, two in total per organizers), always-robust Main 991595 / Small 991597 (baseline, not nominated).
  `tools/build_twostage_bundle.py` builds them; `submissions/hybrid-v1/` is their source.
- `decision_log/DECISIONS.md`: the dated decision log the report cites (arms, comparisons and the nomination rule, #46-#53;
  post-hoc check #54; final nomination #55); `decision_log/commit_history.txt`: every commit that touched it (hash, time in UTC, subject). The hashes
  can be checked against the public repository after review.
- Anonymization: user names, home paths and Kaggle kernel slugs are replaced by placeholders (`<kaggle-user>`, `kx-*`).

## Data and licences

- `data/seed/` (shipped): labels and base accuracies of byeolki/aimo-interp, MIT licence (`data/seed/LICENSE-byeolki-MIT.txt`).
- `aimo-interp/train-main-v2` and `aimo-interp/aimo-interp-challenge-sample-full` (Hugging Face, public): the card's licence is
  "other" / not stated, so rows are not redistributed; `tools/fetch_hf_public.py` downloads them at pinned revisions.
- Problem texts (AIMO validation sets, MathArena, train-main-v2) are not redistributed; `tools/clone_external.sh` and
  `tools/build_problem_pool.py` rebuild the pool. AIME 2024-2026 problems are excluded everywhere.
- Code: MIT (`LICENSE`).
