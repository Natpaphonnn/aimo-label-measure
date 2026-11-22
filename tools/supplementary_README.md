# Code for "What Can the AIMO Robustness Labels Measure?" (AIMO Interpretability Challenge, NeurIPS 2026 Workshop)

Open-source licence: MIT (`LICENSE`). This tree is the public code repository linked from the paper; the same tree
was reviewed through the anonymous mirror https://anonymous.4open.science/r/aimo-label-measure, so user names, home paths and
Kaggle kernel slugs are placeholders (`<kaggle-user>`, `kx-*`).

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
| Sec 6 recommendations, App D | `python analysis/label_budget.py` | nothing |
| Figures 1-2 | `python analysis/make_figures.py --tag v3` | shipped `notes/results/*_v3.json` |
| stage-2 labels | `python tools/fetch_hf_public.py && python tools/build_case_labels.py` | network (public Hugging Face datasets, no token); shipped `data/seed/` |

## Revision of the report (conditional accept of #21): tables from the shipped cache

`notes/results/cache/` holds what the stage-2 tables need, so they rebuild without residual states or a GPU:
`samples.csv.gz` (every sample: model, problem id, sample index, extracted answer, gold correctness, tokens generated, cut,
finished before the cut), `../stage1_nested_scores.csv.gz` (the nested stage-1 score of every sample in every outer fold and
seed, role outer_test / outer_train), `cases_158.csv` / `cases_139.csv` (the case sets with their sample IDs and labels) and
`regeneration.json` (for every input file of `analysis/run_wb.py`: the Kaggle kernel or laptop run that produced it, with
model, token cap, backend, dtype, seed and samples per problem, and the state-extraction settings). Problem IDs are sha256
prefixes of the problem text.

| report | command | needs |
|---|---|---|
| Table 1, Table 3 (App B), per-class recall (App C), Fig 2a | `python analysis/wb_nested.py --from-cache` | shipped cache + labels (`tools/fetch_hf_public.py && tools/build_case_labels.py`) |
| Sec 2 sensitivity of the simulation, App E | `python analysis/label_sensitivity.py` | `tools/fetch_hf_public.py` (sample-full) |
| Sec 5 band rules, official labels, solve-rate-matched comparison, Fig 2b | `python analysis/label_decomp_revision.py` | shipped cache + `data/seed/` + labels |
| Sec 5 stage 1 inside finished / truncated samples | `python analysis/stage1_finished_truncated.py` | states (not shipped) |
| Sec 4 precision parity (bf16 vs fp16) | `python analysis/precision_parity.py` | states (not shipped) |
| nested stage 1 itself (writes the cache) | `python analysis/wb_nested.py [--probe-subset truncated] --jobs 3` | states (not shipped) |

The analyses were pre-registered before they were run: `decision_log/PREREG_REVISION_2026-11-22.md`.

## What needs the logged generations (not shipped)

These scripts read `runs/` (sampled solutions and residual states, several GB, produced on free Kaggle T4s and a laptop with
`src/selflabel/` and the kernels of `tools/kaggle_push_gen.py`). Without them they stop with a `[missing input] <file>` message
that says what the file is and where it comes from (`analysis/datapaths.py`). Their outputs are shipped in `notes/results/`.

| report | script | output in notes/results/ |
|---|---|---|
| Sec 5 stage 1 (AUCs, layers); submitted version's stage 2 (before nesting) | `analysis/run_wb.py --tag v3` | wb_two_stage_v3.{md,json} |
| submitted version's balanced accuracy (before nesting) | `analysis/wb_balanced.py --tag v3` | wb_balanced_v3.{md,json} |
| Sec 5 layers, controls, transfer, Fig 1 left | `analysis/wb_diagnostics.py --tag v3` | wb_diagnostics_v3.{md,json} |
| (submitted version, before nesting) white-box ablation | `analysis/wb_vs_offsets.py` | wb_vs_offsets_v3.md |
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
