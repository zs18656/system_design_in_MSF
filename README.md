# Forecasting system design

A CPU-only reference implementation for Chapters 5–7 of Riku Green's PhD thesis. It demonstrates how to construct, fit and deploy multi-step predictors, then analyse their saved forecasts. The examples use newly generated synthetic series and small fixed grids. Their scores are specific to these examples. The strategy space is restricted to RecMO bases crossed with RecMO rectifiers and the option of no rectifier. The full Stratify space also crosses DirMO and DirRecMO at both stages.

## Quick start

Use Python 3.11 or newer. The pinned environment was verified on Python 3.12.14.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

python -m thesis_forecasting.train --config configs/tiny.json --out runs/tiny
python -m thesis_forecasting.analyse --bundle runs/tiny --out runs/tiny-analysis

python -m thesis_forecasting.train --config configs/single_task.json --out runs/single_task
python -m thesis_forecasting.analyse --bundle runs/single_task --out runs/single_task-analysis
```

The output paths must be new. Part II can also analyse the bundled tiny example directly with `--bundle examples/tiny/bundle`. It imports no model fitting code. On Windows, activate the virtual environment using the command for your shell.

| Preset | Observations | Context / horizon | Architectures | Base and rectifier block sizes | Fit seeds |
| --- | ---: | --- | --- | --- | --- |
| `tiny` | 800 | 16 / 8 | Ridge, MLP | 1, 8 | 0 |
| `single_task` | 6,000 | 32 / 16 | Ridge, tree, MLP | 1, 2, 4, 16 | 0 |
| `three_noise` | 6,000 per regime | 32 / 200 | Ridge, tree, MLP | 1, 8, 32, 200 | 0, 1, 2 (ridge uses 0) |

To run the three regimes with the same design and independently saved bundles, allow roughly 2.5 GB of disk space for the three forecast caches:

```bash
python -m thesis_forecasting.noise --config configs/three_noise.json --out runs/noise-configs
for regime in low medium high; do
  python -m thesis_forecasting.train --config "runs/noise-configs/${regime}.json" --out "runs/${regime}" || exit
  python -m thesis_forecasting.analyse --bundle "runs/${regime}" --out "runs/${regime}-analysis" || exit
done
```

## Verified examples

All 13 automated tests passed. They cover training-only residual targets, base reuse, corrected-value feedback, final-block truncation, validation-only selection and repeatable tiny forecasts. The tiny bundle was analysed in a separate process with model fitting imports disabled.

| Run | Fitted candidates | Chapter 5 configurations (base / expanded) | Held-out MSE (base / expanded) | Successful Chapter 7 horizons |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 12 | 4 / 12 | 0.028770 / 0.028770 | 7/8 |
| Single task | 60 | 12 / 60 | 0.011337 / 0.011560 | 10/16 |
| Low noise | 140 | 12 / 60 | 0.018200 / 0.015818 | 47/200 |
| Medium noise | 140 | 12 / 60 | 0.060574 / 0.060574 | 147/200 |
| High noise | 140 | 12 / 60 | 0.109702 / 0.109813 | 165/200 |

The ZIP includes the tiny forecast bundle and the analysis CSVs, JSON summaries and plots for all five runs. The commands above regenerate the other forecast bundles.

## Part I · fit and deploy

- The data are synthetic Mackey–Glass observations. The generator uses the thesis's unit-time discrete recurrence with decay 0.1, feedback 0.2, delay 17, burn-in 100 and no observation noise in these presets. The initial history is constant at 1.0. Process noise is Gaussian with standard deviation per time step of 0.01, 0.05 or 0.10 for low, medium or high, taken from the thesis's controlled grid. The regimes share a generation seed, so they scale the same innovation draws.
- Training, validation and test are chronological (70/10/20). Scaling uses training observations only. Each window's target lies entirely within its split. A window can use earlier observations, including prior splits or already observed points within the current split. Origins are zero-based indices of the last observed point.
- One model per architecture, RecMO block size and fit seed learns the first `s` targets on the common training windows. Inference feeds predicted blocks into the rolling history and truncates the final block to `H`. The endpoints are recursive (`s=1`) and MIMO (`s=H`). The MLP uses a fixed Adam epoch budget with no early stopping.
- `block_sizes` sets the base grid and `rectifier_block_sizes` sets the independent correction grid. Each architecture and fit seed gets every configured base–rectifier pair and every unrectified base. Omit `rectifier_block_sizes` or use an empty list for an unrectified-only run.
- Each base fit and its split forecasts are reused across its rectifiers. The rectifier shares the base architecture and hyperparameters. On training windows, roll the base out to the full horizon and fit the rectifier on the first `s_rect` columns of the aligned in-sample residuals, defined as base forecast minus target in standardised units. At inference, subtract each predicted residual block from the corresponding fixed base forecast block. Feed the corrected observations back into the rectifier's rolling context and truncate the final block to the horizon. Validation and test targets are reserved for scoring.
- Each bundle contains `metadata.json` (configuration, scaling, package versions, candidate IDs and lineage), `forecasts.npz` (common origins `[N]`, targets `[N,H]`, and candidate-ordered predictions `[C,N,H]` for each split), `metrics.csv` (candidate/split/horizon metrics), and `manifest.json` (schema version and file checksums). Predictions and metrics use observation units. Fitted models are optional under the specification and omitted here.

## Part II · analyse saved forecasts

| Output | Method |
| --- | --- |
| `summary.json`, `configuration_ranks.csv` | Chapter 5 validation selection and descriptive ranks. Compare selection from unrectified RecMO with selection from the expanded pool containing the same bases and every configured pair. Both select by mean validation loss across origins, horizons and fit seeds. Report held-out scores for all seeds of each winner, plus the expanded-to-unrectified test loss difference and ratio. Additional pools can be declared with `selection_subsets`, using configuration IDs such as `ridge-recmo-1` or `ridge-recmo-1-rect-recmo-8`. |
| `test_surface.png`, `top_concentration.csv` | Chapter 6 decomposition of the unrectified, seed-averaged test response surface with equal weight per architecture–strategy cell. Top-k reports the largest architecture and strategy shares in each ranked prefix. A constant surface has undefined variance fractions. |
| `budget_search.csv`, `budget_search.png` | Random, architecture-first and strategy-first search over unrectified cells. Each queried cell reveals one fit seed's validation loss. Each returned cell is scored using all of its test seeds. Test regret uses the retrospective minimum over the full test grid as its reference. |
| `horizon_pareto.csv`, `pareto_final_horizon.png` | Chapter 7 horizon-wise MSE, MAE and equal-weight empirical marginal Wasserstein-1 across origins. Both unrectified and rectified configurations participate. Fit-seed metrics are averaged before comparing configurations. |
| `opportunity.csv`, `summary.json` | Retrospective test comparison over the expanded pool. At each horizon, select the distinct lowest-W1 alternative within 5% of minimum MSE. An eligible alternative can have zero or negative gain. The success rate uses all evaluated horizons. The conditional median uses eligible horizons with defined relative gain. Zero baseline W1 yields undefined relative gain. Zero baseline MSE admits only zero-MSE alternatives, for which the MSE cost is recorded as zero. |

Repeated horizons share forecast origins and outcomes, so these summaries are descriptive within each task. Deferred work includes DirMO and DirRecMO bases and rectifiers, RNN/LSTM and original TCN/N-BEATS adapters, probabilistic flow deployment, TPE, full thesis sweeps and cross-dataset inference.

## Source provenance

- [Stratify](https://github.com/zs18656/stratify_unifying_MSF) at `6e4f514`. Adapted the supervised prefix and rolling state in `strategies.py::RECMO`, plus the residual sign, first-block fitting and corrected-value feedback in `Stratify._fit_recmo` and `Stratify._rectified_recmo`.
- [Architecture–RecMO](https://github.com/ICDMempiricalstructure/paper_code) at `89302c6`. Adapted final-block truncation from `exploration_helpers.py::rollout_mo_net` and the equal-cell decomposition and search logic in `clean_hpo_analysis.py`. The response surface here uses fixed hyperparameters.
- Supplied KDD realism code. Adapted `helpers.py::simulate_mackey_dgp` and the marginal W1 and Pareto calculations in `A_B_rebuttals.py`. Reviewed `helpers_probabilistic.py::FlowMIMO` for future deployment work. Opportunity follows the thesis and counts ineligible horizons in the success denominator.
