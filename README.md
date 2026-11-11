# Forecasting System Design

**Code and reference implementations accompanying the PhD thesis *System Design and Selection for Multi-Step Time Series Forecasting*.**

This repository accompanies research conducted at the University of Bristol on the construction, evaluation, and selection of multi-step time-series forecasting systems.

The central premise is that forecasting performance depends not only on the predictive model, but also on how that model is trained, deployed across the forecast horizon, and evaluated. The aim of this repository is to provide a common framework for studying these design choices and their interactions.

## Research overview

The framework is organised around three research directions:

**1. Forecasting strategy construction (Stratify)**

Represent established multi-step forecasting strategies within a unified design space, including recursive, direct, multi-output, and base–rectifier constructions. Compare established strategies against newly constructed candidates using validation-based selection and held-out evaluation.

**2. Architecture–strategy interaction**

Evaluate predictive architectures across different forecasting strategies to investigate ranking stability and the structure of the resulting performance surfaces. Analyses include configuration rankings, interaction effects, top-performing regions, and budgeted search.

**3. Accuracy and distributional evaluation**

Evaluate forecasting systems using pointwise accuracy metrics such as MSE and MAE, alongside marginal distributional discrepancy measured using Wasserstein distance. Study Pareto fronts and the availability of alternative predictors within specified accuracy tolerances.

## Planned framework components

The repository is intended to bring together the following capabilities:

- **Data preparation:** Time-series loading, chronological train/validation/test partitioning, training-only normalisation, and construction of forecasting windows.
- **Candidate construction:** Model architectures, multi-step forecasting strategies, and compatible deployment procedures.
- **Training and prediction:** Fitting candidate systems and generating horizon-level forecasts.
- **Evaluation and selection:** MSE, MAE, marginal Wasserstein distance, validation-based configuration selection, and held-out comparison.
- **Design-space analysis:** Strategy rankings, accuracy–realism Pareto fronts, architecture–strategy interaction, and top-performing configuration regions.

The framework is intended to support controlled comparisons in which model architecture, forecasting strategy, and evaluation objective can be varied independently where appropriate.

## Repository status

**Initial release — code being prepared.**

This repository has been established as a public location for the reference implementations accompanying the thesis. The implementation will be uploaded progressively, beginning with the core data preparation, candidate-construction, forecasting, and evaluation pipeline.

Additional experiment-specific analysis and reproduction instructions are planned for subsequent updates.

The initial framework is intended to provide reusable implementations of the research methods. It should not be interpreted as a complete reproduction package for every experiment or figure in the thesis unless explicitly documented as such.


## Reproducibility and scope

As implementations are added, the repository documentation will identify:

- Which thesis methods and experiments are supported.
- Which components reproduce original experiments and which are independent reference implementations.
- Required datasets, dependencies, configurations, and execution instructions.
- Known limitations and any differences from the original experimental setups.
