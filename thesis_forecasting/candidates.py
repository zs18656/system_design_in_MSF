"""Fit a block predictor and deploy it over the full horizon."""

from collections import defaultdict
from itertools import product
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits


def registry(config):
    architectures = config["models"]
    sizes = config["block_sizes"]
    rectifier_sizes = config.get("rectifier_block_sizes", [])
    seeds = config["fit_seeds"]
    horizon = config["horizon"]
    if (not architectures or not sizes or not seeds or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds) or len(set(sizes)) != len(sizes)
            or any(type(s) is not int or not 1 <= s <= horizon for s in sizes)):
        raise ValueError("Provide models, unique nonnegative seeds and unique block sizes in [1, H]")
    if set(architectures) - {"ridge", "tree", "mlp"}:
        raise ValueError("Supported models are ridge, tree and mlp")
    if (not isinstance(rectifier_sizes, list)
            or any(type(s) is not int or not 1 <= s <= horizon for s in rectifier_sizes)
            or len(set(rectifier_sizes)) != len(rectifier_sizes)):
        raise ValueError("rectifier_block_sizes must be a list of unique integers in [1, H]")
    rows = []
    for architecture, size in product(architectures, sizes):
        parameters = dict(architectures[architecture])
        if "random_state" in parameters:
            raise ValueError("Set model randomness through fit_seeds")
        training = {"loss": "squared_error"}
        if architecture == "mlp":
            if {"early_stopping", "max_iter", "solver"} & parameters.keys():
                raise ValueError("MLP uses fixed epochs and the Adam solver")
            epochs = parameters.pop("epochs")
            if type(epochs) is not int or epochs < 1:
                raise ValueError("MLP epochs must be a positive integer")
            training.update(epochs=epochs, solver="adam")
        if architecture in {"tree", "mlp"} and "loss" in parameters:
            raise ValueError("Set no loss override for the squared-error models")
        strategy = {"base": {"family": "recmo", "block_size": size}, "rectifier": None}
        config_id = f"{architecture}-recmo-{size}"
        for seed in seeds[:1] if architecture == "ridge" else seeds:
            fit_id = f"{config_id}-fit-{seed}"
            base = {
                "candidate_id": f"{fit_id}-point", "configuration_id": config_id,
                "fit_id": fit_id, "fit_seed": seed, "inference_seed": None,
                "architecture": architecture, "strategy": strategy,
                "inference": "point", "training": training,
                "hyperparameters": parameters,
            }
            rows.append(base)
            for rectifier_size in rectifier_sizes:
                pair_id = f"{config_id}-rect-recmo-{rectifier_size}"
                pair_fit_id = f"{pair_id}-fit-{seed}"
                rows.append({
                    **base,
                    "candidate_id": f"{pair_fit_id}-point",
                    "configuration_id": pair_id,
                    "fit_id": pair_fit_id,
                    "base_fit_id": fit_id,
                    "strategy": {
                        "base": strategy["base"],
                        "rectifier": {
                            "family": "recmo", "block_size": rectifier_size,
                            "architecture": architecture, "fit_seed": seed,
                            "training": training, "hyperparameters": parameters,
                            "residual": "base_forecast_minus_target",
                            "residual_split": "train",
                        },
                    },
                })
    return rows


def fit_block(component, inputs, targets, size):
    block = targets[:, :size]
    y = block[:, 0] if size == 1 else block
    params = component["hyperparameters"]
    seed = component["fit_seed"]
    match component["architecture"]:
        case "ridge":
            return Ridge(**params).fit(inputs, y)
        case "tree":
            return DecisionTreeRegressor(random_state=seed, **params).fit(inputs, y)
        case "mlp":
            epochs = component["training"]["epochs"]
            model = MLPRegressor(random_state=seed, max_iter=epochs, tol=0,
                                 n_iter_no_change=epochs, early_stopping=False, **params)
            with warnings.catch_warnings():
                # A fixed training budget intentionally reaches the epoch cap.
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                return model.fit(inputs, y)
        case _:
            raise ValueError(f"Unknown architecture {component['architecture']}")


def rollout(predict_block, history, horizon, size, *, base_forecast=None):
    """Roll RecMO forward, optionally subtracting residuals from a fixed base trajectory."""
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or not all(history.shape) or not 1 <= size <= horizon:
        raise ValueError("Expected a nonempty [N,L] history and 1 <= block size <= horizon")
    n_origins, context = history.shape
    if base_forecast is not None:
        base_forecast = np.asarray(base_forecast, dtype=float)
        if base_forecast.shape != (n_origins, horizon) or not np.isfinite(base_forecast).all():
            raise ValueError("Expected a finite base forecast aligned with history and horizon")
    # Preallocated state from Stratify RECMO; final-block truncation from the
    # architecture–RecMO rollout. This also supports blocks wider than context.
    state = np.empty((n_origins, context + horizon), dtype=float)
    state[:, :context] = history
    for start in range(0, horizon, size):
        block = np.asarray(predict_block(state[:, start:start + context]), dtype=float)
        if size == 1 and block.shape == (n_origins,):
            block = block[:, None]
        if block.shape != (n_origins, size) or not np.isfinite(block).all():
            raise ValueError(f"Expected a finite block of shape {(n_origins, size)}")
        take = min(size, horizon - start)
        next_values = block[:, :take]
        if base_forecast is not None:
            # Stratify's closed-loop RecMO rectifier feeds back corrected values.
            next_values = base_forecast[:, start:start + take] - next_values
        state[:, context + start:context + start + take] = next_values
    return state[:, context:]


def fit_predict(config, windows, scaling, candidates):
    """Fit one base per group and reuse its full forecasts for each rectifier."""
    train = windows["train"]
    scale, mean = scaling["scale"], scaling["mean"]
    targets = (train.targets - mean) / scale
    horizon = config["horizon"]
    predictions = {split: np.empty((len(candidates), len(data.origins), horizon))
                   for split, data in windows.items()}
    bases = {}
    pairs = defaultdict(list)
    for index, candidate in enumerate(candidates):
        if candidate["strategy"]["rectifier"] is None:
            bases[candidate["fit_id"]] = (index, candidate)
        else:
            pairs[candidate["base_fit_id"]].append((index, candidate["strategy"]["rectifier"]))
    if pairs.keys() - bases.keys():
        raise ValueError("A rectified candidate references a missing base fit")
    with threadpool_limits(limits=1):
        for base_fit_id, (index, candidate) in bases.items():
            size = candidate["strategy"]["base"]["block_size"]
            model = fit_block(candidate, train.history, targets, size)
            base = {split: rollout(model.predict, data.history, horizon, size)
                    for split, data in windows.items()}
            for split, point in base.items():
                predictions[split][index] = point * scale + mean
            if base_fit_id in pairs:
                residuals = base["train"] - targets
                for pair_index, rectifier in pairs[base_fit_id]:
                    size = rectifier["block_size"]
                    model = fit_block(rectifier, train.history, residuals, size)
                    for split, data in windows.items():
                        corrected = rollout(model.predict, data.history, horizon, size,
                                            base_forecast=base[split])
                        predictions[split][pair_index] = corrected * scale + mean
    return predictions
