"""Metrics on aligned point forecasts in observation units."""

import numpy as np

METRICS = ("mse", "mae", "w1")


def horizon_metrics(targets, predictions):
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if (targets.ndim != 2 or not all(targets.shape)
            or predictions.ndim not in (2, 3) or predictions.shape[-2:] != targets.shape
            or not np.isfinite(targets).all() or not np.isfinite(predictions).all()):
        raise ValueError("Expected finite aligned [N,H] targets and [...,N,H] predictions")
    error = predictions - targets
    # Equal-weight empirical W1, adapted from the realism code's sorted samples.
    return {
        "mse": np.mean(error**2, axis=-2),
        "mae": np.mean(np.abs(error), axis=-2),
        "w1": np.mean(np.abs(np.sort(predictions, axis=-2) - np.sort(targets, axis=0)), axis=-2),
    }
