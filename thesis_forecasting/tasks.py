"""Synthetic tasks and chronological, split-aligned forecasting windows."""

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Windows:
    origins: np.ndarray
    history: np.ndarray
    targets: np.ndarray


def mackey_glass(n_steps, seed, a, gamma, tau, sigma_s, sigma_e, burn_in, initial_state):
    """Discrete unit-time delayed feedback with additive Gaussian innovations."""
    if (min(n_steps, tau) < 1 or burn_in < 0 or min(sigma_s, sigma_e) < 0
            or not np.isfinite([a, gamma, sigma_s, sigma_e, initial_state]).all()):
        raise ValueError("Invalid Mackey–Glass parameters")
    # Adapted from realism helpers.simulate_mackey_dgp. The initial history is
    # constant and precedes the first simulated observation.
    rng = np.random.default_rng(seed)
    state_noise = rng.normal(0, sigma_s, n_steps + burn_in)
    observation_noise = rng.normal(0, sigma_e, n_steps + burn_in)
    state = np.full(tau + len(state_noise), initial_state, dtype=float)
    for i, innovation in enumerate(state_noise, start=tau):
        lagged = state[i - tau]
        previous = state[i - 1]
        state[i] = previous - a * previous + gamma * lagged / (1 + lagged**10) + innovation
        if not np.isfinite(state[i]) or abs(state[i]) > 1e6:
            raise ValueError("Unstable Mackey–Glass simulation")
    return (state[tau:] + observation_noise)[burn_in:]


def split_windows(series, context, horizon, fractions):
    """Use observed history across boundaries; contain every target within its split."""
    series = np.asarray(series, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if (series.ndim != 1 or not np.isfinite(series).all() or context < 1 or horizon < 1
            or fractions.shape != (3,) or not np.all(fractions > 0)
            or not np.isclose(fractions.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("Expected a finite series, positive L/H and three split fractions")
    boundaries = [0, int(len(series) * fractions[0]),
                  int(len(series) * (fractions[0] + fractions[1])), len(series)]
    if boundaries[1] < context + horizon or min(np.diff(boundaries)[1:]) < horizon:
        raise ValueError("Each split must provide at least one full forecast window")
    train = series[:boundaries[1]]
    mean, std = float(train.mean()), float(train.std())
    scale = std if std > 0 else 1.0
    history = sliding_window_view((series - mean) / scale, context)
    targets = sliding_window_view(series, horizon)
    windows = {}
    for split, start, stop in zip(SPLITS, boundaries[:-1], boundaries[1:]):
        first_target = np.arange(max(start, context), stop - horizon + 1)
        windows[split] = Windows(first_target - 1, history[first_target - context],
                                 targets[first_target])
    return windows, {"mean": mean, "std": std, "scale": scale, "boundaries": boundaries}
