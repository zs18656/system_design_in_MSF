"""Chapter 7 retrospective comparisons of accuracy and marginal realism."""

import numpy as np


def pareto_mask(mse, w1):
    """Keep ties and exclude cells strictly dominated on either objective."""
    points = np.column_stack((mse, w1))
    # Same strict-dominance rule as the realism code's get_pareto_front_df.
    dominates = ((points[:, None] <= points[None, :]).all(axis=-1)
                 & (points[:, None] < points[None, :]).any(axis=-1))
    return ~dominates.any(axis=0)


def opportunities(ids, mse, w1, epsilon):
    """Compare the MSE minimizer to the best distinct in-band alternative."""
    mse, w1 = np.asarray(mse, dtype=float), np.asarray(w1, dtype=float)
    if (mse.ndim != 2 or mse.shape != w1.shape or mse.shape[0] != len(ids)
            or len(set(ids)) != len(ids) or not np.isfinite(epsilon) or epsilon < 0
            or not np.isfinite(mse).all() or not np.isfinite(w1).all()
            or (mse < 0).any() or (w1 < 0).any()):
        raise ValueError("Expected nonnegative, finite [configuration,horizon] losses")
    rows = []
    for step, (mse_h, w1_h) in enumerate(zip(mse.T, w1.T), 1):
        baseline = min(range(len(ids)), key=lambda i: (mse_h[i], w1_h[i], ids[i]), default=None)
        alternative = None
        if baseline is not None:
            band = [i for i in range(len(ids))
                    if i != baseline and mse_h[i] <= (1 + epsilon) * mse_h[baseline]]
            alternative = min(band, key=lambda i: (w1_h[i], mse_h[i], ids[i]), default=None)
        gain = cost = None
        if alternative is not None:
            gain = (float((w1_h[baseline] - w1_h[alternative]) / w1_h[baseline])
                    if w1_h[baseline] else None)
            # A zero-MSE baseline only admits zero-MSE alternatives.
            cost = (float((mse_h[alternative] - mse_h[baseline]) / mse_h[baseline])
                    if mse_h[baseline] else 0.0)
        rows.append({
            "horizon": step, "baseline": ids[baseline] if baseline is not None else None,
            "alternative": ids[alternative] if alternative is not None else None,
            "eligible": alternative is not None, "realism_gain": gain, "mse_cost": cost,
            "success": gain is not None and gain > cost,
        })
    gains = [row["realism_gain"] for row in rows if row["realism_gain"] is not None]
    successful = sum(row["success"] for row in rows)
    return rows, {
        "epsilon": epsilon, "evaluated_horizons": len(rows),
        "eligible_horizons": sum(row["eligible"] for row in rows),
        "horizons_with_defined_gain": len(gains), "successful_horizons": successful,
        "median_eligible_realism_gain": float(np.median(gains)) if gains else None,
        "success_rate": successful / len(rows) if rows else None,
    }
