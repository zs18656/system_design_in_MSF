"""Chapter 6 summaries of a fully crossed architecture–RecMO grid."""

from collections import Counter

import numpy as np


def variance_decomposition(surface):
    """Equal cell weights on a finite, complete grid."""
    surface = np.asarray(surface, dtype=float)
    if surface.ndim != 2 or not all(surface.shape) or not np.isfinite(surface).all():
        raise ValueError("Expected a complete finite architecture–RecMO grid")
    # Adapted from clean_hpo_analysis.surface_variance_decomposition.
    grand = surface.mean()
    rows = surface.mean(axis=1, keepdims=True) - grand
    columns = surface.mean(axis=0, keepdims=True) - grand
    interaction = surface - grand - rows - columns
    total = float(np.mean((surface - grand)**2))
    components = {
        "architecture": float(np.mean(rows**2)),
        "strategy": float(np.mean(columns**2)),
        "interaction": float(np.mean(interaction**2)),
    }
    return {
        "weighting": "equal configuration weight after averaging fit-seed losses",
        "total_variance": total,
        "component_variance": components,
        "fraction": {name: value / total if total else None
                     for name, value in components.items()},
    }


def top_concentration(ranked):
    """Return dominant architecture and rollout shares for each top-k prefix."""
    architecture_counts, strategy_counts = Counter(), Counter()
    output = []
    for k, cell in enumerate(ranked, 1):
        architecture_counts[cell["architecture"]] += 1
        strategy_counts[cell["block_size"]] += 1
        architecture = max(architecture_counts.values()) / k
        strategy = max(strategy_counts.values()) / k
        output.append({"k": k, "architecture_share": architecture,
                       "strategy_share": strategy, "share_difference": architecture - strategy})
    return output


def search_trace(validation, sizes, method, rng):
    """Query one fit-seed loss per cell; yield a nested sequence of observations."""
    n_arch, n_sizes = len(validation), len(sizes)

    def observe(i, j):
        return i, j, float(rng.choice(validation[i][j]))

    if method == "random":
        for index in rng.permutation(n_arch * n_sizes):
            yield observe(*divmod(int(index), n_sizes))
        return

    # Adapted from the architecture–RecMO axis-first traces. Cycle over
    # screened rows/columns by validation score, refining within each.
    if method == "architecture_first":
        anchor = int(rng.integers(n_sizes))
        order = rng.permutation(n_arch).tolist()
        screening = [observe(i, anchor) for i in order]
        yield from screening
        ranked = [i for i, _, _ in sorted(screening, key=lambda cell: cell[2])]
        seen = {i: [anchor] for i in ranked}
        for _ in range(n_sizes - 1):
            for i in ranked:
                j = max((j for j in range(n_sizes) if j not in seen[i]),
                        key=lambda j: min(abs(sizes[j] - sizes[k]) for k in seen[i]))
                yield observe(i, j)
                seen[i].append(j)
    elif method == "strategy_first":
        anchor = int(rng.integers(n_arch))
        order = rng.permutation(n_sizes).tolist()
        screening = [observe(anchor, j) for j in order]
        yield from screening
        ranked = [j for _, j, _ in sorted(screening, key=lambda cell: cell[2])]
        order_by_column = {j: rng.permutation([i for i in range(n_arch) if i != anchor])
                           for j in ranked}
        for step in range(n_arch - 1):
            for j in ranked:
                yield observe(int(order_by_column[j][step]), j)
    else:
        raise ValueError(f"Unknown search method: {method}")


def evaluate_search(validation, test, architectures, sizes, seed, repeats):
    """Score returned configurations against a retrospective test oracle."""
    oracle = float(test.min())
    rows = []
    for method in ("random", "architecture_first", "strategy_first"):
        for repeat in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence([seed, repeat]))
            winner = (0, 0, float("inf"))
            for budget, (i, j, loss) in enumerate(search_trace(validation, sizes, method, rng), 1):
                winner = min(winner, (i, j, loss), key=lambda cell: (cell[2], cell[0], cell[1]))
                win_i, win_j, win_val = winner
                test_loss = float(test[win_i, win_j])
                gap = test_loss - oracle
                rows.append({
                    "method": method, "repeat": repeat, "budget": budget,
                    "queried_architecture": architectures[i], "queried_block_size": sizes[j],
                    "queried_val_loss": loss,
                    "architecture": architectures[win_i], "block_size": sizes[win_j],
                    "selected_val_loss": win_val, "selected_test_loss": test_loss,
                    "test_oracle_loss": oracle, "test_regret": gap,
                    "relative_test_regret": gap / oracle if oracle else None,
                })
    return rows
