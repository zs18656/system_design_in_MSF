#!/usr/bin/env python3
"""Standalone horizon-wise thesis plots for thesis-forecasting-reference v0.2.

Reads a previously generated Part I bundle and Part II analysis. Does not train,
modify the bundle, or require modifications to the existing repository.

Run from the repository root:
    python plot_horizon_analyses.py \
      --bundle runs/single_task \
      --analysis runs/single_task-analysis

Dependencies: numpy, matplotlib (already included in the repository requirements).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows to save in {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, path: Path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def to_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def mean_losses(metadata: dict, metrics: list[dict], names=("mse", "mae", "w1")):
    """Map (split, config, horizon, metric) to fit-seed-averaged metric."""
    by_candidate = {c["candidate_id"]: c["configuration_id"] for c in metadata["candidates"]}
    groups = defaultdict(list)
    for row in metrics:
        candidate = row["candidate_id"]
        if candidate not in by_candidate:
            raise ValueError(f"Unknown candidate ID in metrics: {candidate}")
        for name in names:
            groups[(row["split"], by_candidate[candidate], int(row["horizon"]), name)].append(float(row[name]))
    return {key: float(np.mean(values)) for key, values in groups.items()}


def select_by_horizon(ids: list[str], h: int, losses: dict, metric: str):
    """Select configuration using validation only; deterministic ID tie-break."""
    return min(ids, key=lambda cid: (losses[("val", cid, h, metric)], cid))


def decomposition(surface: np.ndarray):
    """Equal-weight main-effect/interaction variance fractions on a complete grid."""
    grand = surface.mean()
    row_effect = surface.mean(axis=1, keepdims=True) - grand
    col_effect = surface.mean(axis=0, keepdims=True) - grand
    interaction = surface - grand - row_effect - col_effect
    total = np.mean((surface - grand)**2)
    if total <= 1e-24:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(row_effect**2) / total),
            float(np.mean(col_effect**2) / total),
            float(np.mean(interaction**2) / total))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Part I result bundle")
    parser.add_argument("--analysis", type=Path, required=True, help="Existing Part II analysis directory")
    parser.add_argument("--out", type=Path, default=None, help="Default: ANALYSIS/horizon_figures")
    parser.add_argument("--horizon", type=int, default=None, help="Horizon for heatmap and Pareto plot (default: H)")
    parser.add_argument("--metric", choices=("mse", "mae"), default="mse", help="Selection and Ch6 surface metric")
    args = parser.parse_args()

    out = args.out or (args.analysis / "horizon_figures")
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.bundle / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((args.analysis / "summary.json").read_text(encoding="utf-8"))
    metrics = csv_rows(args.bundle / "metrics.csv")
    pareto = csv_rows(args.analysis / "horizon_pareto.csv")
    opportunity = csv_rows(args.analysis / "opportunity.csv")

    H = int(metadata["config"]["horizon"])
    horizons = np.arange(1, H + 1)
    selected_h = args.horizon or H
    if not (1 <= selected_h <= H):
        parser.error(f"--horizon must be within 1..{H}")

    configs = {}
    for c in metadata["candidates"]:
        cid = c["configuration_id"]
        if cid in configs and (c["architecture"], c["strategy"]) != (configs[cid]["architecture"], configs[cid]["strategy"]):
            raise ValueError(f"Conflicting metadata for {cid}")
        configs[cid] = c
    losses = mean_losses(metadata, metrics)
    unrectified = sorted(cid for cid, c in configs.items() if c["strategy"]["rectifier"] is None)
    all_ids = sorted(configs)
    if not unrectified or len(all_ids) <= len(unrectified):
        raise ValueError("Need both unrectified and expanded candidates to compare Chapter 5 selections")
    for h in horizons:
        for cid in all_ids:
            for split in ("val", "test"):
                for metric in (args.metric, "mse", "w1"):
                    if (split, cid, int(h), metric) not in losses:
                        raise ValueError(f"Missing metric: {split}, {cid}, h={h}, {metric}")

    # Chapter 5: each horizon is independently validation-selected; test is only read afterwards.
    records = []
    for h in horizons:
        h = int(h)
        old = select_by_horizon(unrectified, h, losses, args.metric)
        new = select_by_horizon(all_ids, h, losses, args.metric)
        old_test = losses[("test", old, h, args.metric)]
        new_test = losses[("test", new, h, args.metric)]
        records.append({
            "horizon": h,
            "restricted_configuration": old,
            "expanded_configuration": new,
            "restricted_validation_loss": losses[("val", old, h, args.metric)],
            "expanded_validation_loss": losses[("val", new, h, args.metric)],
            "restricted_test_loss": old_test,
            "expanded_test_loss": new_test,
            "test_difference_expanded_minus_restricted": new_test - old_test,
            "test_ratio_expanded_to_restricted": new_test / old_test if old_test else "",
            "expanded_selected_rectifier": configs[new]["strategy"]["rectifier"] is not None,
        })
    write_csv(out / "ch5_horizon_selection.csv", records)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(horizons, [r["restricted_test_loss"] for r in records], label="Select within unrectified RecMO", lw=1.8)
    ax.plot(horizons, [r["expanded_test_loss"] for r in records], label="Select within expanded RecMO–RecMO", lw=1.8)
    ax.set(xlabel="Forecast horizon h", ylabel=f"Held-out test {args.metric.upper()}",
           title="Chapter 5: horizon-wise validation selection")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    save(fig, out / "ch5_validation_selection_by_horizon.png")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    difference = np.array([r["test_difference_expanded_minus_restricted"] for r in records])
    ax.plot(horizons, difference, lw=1.3)
    ax.axhline(0, color="black", lw=.8)
    ax.set(xlabel="Forecast horizon h", ylabel=f"Test {args.metric.upper()} difference (expanded − restricted)",
           title="Chapter 5: effect of expanding the validation candidate set")
    ax.grid(alpha=.2)
    save(fig, out / "ch5_expansion_test_difference.png")

    # Chapter 6: each h supplies a separate fully crossed architecture × rollout plane.
    architectures = sorted({configs[cid]["architecture"] for cid in unrectified})
    blocks = sorted({int(configs[cid]["strategy"]["base"]["block_size"]) for cid in unrectified})
    cell_ids = {}
    for cid in unrectified:
        c = configs[cid]
        key = (c["architecture"], int(c["strategy"]["base"]["block_size"]))
        if key in cell_ids:
            raise ValueError(f"Duplicate unrectified configuration at {key}")
        cell_ids[key] = cid
    if len(cell_ids) != len(architectures) * len(blocks):
        raise ValueError("Chapter 6 requires a complete architecture × RecMO grid")

    interactions, topk_rows = [], []
    chosen_surface = None
    n_cells = len(unrectified)
    for h in horizons:
        h = int(h)
        surface = np.array([[losses[("test", cell_ids[(a, b)], h, args.metric)] for b in blocks]
                            for a in architectures], dtype=float)
        a_frac, b_frac, i_frac = decomposition(surface)
        interactions.append({"horizon": h, "architecture_fraction": a_frac,
                             "strategy_fraction": b_frac, "interaction_fraction": i_frac})
        if h == selected_h:
            chosen_surface = surface
        ordered = sorted(((surface[i, j], architectures[i], blocks[j])
                          for i in range(len(architectures)) for j in range(len(blocks))))
        arch_counts = defaultdict(int)
        block_counts = defaultdict(int)
        for k, (_, a, b) in enumerate(ordered, 1):
            arch_counts[a] += 1
            block_counts[b] += 1
            topk_rows.append({"horizon": h, "k": k,
                              "architecture_share": max(arch_counts.values()) / k,
                              "strategy_share": max(block_counts.values()) / k})
    write_csv(out / "ch6_horizon_interaction.csv", interactions)
    write_csv(out / "ch6_horizon_topk.csv", topk_rows)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    im = ax.imshow(chosen_surface, aspect="auto")
    ax.set_xticks(range(len(blocks)), blocks)
    ax.set_yticks(range(len(architectures)), architectures)
    ax.set(xlabel="RecMO block size", ylabel="Architecture",
           title=f"Chapter 6: test-{args.metric.upper()} response surface at h={selected_h}")
    fig.colorbar(im, ax=ax, label=f"Test {args.metric.upper()}")
    save(fig, out / "ch6_surface_selected_horizon.png")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for key, label in [("architecture_fraction", "Architecture"),
                       ("strategy_fraction", "RecMO strategy"),
                       ("interaction_fraction", "Interaction")]:
        ax.plot(horizons, [r[key] for r in interactions], lw=1.5, label=label)
    ax.set(xlabel="Forecast horizon h", ylabel="Fraction of cell variance",
           ylim=(-.03, 1.03), title="Chapter 6: per-horizon response-surface decomposition")
    ax.grid(alpha=.2)
    ax.legend()
    save(fig, out / "ch6_interaction_by_horizon.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for key, label in [("architecture_share", "Dominant architecture share"),
                       ("strategy_share", "Dominant RecMO block-size share")]:
        ys = [np.mean([r[key] for r in topk_rows if r["k"] == k])
              for k in range(1, n_cells + 1)]
        ax.plot(np.arange(1, n_cells + 1), ys, label=label)
    ax.set(xlabel="Top-k configurations at a horizon", ylabel="Mean dominant share across horizons",
           ylim=(0, 1.05), title="Chapter 6: descriptive top-k concentration")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    save(fig, out / "ch6_topk_by_horizon.png")

    # Chapter 7: use existing test-based, retrospective analysis (all candidates).
    frontier = [r for r in pareto if int(r["horizon"]) == selected_h]
    if not frontier:
        raise ValueError(f"No Chapter 7 Pareto data for h={selected_h}")
    x = np.array([float(r["mse"]) for r in frontier])
    y = np.array([float(r["w1"]) for r in frontier])
    pareto_keep = np.array([to_bool(r["pareto"]) for r in frontier])
    is_rectifier = np.array([bool(r["rectifier_block_size"].strip()) for r in frontier])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(x[~is_rectifier], y[~is_rectifier], marker="o", alpha=.8, label="Unrectified RecMO")
    ax.scatter(x[is_rectifier], y[is_rectifier], marker="x", alpha=.65, label="RecMO–RecMO")
    front_idx = np.flatnonzero(pareto_keep)
    front_idx = front_idx[np.argsort(x[front_idx])]
    ax.plot(x[front_idx], y[front_idx], "-o", lw=1.5, markersize=3, label="Test Pareto front")
    best_idx = int(np.argmin(x))
    ax.scatter([x[best_idx]], [y[best_idx]], s=125, facecolors="none", edgecolors="black", lw=1.3,
               label="Minimum test MSE")
    ax.set(xlabel="Held-out test MSE", ylabel="Held-out test marginal W₁",
           title=f"Chapter 7: retrospective MSE–W₁ space at h={selected_h}")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    save(fig, out / "ch7_pareto_selected_horizon.png")

    opp = sorted(opportunity, key=lambda r: int(r["horizon"]))
    if len(opp) != H or [int(r["horizon"]) for r in opp] != list(horizons):
        raise ValueError("Expected exactly one opportunity record per horizon")
    eligible = np.array([to_bool(r["eligible"]) for r in opp])
    success = np.array([to_bool(r["success"]) for r in opp])
    gains = np.array([float(r["realism_gain"]) if r["realism_gain"].strip() else np.nan for r in opp])
    if np.any(success & ~eligible):
        raise ValueError("Ineligible horizons cannot be successful")
    defined = eligible & np.isfinite(gains)
    median = np.median(gains[defined]) if defined.any() else np.nan
    rate = success.sum() / H
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for condition, marker, label in ((eligible & success, "o", "Successful"),
                                     (eligible & ~success, "x", "Eligible but not successful")):
        ax.scatter(horizons[condition], 100 * gains[condition], marker=marker, s=24, label=label)
    if (~eligible).any():
        ax.scatter(horizons[~eligible], np.zeros((~eligible).sum()), marker="|", s=85,
                   label="Ineligible (indicator only)")
    ax.axhline(0, color="black", lw=.7, alpha=.5)
    if np.isfinite(median):
        ax.axhline(100 * median, ls="--", lw=1.1,
                   label=f"Median gain where defined: {100 * median:.1f}%")
    ax.set(xlabel="Forecast horizon h", ylabel="Relative reduction in marginal W₁ (%)",
           title=f"Chapter 7: in-band realism opportunities — success {100*rate:.1f}% ({success.sum()}/{H})")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    save(fig, out / "ch7_opportunities_by_horizon.png")
    print(f"\nCreated seven figures and three CSVs in {out}")
    print(f"Horizon-wise selection: expanded space won test {args.metric.upper()} on "
          f"{sum(r['expanded_test_loss'] < r['restricted_test_loss'] for r in records)}/{H} horizons")
    print("Caution: horizons share the same series; do not treat these as independent datasets for inference.")


if __name__ == "__main__":
    main()
