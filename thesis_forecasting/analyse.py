"""Part II command line entry point. Uses saved forecasts and performs no fitting."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .bundle import load_bundle, write_csv, write_json
from .metrics import METRICS, horizon_metrics
from .realism import opportunities, pareto_mask
from .surface import evaluate_search, top_concentration, variance_decomposition


def score_configurations(candidates, losses, metric):
    groups = defaultdict(list)
    for index, candidate in enumerate(candidates):
        groups[candidate["configuration_id"]].append(index)
    ids = sorted(groups)
    seed_val = losses["val"][metric].mean(axis=1)
    averaged = {
        split: {name: np.stack([values[name][groups[key]].mean(axis=0) for key in ids])
                for name in METRICS}
        for split, values in losses.items()
    }
    rows = []
    for pos, key in enumerate(ids):
        indices = groups[key]
        entry = candidates[indices[0]]
        rectifier = entry["strategy"]["rectifier"]
        rows.append({
            "configuration_id": key, "architecture": entry["architecture"],
            "block_size": entry["strategy"]["base"]["block_size"], "fit_count": len(indices),
            "rectifier_block_size": rectifier["block_size"] if rectifier else None,
            "val_loss": float(averaged["val"][metric][pos].mean()),
            "test_loss": float(averaged["test"][metric][pos].mean()),
            "val_seed_losses": seed_val[indices],
        })
    return rows, averaged


def select_validation(rows, allowed):
    """Select a configuration from a declared subset using validation loss only."""
    pool = [row for row in rows if row["configuration_id"] in allowed]
    return min(pool, key=lambda row: (row["val_loss"], row["configuration_id"]), default=None)


def selection_pools(rows, custom):
    ids = {row["configuration_id"] for row in rows}
    unrectified = {row["configuration_id"] for row in rows if row["rectifier_block_size"] is None}
    pools = {"unrectified_recmo": unrectified}
    if ids != unrectified:
        pools["expanded_recmo_recmo"] = ids
    if pools.keys() & custom.keys() or any(not members or set(members) - ids
                                          for members in custom.values()):
        raise ValueError("Custom selection subsets need distinct names and known configuration IDs")
    return {**pools, **custom}


def plot_results(output, architectures, sizes, test_surface, frontier, search_rows, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(layout="constrained")
    image = ax.imshow(test_surface, aspect="auto", cmap="viridis")
    ax.set(xticks=range(len(sizes)), xticklabels=sizes, xlabel="RecMO block size",
           yticks=range(len(architectures)), yticklabels=architectures,
           title=f"Test {metric.upper()} by architecture and block size")
    fig.colorbar(image, ax=ax, label=metric.upper())
    fig.savefig(output / "test_surface.png", dpi=150)
    plt.close(fig)

    final_horizon = max(row["horizon"] for row in frontier)
    fig, ax = plt.subplots(layout="constrained")
    final = [row for row in frontier if row["horizon"] == final_horizon]
    ax.scatter([row["mse"] for row in final], [row["w1"] for row in final],
               color="#677d8d", label="Configurations")
    pareto = sorted((row for row in final if row["pareto"]), key=lambda row: row["mse"])
    ax.plot([row["mse"] for row in pareto], [row["w1"] for row in pareto],
            "o-", color="#a53629", label="Pareto front")
    ax.set(xlabel="Test MSE", ylabel="Test marginal W1", title=f"Accuracy and realism at h={final_horizon}")
    ax.legend()
    fig.savefig(output / "pareto_final_horizon.png", dpi=150)
    plt.close(fig)

    curves = defaultdict(lambda: defaultdict(list))
    for row in search_rows:
        curves[row["method"]][row["budget"]].append(row["test_regret"])
    fig, ax = plt.subplots(layout="constrained")
    for method, curve in curves.items():
        ax.plot(list(curve), [np.mean(values) for values in curve.values()],
                marker=".", label=method.replace("_", " ").capitalize())
    ax.set(xlabel="Queried configurations", ylabel=f"Mean test {metric.upper()} regret",
           title="Validation-driven search versus test oracle")
    ax.legend()
    fig.savefig(output / "budget_search.png", dpi=150)
    plt.close(fig)


def analyse(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output == bundle or bundle in output.parents:
        raise ValueError("Analysis output must be outside the immutable forecast bundle")
    metadata, data = load_bundle(bundle)
    candidates = metadata["candidates"]
    config = metadata["config"]
    options = config["analysis"]
    metric = options["selection_metric"]
    if metric not in ("mse", "mae") or options["search_repeats"] < 1:
        raise ValueError("Use MSE or MAE selection and a positive number of search repeats")
    losses = {split: horizon_metrics(targets, predictions)
              for split, (_, targets, predictions) in data.items()}
    rows, averaged = score_configurations(candidates, losses, metric)
    ids = [row["configuration_id"] for row in rows]
    subsets = selection_pools(rows, config.get("selection_subsets", {}))
    selection = {}
    for name, allowed in subsets.items():
        chosen = select_validation(rows, allowed)
        key = chosen["configuration_id"]
        selection[name] = {
            "configuration_id": key, "configuration_count": len(set(allowed)),
            "architecture": chosen["architecture"], "block_size": chosen["block_size"],
            "rectifier_block_size": chosen["rectifier_block_size"],
            "mean_validation_loss": chosen["val_loss"],
            "mean_test_loss": chosen["test_loss"],
            "test_loss_by_seed": [
                {"seed": item["fit_seed"], "loss": float(losses["test"][metric][i].mean())}
                for i, item in enumerate(candidates) if item["configuration_id"] == key
            ],
        }
    baseline_loss = selection["unrectified_recmo"]["mean_test_loss"]
    expanded = selection.get("expanded_recmo_recmo")
    if expanded is not None:
        expanded["test_loss_difference_vs_unrectified"] = expanded["mean_test_loss"] - baseline_loss
        expanded["test_loss_ratio_vs_unrectified"] = (
            expanded["mean_test_loss"] / baseline_loss if baseline_loss else None
        )

    unrectified = [row for row in rows if row["rectifier_block_size"] is None]
    architectures = sorted({row["architecture"] for row in unrectified})
    sizes = sorted({row["block_size"] for row in unrectified})
    grid = {(row["architecture"], row["block_size"]): row for row in unrectified}
    if len(grid) != len(unrectified) or len(grid) != len(architectures) * len(sizes):
        raise ValueError("Chapter 6 requires one configuration per architecture–RecMO cell")
    test_surface = np.array([[grid[a, s]["test_loss"] for s in sizes] for a in architectures])
    val_cells = [[grid[a, s]["val_seed_losses"] for s in sizes] for a in architectures]
    search_rows = evaluate_search(val_cells, test_surface, architectures, sizes,
                                  options["search_seed"], options["search_repeats"])
    test_ranked = sorted(rows, key=lambda row: (row["test_loss"], row["configuration_id"]))
    val_ranked = sorted(rows, key=lambda row: (row["val_loss"], row["configuration_id"]))
    val_rank = {row["configuration_id"]: rank for rank, row in enumerate(val_ranked, 1)}
    ranking = []
    for rank, row in enumerate(test_ranked, 1):
        ranking.append({**{key: value for key, value in row.items() if key != "val_seed_losses"},
                        "test_rank": rank, "val_rank": val_rank[row["configuration_id"]]})

    test = averaged["test"]
    frontier = []
    for index in range(config["horizon"]):
        keep = pareto_mask(test["mse"][:, index], test["w1"][:, index])
        for pos, row in enumerate(rows):
            frontier.append({
                "horizon": index + 1, "configuration_id": row["configuration_id"],
                "rectifier_block_size": row["rectifier_block_size"],
                **{name: test[name][pos, index] for name in METRICS},
                "pareto": bool(keep[pos]),
            })
    slices, opportunity_summary = opportunities(ids, test["mse"], test["w1"], options["epsilon"])
    summary = {
        "task": config["name"], "horizon": config["horizon"],
        "process_noise_std": config["generator"]["parameters"]["sigma_s"],
        "candidate_count": len(candidates), "configuration_count": len(rows),
        "origin_count": {split: len(values[0]) for split, values in data.items()},
        "selection_metric": metric,
        "chapter_5_validation_selection": selection,
        "chapter_6_test_surface": {
            "unrectified_configuration_count": len(unrectified),
            **variance_decomposition(test_surface),
        },
        "chapter_7_retrospective_opportunity": opportunity_summary,
    }

    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    write_csv(output / "configuration_ranks.csv", ranking, tuple(ranking[0]))
    concentration = top_concentration([row for row in test_ranked
                                       if row["rectifier_block_size"] is None])
    write_csv(output / "top_concentration.csv", concentration, tuple(concentration[0]))
    write_csv(output / "budget_search.csv", search_rows, tuple(search_rows[0]))
    write_csv(output / "horizon_pareto.csv", frontier, tuple(frontier[0]))
    write_csv(output / "opportunity.csv", slices, tuple(slices[0]))
    plot_results(output, architectures, sizes, test_surface, frontier, search_rows, metric)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = analyse(args.bundle, args.out)
    print(f"Analysed {result['configuration_count']} configurations from {args.bundle}")


if __name__ == "__main__":
    main()
