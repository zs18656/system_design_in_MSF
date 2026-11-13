"""Part I command line entry point."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

from . import __version__
from .bundle import save_bundle
from .candidates import fit_predict, registry
from .tasks import mackey_glass, split_windows


def train(config_path, output):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if "noise_regimes" in config:
        raise ValueError("Expand the noise preset with python -m thesis_forecasting.noise first")
    generator = config["generator"]
    if generator["name"] != "mackey_glass":
        raise ValueError("Only the Mackey–Glass generator is currently supported")
    series = mackey_glass(**generator["parameters"])
    windows, scaling = split_windows(series, config["context"], config["horizon"],
                                     config["split_fractions"])
    candidates = registry(config)
    predictions = fit_predict(config, windows, scaling, candidates)
    metadata = {
        "config": config, "scaling": scaling, "sampling_interval": 1,
        "origin_convention": "zero-based index of last observed value",
        "forecast_units": "original observation units",
        "versions": {
            "reference": __version__, "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
        },
        "candidates": candidates,
    }
    save_bundle(output, metadata, windows, predictions)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    meta = train(args.config, args.out)
    print(f"Saved {len(meta['candidates'])} deployed candidates to {args.out}")


if __name__ == "__main__":
    main()
