"""Expand a shared experiment design into individual Mackey–Glass task configs."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .bundle import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    regimes = config.pop("noise_regimes")
    args.out.mkdir(parents=True, exist_ok=False)
    for regime, sigma in regimes.items():
        task = deepcopy(config)
        task["name"] = f"{config['name']}_{regime}"
        task["generator"]["parameters"]["sigma_s"] = sigma
        write_json(args.out / f"{regime}.json", task)


if __name__ == "__main__":
    main()
