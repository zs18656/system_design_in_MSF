"""Versioned, checksummed forecast bundles with no serialized estimators."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .metrics import horizon_metrics
from .tasks import SPLITS

SCHEMA_VERSION = 1


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_bundle(destination, metadata, windows, predictions):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Bundle already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".bundle-", dir=destination.parent))
    try:
        arrays = {}
        metric_rows = []
        ids = [candidate["candidate_id"] for candidate in metadata["candidates"]]
        for split in SPLITS:
            data = windows[split]
            point = predictions[split]
            arrays.update({f"{split}_origins": data.origins, f"{split}_targets": data.targets,
                           f"{split}_predictions": point})
            metrics = horizon_metrics(data.targets, point)
            for index, candidate_id in enumerate(ids):
                for step in range(data.targets.shape[1]):
                    metric_rows.append({
                        "split": split, "candidate_id": candidate_id, "horizon": step + 1,
                        **{name: values[index, step] for name, values in metrics.items()},
                    })
        write_json(staging / "metadata.json", metadata)
        np.savez_compressed(staging / "forecasts.npz", candidate_ids=np.asarray(ids), **arrays)
        write_csv(staging / "metrics.csv", metric_rows,
                  ("split", "candidate_id", "horizon", "mse", "mae", "w1"))
        write_json(staging / "manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "sha256": {name: digest(staging / name)
                       for name in ("metadata.json", "forecasts.npz", "metrics.csv")},
        })
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported bundle schema version")
    for name in ("metadata.json", "forecasts.npz", "metrics.csv"):
        if digest(directory / name) != manifest["sha256"][name]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    ids = [entry["candidate_id"] for entry in metadata["candidates"]]
    config = metadata["config"]
    horizon, context = config["horizon"], config["context"]
    boundaries = metadata["scaling"]["boundaries"]
    data = {}
    with np.load(directory / "forecasts.npz", allow_pickle=False) as arrays:
        if not ids or len(set(ids)) != len(ids) or arrays["candidate_ids"].tolist() != ids:
            raise ValueError("Candidate registry does not match forecast order")
        for split, start, stop in zip(SPLITS, boundaries[:-1], boundaries[1:], strict=True):
            origins = arrays[f"{split}_origins"]
            targets = arrays[f"{split}_targets"]
            predictions = arrays[f"{split}_predictions"]
            expected = np.arange(max(start, context) - 1, stop - horizon)
            if (not len(expected) or not np.array_equal(origins, expected)
                    or targets.shape != (len(origins), horizon)
                    or predictions.shape != (len(ids), *targets.shape)):
                raise ValueError(f"Invalid split alignment in {split}")
            data[split] = (origins, targets, predictions)
    return metadata, data
