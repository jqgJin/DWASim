"""Prepare the exact HGBn-ACM, HGBn-DBLP, and HGBn-IMDB relations.

The raw HGB node identifiers are global and grouped by node type.  The
historical DWASim implementation instead expects one rectangular interaction
matrix for every ordered pair of node types.  This script performs only that
lossless conversion and preserves the official training/test label files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
PROCESSED_ROOT = ROOT / "data" / "processed"

EXPECTED_COUNTS = {
    "ACM": {0: 3025, 1: 5959, 2: 56, 3: 1902},
    "DBLP": {0: 4057, 1: 14328, 2: 7723, 3: 20},
    "IMDB": {0: 4932, 1: 2393, 2: 6124, 3: 7971},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_info(dataset_dir: Path) -> dict:
    with (dataset_dir / "info.dat").open("r", encoding="utf-8") as stream:
        return json.load(stream)


def scan_nodes(node_path: Path) -> tuple[dict[int, int], dict[int, int]]:
    counts: Counter[int] = Counter()
    first_id: dict[int, int] = {}
    last_id: dict[int, int] = {}

    with node_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t", 3)
            if len(fields) < 3:
                raise ValueError(f"Malformed node record: {line[:80]!r}")
            node_id = int(fields[0])
            node_type = int(fields[2])
            counts[node_type] += 1
            first_id.setdefault(node_type, node_id)
            last_id[node_type] = node_id

    shifts: dict[int, int] = {}
    running = 0
    for node_type in sorted(counts):
        shifts[node_type] = running
        if first_id[node_type] != running:
            raise ValueError(
                f"Node type {node_type} starts at {first_id[node_type]}, expected {running}."
            )
        expected_last = running + counts[node_type] - 1
        if last_id[node_type] != expected_last:
            raise ValueError(
                f"Node type {node_type} ends at {last_id[node_type]}, expected {expected_last}."
            )
        running += counts[node_type]
    return dict(counts), shifts


def read_labels(
    path: Path,
    target_shift: int,
    class_count: int,
    *,
    multilabel: bool,
) -> tuple[np.ndarray, np.ndarray]:
    node_ids: list[int] = []
    label_sets: list[list[int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise ValueError(f"Malformed label record: {line!r}")
            record_labels = [int(value) for value in fields[3].split(",")]
            if not multilabel and len(record_labels) != 1:
                raise ValueError(f"Expected one class label, found {fields[3]!r}")
            if not record_labels or min(record_labels) < 0 or max(record_labels) >= class_count:
                raise ValueError(f"Class label outside [0, {class_count}): {fields[3]!r}")
            node_ids.append(int(fields[0]) - target_shift)
            label_sets.append(record_labels)

    ids = np.asarray(node_ids, dtype=np.int64)
    if len(np.unique(ids)) != ids.size:
        raise ValueError(f"Duplicate node identifiers in {path}")
    if multilabel:
        labels = np.zeros((ids.size, class_count), dtype=np.int8)
        for row, record_labels in enumerate(label_sets):
            labels[row, record_labels] = 1
    else:
        labels = np.asarray([values[0] for values in label_sets], dtype=np.int64)
    return ids, labels


def prepare_dataset(dataset: str) -> dict:
    raw_dir = RAW_ROOT / dataset
    out_dir = PROCESSED_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    info = read_info(raw_dir)
    counts, shifts = scan_nodes(raw_dir / "node.dat")
    if counts != EXPECTED_COUNTS[dataset]:
        raise ValueError(f"{dataset} node counts {counts} do not match {EXPECTED_COUNTS[dataset]}")

    relation_info = {
        int(key): {
            "source_type": int(value["start"]),
            "target_type": int(value["end"]),
            "meaning": value["meaning"],
        }
        for key, value in info["link.dat"]["link type"].items()
    }

    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    values: dict[int, list[float]] = defaultdict(list)
    with (raw_dir / "link.dat").open("r", encoding="utf-8") as stream:
        for line in stream:
            source, target, relation, weight = line.rstrip("\n").split("\t")
            relation_id = int(relation)
            meta = relation_info[relation_id]
            rows[relation_id].append(int(source) - shifts[meta["source_type"]])
            cols[relation_id].append(int(target) - shifts[meta["target_type"]])
            values[relation_id].append(float(weight))

    relation_manifest: dict[str, dict] = {}
    matrices: dict[int, sp.csr_matrix] = {}
    for relation_id in sorted(relation_info):
        meta = relation_info[relation_id]
        shape = (counts[meta["source_type"]], counts[meta["target_type"]])
        matrix = sp.coo_matrix(
            (values[relation_id], (rows[relation_id], cols[relation_id])),
            shape=shape,
            dtype=np.float32,
        ).tocsr()
        matrix.sum_duplicates()
        matrix.sort_indices()
        filename = f"relation_{relation_id}.npz"
        sp.save_npz(out_dir / filename, matrix, compressed=True)
        matrices[relation_id] = matrix
        relation_manifest[str(relation_id)] = {
            **meta,
            "shape": list(shape),
            "records": len(rows[relation_id]),
            "nonzero": int(matrix.nnz),
            "file": filename,
        }

    label_names = info["label.dat"]["node type"]["0"]
    class_count = len(label_names)
    multilabel = dataset == "IMDB"
    train_ids, train_labels = read_labels(
        raw_dir / "label.dat",
        shifts[0],
        class_count,
        multilabel=multilabel,
    )
    test_ids, test_labels = read_labels(
        raw_dir / "label.dat.test",
        shifts[0],
        class_count,
        multilabel=multilabel,
    )
    if np.intersect1d(train_ids, test_ids).size:
        raise ValueError(f"{dataset} train/test label node sets overlap")
    np.savez_compressed(
        out_dir / "labels.npz",
        train_ids=train_ids,
        train_labels=train_labels,
        test_ids=test_ids,
        test_labels=test_labels,
    )

    reverse_checks: list[dict] = []
    for left_id, left in relation_info.items():
        for right_id, right in relation_info.items():
            if left_id >= right_id:
                continue
            if (
                left["source_type"] == right["target_type"]
                and left["target_type"] == right["source_type"]
            ):
                difference = matrices[left_id] - matrices[right_id].T
                difference.eliminate_zeros()
                reverse_checks.append(
                    {
                        "relations": [left_id, right_id],
                        "exact_transposes": difference.nnz == 0,
                        "difference_nonzero": int(difference.nnz),
                    }
                )

    manifest = {
        "dataset": dataset,
        "node_counts": {str(key): value for key, value in counts.items()},
        "node_shifts": {str(key): value for key, value in shifts.items()},
        "relations": relation_manifest,
        "labels": {
            "train": int(train_ids.size),
            "test": int(test_ids.size),
            "class_count": class_count,
            "class_names": {str(key): value for key, value in label_names.items()},
            "multilabel": multilabel,
            "train_assignments": int(train_labels.sum()) if multilabel else int(train_ids.size),
            "test_assignments": int(test_labels.sum()) if multilabel else int(test_ids.size),
            "train_mean_cardinality": (
                float(train_labels.sum(axis=1).mean()) if multilabel else 1.0
            ),
            "test_mean_cardinality": (
                float(test_labels.sum(axis=1).mean()) if multilabel else 1.0
            ),
        },
        "reverse_relation_checks": reverse_checks,
        "raw_sha256": {
            name: sha256(raw_dir / name)
            for name in ["node.dat", "link.dat", "label.dat", "label.dat.test", "info.dat"]
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["ACM", "DBLP", "IMDB", "all"], default="all"
    )
    args = parser.parse_args()
    datasets = ["ACM", "DBLP", "IMDB"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        manifest = prepare_dataset(dataset)
        print(
            f"{dataset}: nodes={sum(manifest['node_counts'].values())}, "
            f"relations={len(manifest['relations'])}, "
            f"train_labels={manifest['labels']['train']}, "
            f"test_labels={manifest['labels']['test']}"
        )


if __name__ == "__main__":
    main()
