"""Shared graph/score contracts. Training graphs contain no labels or masks."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_graph(path, max_nodes=12000):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"x", "edge_index"}:
            raise ValueError("Training archive must contain exactly x and edge_index")
        x, edges = data["x"], data["edge_index"]
    if x.ndim != 2 or not 1 < x.shape[0] <= max_nodes or x.shape[1] == 0:
        raise ValueError("Invalid graph size or configured dense-graph limit exceeded")
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError("Features must be finite numeric values")
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("edge_index must be an integer array with shape (2, E)")
    if edges.size and (edges.min() < 0 or edges.max() >= len(x)):
        raise ValueError("Edge endpoint out of bounds")
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(edges, dtype=np.int64)


def evaluate(labels_path, scores_path, expected_nodes):
    with np.load(labels_path, allow_pickle=False) as data:
        y = data["y"]
    scores = np.load(scores_path, allow_pickle=False)
    if y.shape != (expected_nodes,) or scores.shape != y.shape:
        raise ValueError("A score is required for every node, in original node order")
    if not np.isfinite(scores).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite labels or scores")
    if not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("Evaluation needs binary labels with both classes present")
    return {"roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores)),
            "nodes": len(y), "anomalies": int(y.sum())}
