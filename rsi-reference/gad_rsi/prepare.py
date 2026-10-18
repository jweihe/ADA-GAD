"""Operator-only preparation of hash-pinned official data and label-free graphs."""
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from . import ROOT
from .contracts import atomic_json, load_graph, sha256


def graph_bytes(payload, name):
    with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
        members = [n for n in zipped.namelist()
                   if n.endswith(".pt") and not n.startswith("__MACOSX/")]
        if members != [name + ".pt"]:
            raise ValueError("Unexpected graph files in the official dataset archive")
        if zipped.getinfo(members[0]).file_size > 1024 ** 3:
            raise ValueError("Graph archive exceeds the preparation size limit")
        return zipped.read(members[0])


def prepare(name, data_root):
    lock = json.loads((ROOT / "data.lock.json").read_text())
    entry = next((item for item in lock["files"] if item["name"] == name), None)
    if entry is None:
        raise ValueError("Dataset is not in the pinned data manifest: " + name)
    data_root = Path(data_root)
    for directory in ["raw", "public", "judge", "receipts"]:
        (data_root / directory).mkdir(parents=True, exist_ok=True)
    archive = data_root / "raw" / entry["file"]
    if not archive.exists():
        url = "https://raw.githubusercontent.com/" + lock["repository"] + "/" + lock["commit"] + "/" + entry["file"]
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read(entry["size"] + 1)
        if len(payload) != entry["size"]:
            raise ValueError("Dataset archive size mismatch")
        temporary = archive.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(archive)
    payload = archive.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    if blob != entry["git_blob_sha"]:
        raise ValueError("Dataset archive differs from its pinned official Git blob")
    public = data_root / "public" / (name + ".npz")
    labels = data_root / "judge" / (name + ".npz")
    receipt_path = data_root / "receipts" / (name + ".json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt["source_commit"] != lock["commit"]
                or sha256(public) != receipt["graph_sha256"]
                or sha256(labels) != receipt["labels_sha256"]):
            raise ValueError("Prepared data differs from its recorded provenance")
        print(json.dumps({"status": "verified_existing", **receipt}, indent=2))
        return
    # These are PyTorch pickles. Load only after verifying the pinned official bytes.
    import torch
    graph = torch.load(io.BytesIO(graph_bytes(payload, name)), map_location="cpu", weights_only=False)
    x = graph.x.detach().cpu().numpy()
    edges = graph.edge_index.detach().cpu().numpy()
    raw_y = graph.y.detach().cpu().numpy().reshape(-1)
    if not np.isfinite(raw_y).all() or not np.isin(raw_y, [0, 1, 2, 3]).all():
        raise ValueError("Unexpected anomaly-label encoding")
    if public.exists() or labels.exists():
        raise FileExistsError("Prepared data exists; use its receipt or choose a fresh data directory")
    np.savez_compressed(public, x=np.asarray(x, dtype=np.float32), edge_index=np.asarray(edges, dtype=np.int64))
    np.savez_compressed(labels, y=(raw_y > 0).astype(np.int64), original_label=raw_y)
    load_graph(public, max_nodes=max(len(x), 2))
    receipt = {"dataset": name, "source_commit": lock["commit"], "archive_sha256": sha256(archive),
               "graph_sha256": sha256(public), "labels_sha256": sha256(labels),
               "nodes": len(x), "edges": edges.shape[1], "features": x.shape[1],
               "anomalies": int((raw_y > 0).sum()),
               "protocol": "transductive_unsupervised_all_nodes_engineering_protocol",
               "label_visibility": "operator_evaluator_only; filesystem sandbox not supplied by this research CLI"}
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    for name in args.datasets:
        prepare(name, args.data_root)


if __name__ == "__main__":
    main()
