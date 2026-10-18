"""Reconstruct the fixed ADA-GAD model and score inert candidate weights.

This module never loads training checkpoints, candidate Python, or labels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

from . import ROOT
from .artifacts import load_weights
from .contracts import atomic_json, load_graph, sha256


def fixed_spec(dataset):
    spec = json.loads((ROOT / "configs/ada-inference.json").read_text())
    if spec["schema_version"] != 1 or dataset not in spec["datasets"]:
        raise ValueError("Unknown fixed inference specification")
    return spec["datasets"][dataset]


def upstream_types():
    source = ROOT / "sources/ADA-GAD"
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    for record in lock["files"]:
        if record["repository"] != "jweihe/ADA-GAD":
            continue
        payload = (ROOT / record["local_path"]).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if record["commit"] != "71a3aca936ccfe430e3809864f256d4bf80b22ee" or actual != record["blob_sha"]:
            raise ValueError("Fixed inference source mismatch")
    loaded = sys.modules.get("pygod")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("Use a fresh process: a different PyGOD implementation is already loaded")
    sys.path[:0] = [str(source / "pygod"), str(source)]
    import torch_geometric.utils as utils
    for name in ["degree", "sort_edge_index", "subgraph"]:
        module_name = "torch_geometric.utils." + name
        if module_name not in sys.modules:
            module = types.ModuleType(module_name)
            setattr(module, name, getattr(utils, name))
            sys.modules[module_name] = module
    from pygod.models.adanet import ADANET, ADANET_Base
    return ADANET, ADANET_Base


def build_fixed_model(dataset, graph, gpu):
    import torch
    from torch_geometric.utils import add_remaining_self_loops, to_dense_adj
    if gpu not in {-1, 0}:
        raise ValueError("Use CPU or one visible CUDA device")
    if gpu == 0 and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("GPU scoring requires exactly one visible CUDA device")
    spec = fixed_spec(dataset)
    if tuple(graph.x.shape) != (spec["nodes"], spec["features"]):
        raise ValueError("Graph dimensions differ from fixed inference specification")
    detector_type, base_type = upstream_types()
    detector = detector_type(gpu=gpu, **spec["params"])
    graph.edge_index = add_remaining_self_loops(graph.edge_index, num_nodes=graph.num_nodes)[0]
    if detector.alpha is None:
        adjacency = to_dense_adj(graph.edge_index)[0]
        detector.alpha = torch.std(adjacency).detach() / (torch.std(graph.x).detach() + torch.std(adjacency).detach())
        del adjacency
    detector.batch_size = graph.num_nodes
    detector.num_node = graph.num_nodes
    names = ["hid_dim", "dropout", "act", "aggr", "use_encoder_num",
             "attr_encoder_name", "struct_encoder_name", "topology_encoder_name",
             "attr_decoder_name", "struct_decoder_name", "node_encoder_num_layers",
             "edge_encoder_num_layers", "subgraph_encoder_num_layers",
             "attr_decoder_num_layers", "struct_decoder_num_layers"]
    kwargs = {name: getattr(detector, name) for name in names}
    detector.model = base_type(in_dim=graph.num_node_features,
                               attention=detector.attention_value,
                               num_node=graph.num_nodes, **kwargs).to(detector.device)
    detector.model.eval()
    return detector


def load_fixed_graph(dataset, path):
    import torch
    from torch_geometric.data import Data
    spec = fixed_spec(dataset)
    if sha256(path) != spec["graph_sha256"]:
        raise ValueError("Input graph differs from the frozen graph bytes")
    x, edges = load_graph(path, max_nodes=12000)
    return Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edges), num_nodes=len(x))


def score(dataset, graph_path, weights_path, gpu):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    graph = load_fixed_graph(dataset, graph_path)
    detector = build_fixed_model(dataset, graph, gpu)
    expected = {name: value.detach().cpu().numpy() for name, value in detector.model.state_dict().items()}
    arrays = load_weights(weights_path, expected)
    detector.model.load_state_dict({name: torch.from_numpy(value) for name, value in arrays.items()}, strict=True)
    with torch.no_grad():
        scores = np.asarray(detector.decision_function(graph))
    if scores.shape != (graph.num_nodes,) or not np.isfinite(scores).all():
        raise ValueError("Fixed model produced invalid node scores")
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["weibo", "reddit"], required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=[-1, 0], default=-1)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if args.gpu == 0 else ""
    import torch
    if args.gpu == 0:
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    scores = score(args.dataset, args.graph, args.weights, args.gpu)
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output / "scores.tmp"
    with temporary.open("wb") as stream:
        np.save(stream, scores, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(args.output / "scores.npy")
    record = {"dataset": args.dataset, "nodes": len(scores), "device": "cuda:0" if args.gpu == 0 else "cpu",
              "graph_sha256": sha256(args.graph), "weights_sha256": sha256(args.weights),
              "inference_spec_sha256": sha256(ROOT / "configs/ada-inference.json"),
              "scores_sha256": sha256(args.output / "scores.npy"),
              "runtime_seconds": time.monotonic() - start,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.gpu == 0 else None}
    atomic_json(args.output / "inference.json", record)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
