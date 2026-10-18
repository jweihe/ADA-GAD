"""One isolated-import training process. This command never accepts a label path."""
import argparse
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np

from . import ROOT
from .contracts import atomic_json, load_graph

METHODS = {"iforest": None, "ada-gad": None, "dominant": "DOMINANT",
           "conad": "CONAD", "gadnr": "GADNR", "anomalydae": "AnomalyDAE"}


def seed_torch(seed, gpu):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if gpu >= 0:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("GPU execution requires exactly one visible CUDA device")
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit_ada(graph, opts, output):
    """Keep upstream two-stage training; omit its unused dense edge-score report."""
    import torch
    from torch_geometric.utils import add_remaining_self_loops

    source = ROOT / "sources" / "ADA-GAD"
    sys.path[:0] = [str(source / "pygod"), str(source)]
    # ADA's PyG 2.2 import paths were renamed. Keep its source bytes intact
    # and expose the same public functions under those legacy module paths.
    import torch_geometric.utils as pyg_utils
    for name in ["degree", "sort_edge_index", "subgraph"]:
        module_name = "torch_geometric.utils." + name
        if module_name not in sys.modules:
            module = types.ModuleType(module_name)
            setattr(module, name, getattr(pyg_utils, name))
            sys.modules[module_name] = module
    from model.utils import build_args, create_optimizer, load_best_configs
    from model.models import build_model
    from pygod.models import ADANET

    previous = sys.argv
    try:
        sys.argv = ["ada-adapter", "--dataset", opts.dataset, "--device", str(opts.gpu)]
        args = build_args()
    finally:
        sys.argv = previous
    args = load_best_configs(args, str(source / "config_ada-gad.yml"))
    if args.alpha_f == "None":
        args.alpha_f = None
    args.num_features = graph.x.shape[1]
    if opts.epochs is not None:
        args.max_epoch = args.max_epoch_f = opts.epochs
    for key, value in opts.params.items():
        if not hasattr(args, key) or key in {"device", "dataset", "num_features", "use_ssl", "use_encoder_num"}:
            raise ValueError("Unsupported ADA parameter: " + key)
        setattr(args, key, value)
    device = torch.device("cuda:0" if opts.gpu >= 0 else "cpu")
    graph.edge_index = add_remaining_self_loops(graph.edge_index, num_nodes=graph.num_nodes)[0]
    models = build_model(args)
    for index, model in enumerate(models):
        model.to(device)
        optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
        scheduler = (torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: (1 + np.cos(epoch * np.pi / args.max_epoch)) * 0.5
        ) if args.scheduler else None)
        checkpoint = output / ("pretrain-" + str(index) + ".pt")
        start_epoch = 0
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(saved["scheduler"])
            start_epoch = saved["epoch"]
            random.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"].cpu())
            if opts.gpu >= 0:
                torch.cuda.set_rng_state_all([r.cpu() for r in saved["cuda_rng"]])
        x, edges = graph.x.to(device), graph.edge_index.to(device)
        for epoch in range(start_epoch, args.max_epoch_f):
            model.train()
            optimizer.zero_grad()
            loss, _, _ = model(x, edges)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if opts.gpu >= 0 else []}, temporary)
            temporary.replace(checkpoint)
            atomic_json(output / "progress.json", {"stage": "pretrain", "encoder": index,
                        "completed_epochs": epoch + 1, "target_epochs": args.max_epoch_f})
        model.eval()
    mapping = {"hid_dim": "num_hidden", "alpha": "alpha_f", "dropout": "dropout_f",
               "lr": "lr_f", "epoch": "max_epoch", "loss_name": "loss_f",
               "loss_weight": "loss_weight_f", "T": "T_f", "aggr": "aggr_f",
               "attention": "attention", "use_encoder_num": "use_encoder_num",
               "attr_encoder_name": "attr_encoder", "struct_encoder_name": "struct_encoder",
               "topology_encoder_name": "topology_encoder", "attr_decoder_name": "attr_decoder",
               "struct_decoder_name": "struct_decoder", "node_encoder_num_layers": "node_encoder_num_layers",
               "edge_encoder_num_layers": "edge_encoder_num_layers", "subgraph_encoder_num_layers": "subgraph_encoder_num_layers",
               "attr_decoder_num_layers": "attr_decoder_num_layers", "struct_decoder_num_layers": "struct_decoder_num_layers",
               "sparse_attention_weight": "sparse_attention_weight", "theta": "theta", "eta": "eta"}
    detector = ADANET(gpu=opts.gpu, **{k: getattr(args, v) for k, v in mapping.items()})
    atomic_json(output / "progress.json", {"stage": "decoder_training", "epochs": args.max_epoch})
    detector.fit(graph, pretrain_attr_encoder=models[0].encoder,
                 pretrain_struct_encoder=models[1].encoder,
                 pretrain_topology_encoder=models[2].encoder)
    return detector, {k: getattr(args, v) for k, v in mapping.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--gpu", type=int, choices=[-1, 0], default=-1)
    parser.add_argument("--max-nodes", type=int, default=12000)
    parser.add_argument("--params", default="{}")
    opts = parser.parse_args()
    if opts.seed < 0 or (opts.epochs is not None and opts.epochs < 1):
        parser.error("seed must be nonnegative and epochs positive")
    opts.params = json.loads(opts.params)
    opts.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    x, edges = load_graph(opts.graph, opts.max_nodes)
    if opts.method == "iforest":
        from sklearn.ensemble import IsolationForest
        detector = IsolationForest(n_estimators=100, random_state=opts.seed, n_jobs=1, **opts.params)
        detector.fit(x)
        scores = -detector.score_samples(x)
        params = detector.get_params()
        import joblib
        joblib.dump(detector, opts.output / "checkpoint.joblib")
        peak = None
    else:
        import torch
        seed_torch(opts.seed, opts.gpu)
        from torch_geometric.data import Data
        graph = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edges), num_nodes=len(x))
        if opts.method == "ada-gad":
            detector, params = fit_ada(graph, opts, opts.output)
        else:
            sys.path.insert(0, str(ROOT / "sources" / "pygod"))
            from pygod import detector as library
            cls = getattr(library, METHODS[opts.method])
            if opts.method == "gadnr":
                # Upstream GADNR overrides DeepDetector's device transfer.
                # Move only the input at forward time, after CPU preprocessing.
                class GADNRDeviceAdapter(cls):
                    def forward_model(self, data):
                        return super().forward_model(data.clone().to(self.device))
                cls = GADNRDeviceAdapter
            params = {"gpu": opts.gpu, "epoch": opts.epochs or 100, "verbose": 0, **opts.params}
            if any(k in opts.params for k in ["gpu", "epoch", "verbose", "compile_model"]):
                raise ValueError("Device, epochs, verbosity and compilation are fixed by the runner")
            detector = cls(**params)
            detector.fit(graph)
        with torch.no_grad():
            scores = detector.decision_function(graph)
        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()
        torch.save({"method": opts.method, "params": params, "state_dict": detector.model.state_dict()},
                   opts.output / "checkpoint.pt")
        peak = torch.cuda.max_memory_allocated() if opts.gpu >= 0 else None
    scores = np.asarray(scores)
    if scores.shape != (len(x),) or not np.isfinite(scores).all():
        raise ValueError("Detector returned invalid scores")
    temporary = opts.output / "scores.tmp"
    with temporary.open("wb") as stream:
        np.save(stream, scores)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(opts.output / "scores.npy")
    atomic_json(opts.output / "training.json", {"method": opts.method, "seed": opts.seed,
                "params": params, "nodes": len(x), "edges": edges.shape[1],
                "runtime_seconds": time.monotonic() - start, "peak_cuda_allocated_bytes": peak})


if __name__ == "__main__":
    main()
