"""Sequential, single-device experiments with durable per-trial evidence."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import ROOT
from .contracts import atomic_json, evaluate, load_graph, sha256
from .worker import METHODS


def run(plan, output, data_root, gpu=None, epochs=None, timeout=1800):
    if gpu is not None and (not str(gpu).strip() or "," in str(gpu) or str(gpu) in {"all", "-1"}):
        raise ValueError("Choose exactly one GPU index or UUID; omit --gpu for CPU")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {"plan": plan, "epochs_override": epochs, "timeout": timeout,
                "source_manifest_sha256": sha256(ROOT / "sources.lock.json"),
                "integration_sha256": {name: sha256(ROOT / "gad_rsi" / name)
                                       for name in ["contracts.py", "worker.py", "suite.py"]}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    receipt = output / "suite.json"
    if receipt.exists() and json.loads(receipt.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Output belongs to a different experiment; choose a new output directory")
    atomic_json(receipt, {"fingerprint": fingerprint, **identity})
    results = []
    for dataset in plan["datasets"]:
        graph = Path(data_root) / "public" / (dataset + ".npz")
        labels = Path(data_root) / "judge" / (dataset + ".npz")
        x, _ = load_graph(graph, plan.get("max_nodes", 12000))
        data_id = {"graph_sha256": sha256(graph), "labels_sha256": sha256(labels)}
        for method in plan["methods"]:
            if method not in METHODS:
                raise ValueError("Unknown method: " + method)
            for seed in plan["seeds"]:
                trial = output / dataset / method / str(seed)
                trial.mkdir(parents=True, exist_ok=True)
                trial_meta = {"fingerprint": fingerprint, "dataset": dataset, "method": method,
                              "seed": seed, **data_id}
                result_path = trial / "result.json"
                previous = None
                if result_path.exists():
                    previous = json.loads(result_path.read_text())
                    if any(previous.get(k) != v for k, v in trial_meta.items()):
                        raise ValueError("Existing trial has different input/configuration")
                    if previous["status"] == "completed":
                        if sha256(Path(previous["attempt"]) / "scores.npy") != previous["scores_sha256"]:
                            raise ValueError("Previously completed scores have changed")
                        results.append(previous)
                        continue
                attempt = trial / ("attempt-" + str(len(list(trial.glob("attempt-*"))) + 1).zfill(3))
                attempt.mkdir()
                if previous is not None and method == "ada-gad":
                    # Preserve every previous attempt. Resume only atomic pretraining
                    # checkpoints under the same code, input and configuration.
                    old_attempt = Path(previous["attempt"])
                    for checkpoint in old_attempt.glob("pretrain-[0-9].pt"):
                        shutil.copy2(checkpoint, attempt / checkpoint.name)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else "",
                           OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                           PYTHONHASHSEED=str(seed))
                command = [sys.executable, "-m", "gad_rsi.worker", "--method", method,
                           "--graph", str(graph), "--dataset", dataset, "--output", str(attempt),
                           "--seed", str(seed), "--gpu", "0" if gpu is not None else "-1",
                           "--max-nodes", str(plan.get("max_nodes", 12000))]
                if epochs is not None:
                    command += ["--epochs", str(epochs)]
                record = {**trial_meta, "attempt": str(attempt), "status": "running"}
                if previous is not None:
                    record["previous_attempt"] = previous["attempt"]
                atomic_json(result_path, record)
                try:
                    with (attempt / "worker.log").open("w") as log:
                        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                                 stderr=subprocess.STDOUT, timeout=timeout)
                    if process.returncode:
                        record.update(status="failed", returncode=process.returncode)
                    else:
                        record.update(status="completed", metrics=evaluate(labels, attempt / "scores.npy", len(x)),
                                      training=json.loads((attempt / "training.json").read_text()),
                                      scores_sha256=sha256(attempt / "scores.npy"))
                except subprocess.TimeoutExpired:
                    record.update(status="timeout")
                except (ValueError, OSError) as error:
                    record.update(status="invalid", error=str(error))
                atomic_json(result_path, record)
                results.append(record)
                atomic_json(output / "results.json", results)
                print(dataset, method, seed, record["status"], record.get("metrics", {}), flush=True)
    atomic_json(output / "results.json", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    lock_dir = ROOT / "runs" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "single-worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = run(json.loads(args.plan.read_text()), args.output, args.data_root,
                      args.gpu, args.epochs, args.timeout)
    if any(r["status"] != "completed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
