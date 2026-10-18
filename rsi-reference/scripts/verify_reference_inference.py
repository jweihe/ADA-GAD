"""Rescore an exported reference in fresh processes; compare every node score."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad_rsi import ROOT
from gad_rsi.contracts import atomic_json, sha256


def verify(bundle, output, gpu):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    manifest = json.loads((bundle / "candidate/manifest.json").read_text())
    receipts = json.loads((bundle / "calibration.json").read_text())
    assert manifest["inference_spec_sha256"] == sha256(ROOT / "configs/ada-inference.json")
    expected_pairs = {(dataset, seed) for dataset in ["weibo", "reddit"] for seed in [0, 1, 2]}
    assert len(manifest["models"]) == 6
    assert {(m["dataset"], m["seed"]) for m in manifest["models"]} == expected_pairs
    calibration = {(r["dataset"], r["seed"]): r for r in receipts["trials"]}
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in manifest["models"]:
        dataset, seed = entry["dataset"], entry["seed"]
        if entry["path"] != f"{dataset}/seed-{seed}.npz":
            raise ValueError("Unexpected baseline weight path")
        weights = bundle / "candidate" / entry["path"]
        assert sha256(weights) == entry["sha256"]
        record = calibration[(dataset, seed)]
        reference_path = bundle / record["reference_scores_path"]
        assert sha256(reference_path) == record["reference_scores_sha256"]
        trial_output = output / dataset / str(seed)
        trial_output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "gad_rsi.inference", "--dataset", dataset,
                   "--graph", str(bundle / "graphs" / (dataset + ".npz")),
                   "--weights", str(weights), "--output", str(trial_output), "--gpu", str(gpu)]
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0" if gpu == 0 else "",
                       "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
        with (trial_output / "inference.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=900)
        reference = np.load(reference_path, allow_pickle=False)
        actual = np.load(trial_output / "scores.npy", allow_pickle=False)
        if actual.shape != reference.shape or not np.isfinite(actual).all():
            raise ValueError("Invalid reloaded reference scores")
        close = bool(np.allclose(actual, reference, rtol=1e-5, atol=1e-6))
        result = {"dataset": dataset, "seed": seed, "allclose_rtol_1e5_atol_1e6": close,
                  "bitwise_equal": bool(np.array_equal(actual, reference)),
                  "max_absolute_difference": float(np.max(np.abs(actual - reference))),
                  "inference": json.loads((trial_output / "inference.json").read_text())}
        results.append(result)
        atomic_json(output / "verification.json", {"status": "running", "trials": results})
        if not close:
            raise ValueError("Reloaded model scores differ from the frozen reference")
    report = {"status": "passed", "device": "cuda:0" if gpu == 0 else "cpu", "trials": results,
              "total_inference_seconds": sum(r["inference"]["runtime_seconds"] for r in results),
              "scope": "Numeric export and fresh-process fixed-model inference parity; not Harbor isolation validation"}
    atomic_json(output / "verification.json", report)
    atomic_json(bundle / "inference-verification.json", report)
    receipts["inference_parity"] = "passed"
    atomic_json(bundle / "calibration.json", receipts)
    hashes = {str(path.relative_to(bundle)): sha256(path) for path in bundle.rglob("*")
              if path.is_file() and path != bundle / "hashes.json"}
    atomic_json(bundle / "hashes.json", hashes)
    print(json.dumps({"status": "passed", "models": len(results),
                      "max_absolute_difference": max(r["max_absolute_difference"] for r in results),
                      "total_inference_seconds": report["total_inference_seconds"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=[-1, 0], default=0)
    args = parser.parse_args()
    verify(args.bundle, args.output, args.gpu)


if __name__ == "__main__":
    main()
