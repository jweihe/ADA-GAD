"""Export six operator-owned ADA-GAD checkpoints to a label-free reference.

Only run on trusted training outputs created by the preparation workflow. The
candidate/Judge interface uses gad_rsi.artifacts and never loads these pickles.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad_rsi import ROOT
from gad_rsi.artifacts import load_weights, write_weights
from gad_rsi.contracts import atomic_json, load_graph, sha256


def export(runs, data_root, destination, expected_receipt=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from gad_rsi.inference import build_fixed_model, fixed_spec, load_fixed_graph
    expected = {}
    if expected_receipt is not None:
        receipt = json.loads(Path(expected_receipt).read_text())
        expected = {(trial["dataset"], trial["seed"]): trial["final_checkpoint_sha256"]
                    for trial in receipt["trials"]}
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Use a new reference destination; existing artifacts are not overwritten")
    destination.mkdir(parents=True)
    (destination / "candidate").mkdir()
    (destination / "graphs").mkdir()
    (destination / "reference-scores").mkdir()
    manifest = {"schema_version": 1, "training_revision": "released-ada-gad-20epochs",
                "source_revision": "71a3aca936ccfe430e3809864f256d4bf80b22ee",
                "training_config_sha256": sha256(ROOT / "sources/ADA-GAD/config_ada-gad.yml"),
                "inference_spec_sha256": sha256(ROOT / "configs/ada-inference.json"), "models": []}
    calibration = []
    for dataset in ["weibo", "reddit"]:
        graph_path = Path(data_root) / "public" / (dataset + ".npz")
        load_graph(graph_path)
        assert sha256(graph_path) == fixed_spec(dataset)["graph_sha256"]
        shutil.copyfile(graph_path, destination / "graphs" / graph_path.name)
        (destination / "candidate" / dataset).mkdir()
        for seed in [0, 1, 2]:
            record = json.loads((Path(runs) / dataset / "ada-gad" / str(seed) / "result.json").read_text())
            if (record.get("status"), record.get("dataset"), record.get("seed"), record.get("method")) != (
                    "completed", dataset, seed, "ada-gad"):
                raise ValueError("Reference trial is not successfully complete")
            attempt = Path(record["attempt"])
            checkpoint = attempt / "checkpoint.pt"
            checkpoint_sha = sha256(checkpoint)
            if expected and checkpoint_sha != expected[(dataset, seed)]:
                raise ValueError("Checkpoint differs from the audited calibration receipt")
            # Trusted operator-owned input only; this is not a submission loader.
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["method"] != "ada-gad" or saved["params"] != fixed_spec(dataset)["params"]:
                raise ValueError("Checkpoint does not use the fixed released reference specification")
            graph = load_fixed_graph(dataset, graph_path)
            detector = build_fixed_model(dataset, graph, -1)
            detector.model.load_state_dict(saved["state_dict"], strict=True)
            arrays = {key: value.detach().cpu().numpy() for key, value in saved["state_dict"].items()}
            schema = {key: value.detach().cpu().numpy() for key, value in detector.model.state_dict().items()}
            relative = f"{dataset}/seed-{seed}.npz"
            weight_path = destination / "candidate" / relative
            write_weights(weight_path, arrays)
            loaded = load_weights(weight_path, schema)
            for key in arrays:
                np.testing.assert_array_equal(loaded[key], arrays[key])
            scores = attempt / "scores.npy"
            if sha256(scores) != record["scores_sha256"]:
                raise ValueError("Reference scores differ from the recorded trial")
            scores_relative = f"reference-scores/{dataset}-seed-{seed}.npy"
            shutil.copyfile(scores, destination / scores_relative)
            manifest["models"].append({"dataset": dataset, "seed": seed, "path": relative,
                                       "sha256": sha256(weight_path), "bytes": weight_path.stat().st_size})
            calibration.append({"dataset": dataset, "seed": seed, "metrics": record["metrics"],
                                "training": record["training"], "graph_sha256": record["graph_sha256"],
                                "source_checkpoint_sha256": checkpoint_sha,
                                "reference_scores_path": scores_relative,
                                "reference_scores_sha256": record["scores_sha256"]})
            del detector, graph, saved, arrays, schema, loaded
    shutil.copyfile(ROOT / "configs/ada-inference.json", destination / "reference-spec.json")
    shutil.copyfile(ROOT / "sources/ADA-GAD/LICENSE", destination / "LICENSE")
    atomic_json(destination / "candidate/manifest.json", manifest)
    atomic_json(destination / "calibration.json", {"trials": calibration, "export_round_trip": "exact",
                "inference_parity": "pending", "labels_included": False,
                "provenance_limit": "Training provenance is supported by operator records, not established by weight files alone."})
    hashes = {str(path.relative_to(destination)): sha256(path) for path in destination.rglob("*") if path.is_file()}
    atomic_json(destination / "hashes.json", hashes)
    print(json.dumps({"destination": str(destination), "models": len(manifest["models"]),
                      "manifest_sha256": sha256(destination / "candidate/manifest.json"),
                      "numeric_round_trip": "exact", "inference_parity": "pending"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-receipt", type=Path)
    args = parser.parse_args()
    export(args.runs, args.data_root, args.destination, args.expected_receipt)


if __name__ == "__main__":
    main()
