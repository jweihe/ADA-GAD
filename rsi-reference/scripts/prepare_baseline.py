"""Independent training reproduction: train, export and check new model states.

Run before research and outside Work/Judge. Source/data downloads happen here;
normal Judge calls do not train. This optional command does not replace the
existing task baseline delivered by materialize_reference.py. Label-bearing raw
and Judge data stay in this trusted reproduction directory.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad_rsi import ROOT
from gad_rsi.contracts import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--gpu", choices=["0"], default="0")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        parser.error("Use a fresh preparation destination; no existing run is overwritten")
    destination.mkdir(parents=True)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "4",
                   "OPENBLAS_NUM_THREADS": "4"}

    def run(arguments):
        subprocess.run([sys.executable, *arguments], cwd=ROOT, env=environment, check=True)

    run(["scripts/verify_sources.py"])
    data = destination / "operator-data"
    runs = destination / "training-runs"
    bundle = destination / "reference"
    run(["-m", "gad_rsi.prepare", "weibo", "reddit", "--data-root", str(data)])
    run(["-m", "gad_rsi.suite", "--plan", str(ROOT / "configs/reference-20epochs.json"),
         "--data-root", str(data), "--output", str(runs), "--gpu", "0", "--timeout", "3600"])
    run(["scripts/export_reference.py", "--runs", str(runs), "--data-root", str(data),
         "--destination", str(bundle)])
    run(["scripts/verify_reference_inference.py", "--bundle", str(bundle),
         "--output", str(destination / "reload-verification"), "--gpu", "0"])
    atomic_json(destination / "preparation.json", {
        "status": "completed", "source_revision": "71a3aca936ccfe430e3809864f256d4bf80b22ee",
        "manifest_sha256": sha256(bundle / "candidate/manifest.json"),
        "inference_verification_sha256": sha256(destination / "reload-verification/verification.json"),
        "boundary": "Stage reference/candidate and reference/graphs only into Work; operator-data is preparation/Judge-only",
        "baseline_contract": "Independent reproduction only; use materialize_reference.py for the existing frozen task baseline. These newly trained weights never replace it."})


if __name__ == "__main__":
    main()
