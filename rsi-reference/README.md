# Frozen single-GPU ADA-GAD reference

The task baseline is an existing, downloadable bundle of six trained ADA-GAD
model states: Weibo and Reddit, each with training seeds 0, 1 and 2. Download it
from [release rsi-reference-v1](https://github.com/jweihe/ADA-GAD/releases/tag/rsi-reference-v1).
Task construction, Solution and Judge require no baseline training and no access
to a contributor development machine.

## Artifact identity and retrieval

- [Direct archive download](https://github.com/jweihe/ADA-GAD/releases/download/rsi-reference-v1/ada-gad-reference-v1.tar.gz):
  30,224,871 bytes.
- Archive SHA256: `8089dd870e105be8df288911321894c262d0d626b035db8aa8feaaee7ea7dd89`.
- Candidate manifest SHA256: `9b32683cea19beaf6ade890f76b1c1b2345fac16376da037921d18df68fe51e2`.
- `reference.lock.json` commits to the archive and every member's size and hash,
  including all six weights, the inference specification and both graphs.
- The export and inference code used for this artifact is available at
  `e85c0e63d43252fb2764156e4f51b9b91a603b00` in this repository. Its unmodified
  ADA-GAD model files are pinned to `71a3aca936ccfe430e3809864f256d4bf80b22ee`.

Retrieve and verify the existing bundle using standard-library Python only:

```bash
cd rsi-reference
python3 scripts/materialize_reference.py --destination prepared/reference-v1
```

The command downloads the pinned asset, checks its length and SHA256, then
verifies every extracted regular file. It rejects unexpected members, duplicate
paths, links, traversal paths and hash mismatches. Use a fresh destination. For
offline staging, pass `--archive /path/to/ada-gad-reference-v1.tar.gz` as well.
This command performs no training or scoring.

## Frozen baseline contract

The selected reference is exactly the six numeric `.npz` states named by
`candidate/manifest.json` in this asset. Preserve those states and the locked
graph/model identities throughout the campaign. Independent retraining is useful
for reproduction, but never replaces this selected baseline.

Normal Solution copies the verified `candidate/` directory to
`WORKDIR/candidate`. Normal Judge reconstructs the fixed model using its own
source, graphs and labels, accepts bounded numeric arrays, and recomputes node
scores. Neither Solution nor Judge trains or regenerates a reference. Stored
reference scores and claimed metric files are never scoring inputs.

The archive contains:

- `candidate/manifest.json` and six numeric states with individual hashes.
- `graphs/{weibo,reddit}.npz`, containing exactly `x` and `edge_index`.
- `reference-spec.json`, `LICENSE` and `hashes.json`.
- `calibration.json`, `evaluation.json` and `inference-verification.json`.
- `reference-scores/`: original label-free score vectors, solely for the supplied
  reload-parity check. They are not needed in Work or normal Judge.

No labels, dataset pickles, Python checkpoint pickles or training-machine paths
are included. Stage only curated source, `candidate/` and label-free graphs into
Work. Do not copy the whole upstream checkout or trusted preparation directories:
the upstream repository's example data can contain labels. Judge-only labels,
protected tests and scoring code must remain outside Work. Work/Judge run offline
after staging; container isolation still needs task-level validation.

## Measured validation

The released per-dataset recipe trained each of three encoders for 20 epochs,
followed by 20 detection epochs. All six training runs used only one A100 80GB.
They took 5,975.11 seconds total with maximum training CUDA allocation
5,493,093,376 bytes. Original checkpoint and graph identities were audited before
export. Every exported tensor round trip is exact.

On October 19, 2026, each numeric state was reloaded into a fresh fixed-model
process on GPU 0, with exactly one CUDA device visible. All six all-node score
comparisons passed `rtol=1e-5, atol=1e-6`. GPU reductions are not bitwise stable;
the frozen reference's freshly recomputed mean AUROC was
**0.7483516115551683**, versus **0.7483520309762254** in the original training
receipt. Both round to 0.748352. The original and reloaded measurements remain
separate records; no label-based checkpoint or score-sign selection was used.

Freshly reloaded Weibo AUROC was 0.9278529771 (population seed standard deviation
0.0034165931); Reddit AUROC was 0.5688502461 (0.0000750768). See
`evidence/evaluation.json` for all six AUROC/AP results. The serial parity driver
took 30.58 seconds end to end; its model-scoring sections totaled 13.46 seconds.
Peak inference CUDA allocation was 3,883,723,776 bytes; measured maximum child
RSS was 2,327,712 KiB. This is an export/reload verification measurement, not a
complete isolated Judge or adaptive research cycle.

The contract/archive/materialization tests pass (31 checks), pinned public data reconstructs the
same graphs, and fixed-model construction passes on both graphs. Full clean-image
Harbor validation, phase resource caps, matched-control costs and a strong-model
research trajectory remain pending. No completed strong-model failure is claimed.

## Environment and independent training reproduction

The calibrated environment is Linux x86_64, Python 3.12.3, PyTorch 2.2.2+cu121,
PyG 2.5.3 and NumPy 1.26.4. Install on persistent storage with one eligible GPU:

```bash
RSI_PYTHON_BIN=python3.12 bash scripts/setup_linux.sh
envs/torch222/bin/python scripts/verify_sources.py
```

`wheels.lock.json` pins official Python 3.12/Linux CUDA 12.1 extension wheel
URLs, lengths and SHA256 digests. Other scientific-library versions are pinned
by the installer, with the full resolved environment recorded after installation.

For optional independent training reproduction, run:

```bash
envs/torch222/bin/python scripts/prepare_baseline.py --destination "$PWD/prepared/independent-reproduction-v1" --gpu 0
```

This separate command downloads trusted public raw data, trains six new models,
exports them and checks their reload parity. Its `operator-data/raw` and
`operator-data/judge` contain labels and must never enter Work. The combined
from-scratch driver has not yet been run as one complete invocation. The released
artifact instead comes from the already completed and audited six-run calibration,
followed by the now completed export/reload validation. Newly reproduced weights
do not change the task baseline.

## Source and license

The integration and released model states follow Apache-2.0. Bundled ADA-GAD and
PyGOD source files are unmodified and identified in `sources.lock.json`, with
their upstream licenses retained under `sources/`. Modern PyGOD is BSD-2-Clause.
The curated source directory contains no dataset pickles or model checkpoints;
trained numeric weights are delivered through the separately locked release asset.
