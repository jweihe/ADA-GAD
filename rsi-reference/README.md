# Reproducible single-GPU ADA-GAD reference preparation

This directory delivers the source and executable preparation interface for the
graph-denoising research task. The baseline is the released training procedure,
materialized once during trusted task preparation. It does not require access to
any checkpoint stored on the contributor's development machines.

## Baseline production contract

1. Verify the bundled source bytes against `sources.lock.json`. ADA-GAD is fixed
   at `71a3aca936ccfe430e3809864f256d4bf80b22ee`; the modern PyGOD reference library
   is fixed at `c84dccafa9bf587cd03cfd3e89b9be7a8f3337e8`.
2. Fetch Weibo and Reddit from PyGOD data revision
   `77976df4129f5f753907beea4945bd3e21d0251f`. Check archive length and Git blob
   identity before unpickling the official data in trusted preparation.
3. Train the released ADA-GAD configuration for seeds 0, 1 and 2 on each graph.
   Each of the three encoders is pretrained for 20 epochs, followed by 20
   detection epochs. Graph preprocessing, model dimensions and scoring are fixed.
4. Export the six trusted training outputs to numeric NumPy archives. Check every
   tensor against a fresh fixed model and require an exact numeric round trip.
5. Reconstruct each model in a fresh scoring process. Compare all node scores
   against that preparation run's stored reference scores using `rtol=1e-5,
   atol=1e-6`. Abort on any failed run, invalid artifact or failed comparison.
6. Freeze the resulting `reference/candidate/manifest.json`, model hashes,
   `reference/hashes.json`, evaluation assets and measured reference score before
   the research campaign. The manifest records the actual newly produced weights;
   there is no dependency on an unspecified previously trained checkpoint bundle.

The materialized baseline may vary slightly across eligible hardware or numeric
libraries. The frozen artifact and its measured score are established by this
preparation, not silently substituted for an earlier calibration. Within a task
campaign, baseline/candidate graph versions, inference specification and scoring
remain identical.

## Commands

Preparation requires Linux x86_64, Python 3.12, a CUDA-compatible driver, and one
A100 80GB or equivalent eligible GPU. Place this checkout and the destination on
persistent storage. CPU-only installation may run on a separate machine sharing
that storage.

```bash
cd rsi-reference
RSI_PYTHON_BIN=python3.12 bash scripts/setup_linux.sh
envs/torch222/bin/python scripts/verify_sources.py
envs/torch222/bin/python scripts/prepare_baseline.py \
  --destination "$PWD/prepared/reference-v1" --gpu 0
```

The installer pins PyTorch 2.2.2, PyG 2.5.3 and the listed scientific libraries.
`wheels.lock.json` gives the exact official Python 3.12/Linux CUDA 12.1 extension
wheel URLs, lengths and SHA256 digests. Full resolved package versions are saved
to `envs/torch222/requirements.resolved.txt` after installation.

The preparation driver invokes the existing label-separated data preparation and
sequential training suite, then the export and reload-parity tools. It uses only
physical GPU 0, with exactly one CUDA device visible to each model process.

## What to stage

The destination has these distinct roles:

- `operator-data/raw` and `operator-data/judge`: trusted preparation/Judge only;
  these contain labels and must never be copied into Work.
- `training-runs`: trusted preparation records and checkpoints; not a scoring
  input supplied by a candidate.
- `reference/candidate`: six numeric weight archives and their manifest.
- `reference/graphs`: fixed label-free feature/edge archives.
- `reference/calibration.json`, `reference/inference-verification.json`,
  `reference/hashes.json`, `reload-verification` and `preparation.json`: preparation
  evidence and commitments.

Normal Solution copies the prepared `reference/candidate` to
`WORKDIR/candidate`, verifying the task-owned committed hashes. It performs no
training or evaluation. Normal Judge reloads submitted arrays into its own fixed
model and computes node scores; it performs no training and does not trust stored
score vectors as rewards. `gad_rsi.inference` accepts no label path and never
loads a pickle or candidate Python. Training-checkpoint loading occurs only in
the explicitly operator-owned exporter.

Stage only the curated code and allowed reference files into the task image. Do
not copy the entire upstream repository into Work: upstream example data and
trusted preparation outputs can contain labels. Work and Judge run offline after
asset staging, with evaluator code/labels protected in a fresh Judge. Actual
container isolation still needs task-level validation.

## Calibration and validation status

`evidence/calibration.json` publishes sanitized receipts from six earlier A100
runs of this training recipe: Weibo AUROC mean 0.927853 and Reddit mean 0.568851,
with six-run mean 0.748352. Reference production took 5,975.11 seconds; maximum
training CUDA allocation was 5,493,093,376 bytes. These measurements provide
calibration context. Those private checkpoints are not required inputs to the
public preparation command and are not presented as downloadable assets here.

The earlier source/data/score/checkpoint receipts were checked and metrics
recomputed. The existing contract and numeric-archive tests pass (23 checks). Both graphs
have also been reconstructed from their pinned public downloads with identical
prepared graph/label hashes. Fixed-model construction and exact numeric state
round trips passed on both real graphs using CPU PyTorch 2.2.2; this is not a
trained-model scoring-parity result.
`evidence/validation.json` distinguishes completed checks from pending execution:
the newly combined preparation/export/reload path has not yet passed a full GPU
run, and no complete Harbor validation or strong-model research trajectory is
claimed. Task preparation must complete those checks and freeze its actual
baseline before research begins. Full Work + Judge + mandatory-overhead timing
remains to be measured; the 99.59-minute training figure is not a full loop time.

## Source and license

The integration follows the repository's Apache-2.0 license. Bundled ADA-GAD and
PyGOD source files are unmodified, identified individually in `sources.lock.json`,
and retain their upstream license files under `sources/`. The modern PyGOD code
is BSD-2-Clause. The self-contained source subset contains no dataset pickles or
pretrained model checkpoints.
