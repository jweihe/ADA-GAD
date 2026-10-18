#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_CACHE_DIR="$PWD/cache/pip"
export TORCH_HOME="$PWD/cache/torch"
export XDG_CACHE_HOME="$PWD/cache/xdg"
mkdir -p "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" envs
"${RSI_PYTHON_BIN:-python3.12}" -m venv envs/torch222
envs/torch222/bin/python -m pip install pip==24.3.1
# Linux x86_64 PyPI wheels for this release use CUDA 12.1.
envs/torch222/bin/python -m pip install torch==2.2.2
envs/torch222/bin/python -m pip install \
    numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2 PyYAML==6.0.2 \
    torch-geometric==2.5.3 networkx==3.2.1 tensorboardX==2.6.2.2 \
    requests==2.32.3 tqdm==4.67.1 pytest==8.3.5
envs/torch222/bin/python scripts/fetch_wheels.py
envs/torch222/bin/python -m pip install --no-index --find-links cache/wheels \
    torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3
envs/torch222/bin/python -m pip freeze > envs/torch222/requirements.resolved.txt
