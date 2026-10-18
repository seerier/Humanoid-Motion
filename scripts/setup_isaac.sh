#!/usr/bin/env bash
# GPU environment for import, training and evaluation: Linux x86_64, NVIDIA driver, Python 3.10.
# Isaac Sim 4.5.0 + Isaac Lab v2.1.0 + BeyondMimic (whole_body_tracking) + rsl_rl, in .venv-isaac.
# Run scripts/setup_cpu.sh first (upstream checkouts, LAFAN1, robot description). Needs roughly 30 GB of disk.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvidia-smi >/dev/null || { echo 'No NVIDIA driver found' >&2; exit 1; }
[[ -d third_party/whole_body_tracking/.git ]] || { echo 'Run scripts/setup_cpu.sh first' >&2; exit 1; }
if command -v uv >/dev/null; then
  uv_bin="$(command -v uv)"
else
  [[ -x .local/bootstrap/bin/uv ]] || { python3 -m venv .local/bootstrap && .local/bootstrap/bin/python -m pip install uv==0.12.20; }
  uv_bin="$PWD/.local/bootstrap/bin/uv"
fi
[[ -x .venv-isaac/bin/python ]] || "$uv_bin" venv --python 3.10 .venv-isaac
"$uv_bin" pip install --python .venv-isaac/bin/python pip==25.1.1 setuptools==80.9.0 wheel==0.45.1 toml==0.10.2

isaaclab_commit="$(python3 -c "import json; print(json.load(open('configs/upstream_pins.json'))['IsaacLab']['commit'])")"
[[ -d third_party/IsaacLab/.git ]] || git clone --depth 1 --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
[[ "$(git -C third_party/IsaacLab rev-parse HEAD)" == "$isaaclab_commit" ]] || { echo 'Unexpected Isaac Lab commit' >&2; exit 1; }

"$uv_bin" pip install --python .venv-isaac/bin/python \
  torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
"$uv_bin" pip install --python .venv-isaac/bin/python \
  'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
"$uv_bin" pip install --python .venv-isaac/bin/python --no-build-isolation \
  --exclude-newer 2025-06-20 \
  -e third_party/IsaacLab/source/isaaclab \
  -e third_party/IsaacLab/source/isaaclab_assets \
  -e third_party/IsaacLab/source/isaaclab_tasks \
  -e 'third_party/IsaacLab/source/isaaclab_rl[rsl_rl]' \
  -e third_party/whole_body_tracking/source/whole_body_tracking \
  numpy==1.26.4 scipy==1.15.3 pytest==8.3.5 tensorboard==2.19.0
"$uv_bin" pip install --python .venv-isaac/bin/python --no-deps -e .
"$uv_bin" pip check --python .venv-isaac/bin/python

# Robot meshes for the simulator, staged inside the pinned tracker's asset tree.
.venv-isaac/bin/python scripts/fetch_robot_asset.py --runtime
.venv-isaac/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())'
