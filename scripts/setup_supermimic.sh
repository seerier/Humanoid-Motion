#!/usr/bin/env bash
# Optional SuperMimic backend (docs/supermimic.md): MorphMimic, the open SuperMimic implementation, at the commit pinned
# in configs/upstream_pins.json, with its own locked environment (Python 3.11, mjlab, MuJoCo Warp, torch cu128) in
# third_party/morphmimic/.venv. Linux, NVIDIA GPU with a CUDA 12 driver; about 9 GB of disk. The checkout is used
# unmodified; this repository's modules run with its interpreter and PYTHONPATH=src.
#   bash scripts/setup_supermimic.sh            # clone and install
#   bash scripts/setup_supermimic.sh --check    # verify an existing install only
set -euo pipefail
cd "$(dirname "$0")/.."
pin() { python3 -c "import json,sys; print(json.load(open('configs/upstream_pins.json'))[sys.argv[1]][sys.argv[2]])" "$1" "$2"; }
dir=third_party/morphmimic
commit="$(pin MorphMimic commit)"
if [[ "${1:-}" != --check ]]; then
  command -v uv >/dev/null || { echo 'Install uv first (https://docs.astral.sh/uv/)' >&2; exit 1; }
  if [[ ! -d $dir/.git ]]; then
    git clone "$(pin MorphMimic url)" "$dir"          # no --recursive: its submodules are only for teleop and hardware
    git -C "$dir" checkout --detach "$commit"
  fi
  (cd "$dir" && for i in 1 2 3; do UV_HTTP_TIMEOUT=300 uv sync --frozen --no-dev --group export && break; sleep 20; done)
fi
[[ "$(git -C "$dir" rev-parse HEAD)" == "$commit" ]] || { echo "Unexpected commit in $dir" >&2; exit 1; }
[[ -z "$(git -C "$dir" status --porcelain --untracked-files=no)" ]] || { echo "Tracked modifications in $dir" >&2; exit 1; }
MUJOCO_GL=disable PYTHONPATH=src "$dir/.venv/bin/python" - <<'PY'
import importlib.metadata as m
import mjlab, morphmimic, onnxruntime, torch, warp  # noqa: F401
import humanoid_motion.tracking.supermimic.library  # noqa: F401
print({p: m.version(p) for p in ('mjlab', 'mujoco', 'warp-lang', 'torch', 'onnxruntime')})
PY
echo "MorphMimic $commit ready in $dir"
