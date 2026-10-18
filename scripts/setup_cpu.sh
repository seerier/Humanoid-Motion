#!/usr/bin/env bash
# CPU environment for retargeting, the viewers, video rendering and tests (macOS arm64 or Linux).
# Creates .venv (Python 3.10), checks out the pinned upstream sources and downloads LAFAN1.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PWD/.local/python}"
mkdir -p .local .cache third_party data outputs
if command -v uv >/dev/null; then
  uv_bin="$(command -v uv)"
else
  if [[ ! -x .local/bootstrap/bin/uv ]]; then
    python3 -m venv .local/bootstrap
    .local/bootstrap/bin/python -m pip install uv==0.12.20
  fi
  uv_bin="$PWD/.local/bootstrap/bin/uv"
fi
"$uv_bin" python install --no-bin 3.10.19
[[ -x .venv/bin/python ]] || "$uv_bin" venv --python 3.10.19 .venv
"$uv_bin" pip sync --python .venv/bin/python requirements/cpu.lock.txt
"$uv_bin" pip install --python .venv/bin/python --no-deps -e .

pin() { .venv/bin/python -c "import json,sys; print(json.load(open('configs/upstream_pins.json'))[sys.argv[1]][sys.argv[2]])" "$1" "$2"; }
clone_pinned() {
  local name="$1" dir="third_party/$1" commit
  commit="$(pin "$name" commit)"
  if [[ ! -d "$dir/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone "$(pin "$name" url)" "$dir"
    git -C "$dir" checkout --detach "$commit"
  fi
  [[ "$(git -C "$dir" rev-parse HEAD)" == "$commit" ]] || { echo "Unexpected commit in $dir" >&2; exit 1; }
  [[ -z "$(git -C "$dir" status --porcelain --untracked-files=no)" ]] || { echo "Tracked modifications in $dir" >&2; exit 1; }
}
clone_pinned GMR
clone_pinned whole_body_tracking
clone_pinned lafan1

if [[ ! -f data/lafan1.zip ]]; then
  curl -fL --retry 2 --connect-timeout 20 \
    "https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/$(pin lafan1 commit)/lafan1/lafan1.zip" \
    -o data/lafan1.zip
fi
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib, json, zipfile
expected = json.loads(Path('configs/upstream_pins.json').read_text())['lafan1']['archive_sha256']
archive = Path('data/lafan1.zip')
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise ValueError('LAFAN1 archive checksum mismatch')
with zipfile.ZipFile(archive) as z:
    content = z.read('walk1_subject1.bvh')
dest = Path('data/walk1_subject1.bvh')
if dest.exists() and dest.read_bytes() != content:
    raise ValueError('Existing BVH differs; refusing to overwrite')
dest.write_bytes(content)
PY
.venv/bin/python scripts/fetch_robot_asset.py
echo 'Ready. Try: .venv/bin/python scripts/retarget_clip.py && .venv/bin/python scripts/view_motion.py'
