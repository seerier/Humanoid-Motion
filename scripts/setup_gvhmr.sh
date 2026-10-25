#!/usr/bin/env bash
# Optional video front end: the official GVHMR (zju3dv/GVHMR at the commit pinned in configs/upstream_pins.json) in its
# own conda environment, as its docs/INSTALL.md describes, with the demo weights checked against pinned SHA256 values.
# Nothing is vendored or modified. GVHMR's licence allows education, research and non-profit use only and requires
# derivatives to be open source (docs/video.md). Running the demo needs an NVIDIA GPU (CUDA 12.1 wheels).
#   scripts/setup_gvhmr.sh
# Settings (environment variables):
#   GVHMR_ENV      conda environment prefix (default: <conda base>/envs/gvhmr)
#   GVHMR_SOURCE   where to clone from (default: the pinned GitHub URL; a local clone works too)
#   GVHMR_WEIGHTS  hf (default: Hugging Face mirror) | drive (official Google Drive, often rate-limited) | none
#   GVHMR_CHECK_ONLY=1  verify an existing installation (commit, environment, body models, weights); install nothing
# SMPL and SMPL-X body models are licensed per user and never downloaded here. Put SMPLX_NEUTRAL.npz (SMPL-X v1.1) in
# data/body_models/smplx/ and SMPL_NEUTRAL.pkl (SMPL v1.1) in data/body_models/smpl/ first; they are linked into GVHMR.
set -euo pipefail
cd "$(dirname "$0")/.."
repo=$PWD
pins=configs/upstream_pins.json
pin() { python3 -c "import json,sys; print(json.load(open('$pins'))['GVHMR'][sys.argv[1]])" "$1"; }
weight() { python3 -c "import json,sys; print(json.load(open('$pins'))['GVHMR'][sys.argv[1]][sys.argv[2]])" "$1" "$2"; }
weight_files="$(python3 -c "import json; print('\n'.join(sorted(json.load(open('$pins'))['GVHMR']['weights_sha256'])))")"

# 1. Pinned checkout, unmodified.
dir=third_party/GVHMR
commit="$(pin commit)"
check_only="${GVHMR_CHECK_ONLY:-0}"
fail() { echo "$1" >&2; exit 1; }
if [[ ! -d $dir/.git ]]; then
  [[ $check_only == 1 ]] && fail "No GVHMR checkout at $dir"
  git clone "${GVHMR_SOURCE:-$(pin url)}" "$dir"
  git -C "$dir" checkout --detach "$commit"
  git -C "$dir" remote set-url origin "$(pin url)"
fi
[[ "$(git -C "$dir" rev-parse HEAD)" == "$commit" ]] || { echo "Unexpected commit in $dir" >&2; exit 1; }
[[ -z "$(git -C "$dir" status --porcelain --untracked-files=no)" ]] || { echo "Tracked modifications in $dir" >&2; exit 1; }

# 2. Conda environment exactly as GVHMR's docs/INSTALL.md (python 3.10, pip install -r requirements.txt, pip install -e .)
#    plus ffmpeg for the 30 fps re-encode in scripts/video_to_reference.py.
conda_bin="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${GVHMR_ENV:-}" ]]; then
  [[ -n $conda_bin ]] || { echo "conda not found; install Miniconda or set GVHMR_ENV to an existing environment" >&2; exit 1; }
  GVHMR_ENV="$("$conda_bin" info --base)/envs/gvhmr"
fi
py=$GVHMR_ENV/bin/python
if [[ ! -x $py ]]; then
  [[ $check_only == 1 ]] && fail "No environment at $GVHMR_ENV"
  [[ -n $conda_bin ]] || { echo "conda not found; cannot create $GVHMR_ENV" >&2; exit 1; }
  "$conda_bin" create -y -p "$GVHMR_ENV" --override-channels -c conda-forge python=3.10 ffmpeg
fi
if ! "$py" -c 'import torch, pytorch3d, hmr4d' 2>/dev/null; then
  [[ $check_only == 1 ]] && fail "$GVHMR_ENV cannot import torch, pytorch3d, hmr4d"
  (cd "$dir"
   # chumpy's setup.py imports pip, which pip's isolated build environment does not contain (unchanged on PyPI);
   # installing it without build isolation first is the only departure from "pip install -r requirements.txt".
   "$py" -m pip install "numpy==1.23.5" "setuptools>=68.0" six scipy
   "$py" -m pip install --no-build-isolation chumpy
   "$py" -m pip install -r requirements.txt
   "$py" -m pip install -e .)
fi
# lightning 2.3.0 imports pkg_resources at import time; setuptools >= 81 no longer ships it.
if ! "$py" -c 'import pkg_resources' 2>/dev/null; then
  [[ $check_only == 1 ]] && fail "$GVHMR_ENV has no pkg_resources (needs setuptools<81)"
  "$py" -m pip install "setuptools==75.8.0"
fi

# 3. Body models (yours, licensed per user) linked into GVHMR's expected layout.
ck=$dir/inputs/checkpoints
mkdir -p "$ck/body_models/smplx" "$ck/body_models/smpl"
for f in data/body_models/smplx/SMPLX_*.npz data/body_models/smpl/SMPL_*.pkl; do
  [[ -e $f ]] || continue
  dest="$ck/body_models/$(basename "$(dirname "$f")")/$(basename "$f")"
  [[ -e $dest || -L $dest || $check_only == 1 ]] || ln -s "$repo/$f" "$dest"
done
for need in smplx/SMPLX_NEUTRAL.npz smpl/SMPL_NEUTRAL.pkl; do
  [[ -e $ck/body_models/$need ]] || echo "WARNING: missing $ck/body_models/$need (see docs/video.md, body models)" >&2
done

# 4. Demo weights: download what is missing, then verify every file against the pinned SHA256.
mode="${GVHMR_WEIGHTS:-hf}"
while read -r f; do
  [[ -s $ck/$f || $mode == none || $check_only == 1 ]] && continue
  mkdir -p "$(dirname "$ck/$f")"
  if [[ $mode == hf ]]; then
    curl -fL --retry 5 --retry-delay 20 -C - -o "$ck/$f.part" "$(pin weights_mirror)/$f"
  elif [[ $mode == drive ]]; then
    "$py" -m pip install gdown
    "$py" -m gdown "$(weight weights_drive_ids "$f")" -O "$ck/$f.part"
  else
    echo "GVHMR_WEIGHTS must be hf, drive or none" >&2; exit 1
  fi
  mv "$ck/$f.part" "$ck/$f"
done <<< "$weight_files"
python3 - "$ck" "$pins" <<'PY'
import hashlib, json, sys
from pathlib import Path
checkpoints, expected = Path(sys.argv[1]), json.load(open(sys.argv[2]))['GVHMR']['weights_sha256']
bad = []
for name, digest in sorted(expected.items()):
    path, h = checkpoints / name, hashlib.sha256()
    if not path.is_file():
        bad.append(name + ' (missing)')
        continue
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            h.update(block)
    print(('ok       ' if h.hexdigest() == digest else 'MISMATCH ') + name)
    if h.hexdigest() != digest:
        bad.append(name)
if bad:
    sys.exit('GVHMR weights failed verification: ' + ', '.join(bad))
PY
echo "GVHMR ready at $dir ($commit). For scripts/video_to_reference.py: export GVHMR_PYTHON=$py"
