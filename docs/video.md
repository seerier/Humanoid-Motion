# Video front end

Turn a monocular video of a person into a Unitree G1 reference motion that the tracking side consumes exactly like a LAFAN1 clip:

```
video -> GVHMR (official demo, unmodified) -> SMPL-X (z up, grounded) -> GMR -> G1 reference (30 Hz) -> 50 Hz reference + URDF FK
```

The code lives in `src/humanoid_motion/video/` and the entry point is `scripts/video_to_reference.py`. GVHMR runs in its own Python environment; this repository only calls its demo script.

## Install

1. CPU environment (`scripts/setup_cpu.sh`), then the SMPL-X Python package. numpy and torch already come from the CPU environment. Install it again after any rerun of `setup_cpu.sh`, because that script syncs `.venv` to the lock file:
   ```bash
   uv pip install --python .venv/bin/python 'smplx==0.1.28'   # or: pip install -e '.[video]'
   ```
2. Body models. These are licensed per user, so register and download them yourself:
   - SMPL-X v1.1 from <https://smpl-x.is.tue.mpg.de>: put `SMPLX_NEUTRAL.npz` in `data/body_models/smplx/`. The adapter and the retargeting need it. To keep the models elsewhere, set `HUMANOID_MOTION_SMPLX_BODY_MODELS` to a directory that contains `smplx/`.
   - SMPL v1.1 from <https://smpl.is.tue.mpg.de>: put `SMPL_NEUTRAL.pkl` in `data/body_models/smpl/`. Only GVHMR's demo rendering uses it.
3. GVHMR (only needed for video input; an NVIDIA GPU is needed to run it):
   ```bash
   scripts/setup_gvhmr.sh               # GVHMR_ENV, GVHMR_SOURCE, GVHMR_WEIGHTS=hf|drive|none, GVHMR_CHECK_ONLY=1
   export GVHMR_PYTHON=<printed path>
   ```
   The script:
   - clones `zju3dv/GVHMR` at the commit pinned in `configs/upstream_pins.json` into `third_party/GVHMR`;
   - builds a conda environment following GVHMR's `docs/INSTALL.md`, with two documented departures: chumpy is installed without build isolation, and setuptools is pinned to 75.8.0 so that lightning finds `pkg_resources`;
   - links your body models into GVHMR's `inputs/checkpoints/body_models/`;
   - downloads the demo weights from the Hugging Face mirror `camenduru/GVHMR` (or, with `GVHMR_WEIGHTS=drive`, from the official Google Drive folder) and checks every file against the pinned SHA256 values. The mirror files are byte-identical to the official ones.

## Use

```bash
# From a video (re-encoded to 30 fps first unless it already is; static camera unless --moving-camera)
.venv/bin/python scripts/video_to_reference.py --video clip.mp4 --out outputs/video --manifest video
# From an existing GVHMR result
.venv/bin/python scripts/video_to_reference.py --gvhmr-results path/to/hmr4d_results.pt --out outputs/video --manifest video
# Import for tracking (Isaac), then evaluate with any policy, e.g. the released one
.venv-isaac/bin/python scripts/import_library.py --manifests video --data-root outputs/video --headless --device cuda:0
.venv-isaac/bin/python scripts/evaluate.py --library outputs/video/runtime/video.json --checkpoint humanoid_motion_g1_v0.1.pt --output outputs/eval/video --headless --device cuda:0
```

Each clip gets `<out>/<clip_id>/`:

| File | Contents |
|---|---|
| `human/smplx.npz` (+ `.json`) | the human, as an AMASS-style SMPL-X npz, plus the adapter's report (ground offset, betas) |
| `retarget/g1/` | GMR run (`motion.csv`, `reference.npz`, `metadata.json`, `diagnostics.json`) |
| `gate.json` | the quality gate |
| `prepared/` | the 50 Hz reference and independent URDF FK |
| `clip.json` | the clip's manifest row |

Every step is skipped when its output already exists. `--manifest NAME` writes `<out>/NAME.json` and `<out>/frozen.json` for `scripts/import_library.py`.

**Choices.**
- Gender is neutral, and one body shape (the mean of GVHMR's per-frame betas) is used for the whole clip.
- GVHMR's y-up world becomes z-up with GMR's own rotation matrix, applied at the parameter level.
- Ground (`--ground`):
  - `static_feet` (default): per frame, the foot that GVHMR's own stationary-joint head marks as planted defines the floor; frames in between interpolate, and the offset is low-passed at 3 Hz.
  - `feet_p05`: one constant offset, which puts the 5th percentile of the lowest foot vertex at z = 0.
- GMR runs unmodified, with three settings of ours:
  - the subject's mesh stature as the human height;
  - an IK floor constraint at z = 0;
  - up to 200 IK iterations on the first frame.

**Gate.** Four checks, each with pass/fail:
- G1 joints within the central 98 % of their range (≤ 1 % of frames may fail);
- G1 sole penetration of 2.5 cm or less (≤ 1 % of frames may fail);
- human pelvis height between 0.6 and 1.5 m;
- human root jerk below 50 m/s³ after a 3 Hz low-pass.

A failed gate only marks the clip as not usable in the manifest. The clip is never edited or dropped.

**Verification tools.**
- `python -m humanoid_motion.video.stage_audit --clip_dir <out>/<clip>` says which stage broke a bad clip: reconstruction (2-D reprojection), interface, retargeting (limb directions, joints pinned at limits) or ground.
- `humanoid_motion.video.reconcile` replays GMR's own GVHMR loader for comparison.
- `humanoid_motion.video.fk_check` recomputes an imported Isaac motion with an independent FK.

**Known limitation.** GMR's first-frame IK can settle in a poor local minimum on some clips (for example, a limb pointing the wrong way for the whole clip). The gate does not catch every such case; `stage_audit` flags it at the retargeting stage.

## Licences

Nothing below is redistributed by this repository.

- **GVHMR** (zju3dv/GVHMR): its licence permits use, copying, modification and distribution "for educational, research and non-profit purposes only. Any modification based on this work must be open-source and prohibited for commercial use." For commercial use, contact the authors. Its demo weights (GVHMR, HMR 2.0, ViTPose, YOLOv8, DPVO) come with their own licences, which you accept when you download them.
- **SMPL and SMPL-X body models**: Max Planck licences for non-commercial scientific research. Every user registers and downloads the models personally. They must not be shared, so `data/` is git-ignored.
- **smplx** (Python package): distributed under the same Max Planck non-commercial licence.
- **Videos you process**: their own terms apply to the input footage and to anything derived from it.

References and motions made from GVHMR output therefore inherit these non-commercial terms. This repository's own code is MIT-licensed.
