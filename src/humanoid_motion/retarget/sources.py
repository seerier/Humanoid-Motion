"""One audited place where a human recording becomes GMR source frames.

Two sources, one shape. `load_source(cfg, root)` returns a MotionSource holding the frames GMR consumes plus everything
humanoid_motion.retarget.gmr records about them, so the retargeting code contains no per-source branching beyond the
GMR key it passes to GeneralMotionRetargeting.

`third_party/GMR` is never modified (configs/upstream_pins.json pins it); everything here is our own layer around its
unmodified loaders.

Conventions, audited per source rather than assumed:

- bvh_lafan1: BVH header units are centimetres with Y up. GMR's load_bvh_file converts to metres Z-up via
  (x, -z, y); quaternions are wxyz. Frame time comes from the BVH header.
- smplx: AMASS-style SMPL-X npz (also what humanoid_motion.video.gvhmr_adapter writes) is already metres with Z up,
  because the upright rotation lives in each frame's `root_orient`, so no axis matrix is applied. Quaternions are wxyz
  (`Rotation.as_quat(scalar_first=True)` inside GMR's get_smplx_data_offline_fast). The SMPL-X *model's* own rest pose
  is Y-up: `measure_stature` therefore works along the body's own up direction and never assumes an axis. Frame rate
  comes from the file's `mocap_frame_rate`. `axis_up_check` in the audit is what actually establishes the data's up
  axis, per sequence.

Why this module builds the SMPL-X body model instead of calling GMR's load_smplx_file: that function fixes smplx's
defaults `num_betas=10` and `batch_size=1` while AMASS SMPL-X ships 16 betas and N frames, so
`torch.cat([betas, expression])` inside smplx gets (N, 16) and (1, 10). The field names it reads are the ones AMASS
uses; only the model construction needs widening, which is what load_amass_smplx does. Its height formula
(1.66 + 0.1 * betas[0]) is available as height mode "gmr_beta0".
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from humanoid_motion import ROOT
from humanoid_motion.geometry import source_fk

if str(ROOT / 'third_party/GMR') not in sys.path:   # the pinned GMR checkout, as humanoid_motion.retarget.gmr does
    sys.path.insert(0, str(ROOT / 'third_party/GMR'))

BVH_AXIS_MATRIX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])   # BVH (x, y, z) -> (x, -z, y)
BVH_CM_PER_M = 100.0
# SMPL-X body models are licensed per user (https://smpl-x.is.tue.mpg.de, register and accept the terms).
SMPLX_MODEL_DIR_SETTING = 'HUMANOID_MOTION_SMPLX_BODY_MODELS'
DEFAULT_SMPLX_MODEL_DIR = 'data/body_models'   # relative to the repository root; holds smplx/SMPLX_<GENDER>.npz
SMPLX_GENDERS = ('neutral', 'male', 'female')
DEFAULT_TARGET_FPS = 30.0   # LAFAN1's rate, so an SMPL-X reference is directly comparable
# The only fields read out of an AMASS file. GMR's get_smplx_data_offline_fast needs pose_body and
# mocap_frame_rate; the rest build the body. Everything else in the file is left unread.
AMASS_FIELDS = ('gender', 'betas', 'root_orient', 'pose_body', 'trans', 'mocap_frame_rate')


@dataclass
class MotionSource:
    """A sliced recording in GMR's input shape, plus what the retargeting run records about it."""
    kind: str                          # our config key: 'bvh_lafan1' | 'smplx'
    gmr_source: str                    # GMR src_human key (IK_CONFIG_DICT / GeneralMotionRetargeting)
    frames: list                       # per frame: {joint name: (position, quaternion)}, GMR's own input
    names: list[str]                   # source joint names, order of positions/quaternions
    parents: np.ndarray                # parent index per joint, -1 at the root
    positions: np.ndarray              # (frames, joints, 3) metres, Z up
    quaternions: np.ndarray            # (frames, joints, 4) wxyz
    frame_time: float                  # seconds between saved frames
    total_frames: int                  # frames the file offers, before slicing
    human_height_m: float              # what GMR scales the skeleton with
    stance_joints: tuple[str, str]     # left, right ankle-level joint for the stance proxy
    ik_foot_roles: tuple[str, str]     # left, right foot key in the GMR IK table
    independent_source_max_error_m: float
    conventions: dict                  # metadata strings: positions, quaternions, derivatives
    meta: dict                         # provenance for metadata.json['source']
    audit: dict = field(default_factory=dict)   # axis, unit and rigidity numbers for diagnostics.json

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(cfg: dict, root: Path) -> MotionSource:
    """Dispatch on cfg['source_kind'] (absent = 'bvh_lafan1', so every existing config keeps its meaning)."""
    kind = cfg.get('source_kind', 'bvh_lafan1')
    source = root / cfg['source_file']
    if kind == 'bvh_lafan1':
        return load_lafan1_bvh(source, cfg['frame_start'], cfg['frame_end_exclusive'], cfg['human_height_m'])
    if kind == 'smplx':
        return load_amass_smplx(source, cfg['frame_start'], cfg['frame_end_exclusive'],
                                target_fps=float(cfg.get('target_fps', DEFAULT_TARGET_FPS)),
                                height=cfg['human_height'], body_model_dir=cfg.get('smplx_body_model_dir'),
                                root=root)
    raise ValueError(f'source_kind {kind!r}; expected bvh_lafan1 or smplx')


def _check_slice(start: int, stop: int, available: int) -> None:
    if not (0 <= start < stop <= available and stop - start >= 3):
        raise ValueError(f'frame range [{start}, {stop}) does not fit {available} frames, or is shorter than 3')


# --------------------------------------------------------------------------------------------------
# LAFAN1 BVH
# --------------------------------------------------------------------------------------------------

def load_lafan1_bvh(source: Path, start: int, stop: int, human_height_m: float) -> MotionSource:
    """GMR's own LAFAN1 loader, plus the independent BVH forward kinematics check on the selected frames."""
    import re

    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

    frame_time = float(re.search(r'Frame Time:\s*([\d.eE+-]+)', source.read_text()).group(1))
    frames, upstream_height = load_bvh_file(str(source), format='lafan1')
    anim = read_bvh(str(source))
    _check_slice(start, stop, len(frames))
    frames = frames[start:stop]
    names = list(anim.bones)
    positions = np.array([[f[n][0] for n in names] for f in frames])
    quaternions = np.array([[f[n][1] for n in names] for f in frames])
    # Independent of GMR's loader: SciPy FK over the BVH local offsets, then the same axis and unit change.
    independent = source_fk(anim.pos[start:stop], anim.quats[start:stop], anim.parents)
    independent = independent @ BVH_AXIS_MATRIX.T / BVH_CM_PER_M
    return MotionSource(
        kind='bvh_lafan1', gmr_source='bvh_lafan1', frames=frames, names=names,
        parents=np.asarray(anim.parents), positions=positions, quaternions=quaternions,
        frame_time=frame_time, total_frames=len(anim.pos), human_height_m=float(human_height_m),
        stance_joints=('LeftFoot', 'RightFoot'), ik_foot_roles=('LeftFootMod', 'RightFootMod'),
        independent_source_max_error_m=float(np.max(np.abs(independent - positions))),
        conventions={
            'positions': 'BVH cm Y-up → metres Z-up via (x,-z,y); world positions, no trajectory recentering',
            'quaternions': 'wxyz',
            'derivatives': ('numpy.gradient at BVH frame_time, central interior / one-sided endpoints; '
                            'diagnostics only, tracker must resample and recompute'),
        },
        # Kept exactly as configured, so LAFAN1 metadata.json is byte-identical to the LAFAN1-only release.
        meta={'height_assumption_m': human_height_m, 'loader_hardcoded_height_m': upstream_height},
        audit=axis_audit(positions, names, root_name='Hips', head_name='Head'),
    )


# --------------------------------------------------------------------------------------------------
# AMASS-style SMPL-X
# --------------------------------------------------------------------------------------------------

def smplx_model_dir(configured: str | None = None, root: Path | None = None) -> Path:
    """Where the licensed SMPL-X body models live: the config value, then HUMANOID_MOTION_SMPLX_BODY_MODELS,
    then data/body_models under the repository (git-ignored, so the download never reaches the repository)."""
    root = ROOT if root is None else root
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else root / path
    value = os.environ.get(SMPLX_MODEL_DIR_SETTING)
    if value:
        return Path(value)
    return root / DEFAULT_SMPLX_MODEL_DIR


def require_smplx_model(model_dir: Path, gender: str) -> Path:
    """The model file smplx.create will open, or an error naming the download and the expected path."""
    candidates = [model_dir / 'smplx' / f'SMPLX_{gender.upper()}.{suffix}' for suffix in ('npz', 'pkl')]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f'No SMPL-X {gender} body model under {model_dir}/smplx/. Expected one of: '
        + ', '.join(p.name for p in candidates)
        + '. The models are licensed per user: register at https://smpl-x.is.tue.mpg.de, accept the terms, '
        'download "SMPL-X v1.1 (NPZ+PKL)" and unpack its models/smplx/ into that directory, or point '
        f'{SMPLX_MODEL_DIR_SETTING} at an existing copy (docs/video.md).')


def _gender(raw: Any) -> str:
    value = raw.item() if isinstance(raw, np.ndarray) else raw
    if isinstance(value, bytes):
        value = value.decode()
    value = str(value).strip().lower()
    if value not in SMPLX_GENDERS:
        raise ValueError(f'Unexpected SMPL-X gender {value!r}; expected one of {SMPLX_GENDERS}')
    return value


def resample_grid(num_frames: int, src_fps: float, tgt_fps: float) -> tuple[np.ndarray, float]:
    """The sample times GMR's get_smplx_data_offline_fast actually produces, and the true frame time.

    GMR takes frame_skip = int(src/tgt) and then np.linspace(0, N-1, N // frame_skip). Its own aligned_fps
    is kept / N * src, while the grid's actual rate is (kept - 1) / (N - 1) * src, so it runs fast by
    kept (N - 1) / (N (kept - 1)), about 1 / (kept - 1): 1.3 % for a 60-frame clip, 0.1 % for 1000 frames.
    Every velocity derived from the timestamps carries that error, so the exact frame time is recomputed
    here and the difference is recorded. Upsampling is refused rather than faked.
    """
    if tgt_fps > src_fps + 1e-9:
        raise ValueError(f'source is {src_fps} fps, below the {tgt_fps} fps target; GMR does not upsample '
                         '(it would silently keep every frame and relabel the rate). Lower target_fps.')
    if abs(tgt_fps - src_fps) <= 1e-9:
        return np.arange(num_frames, dtype=float), 1.0 / src_fps
    frame_skip = int(src_fps / tgt_fps)
    kept = num_frames // frame_skip
    if kept < 3:
        raise ValueError(f'{num_frames} frames at {src_fps} fps leave only {kept} frames at {tgt_fps} fps')
    times = np.linspace(0, num_frames - 1, kept)
    return times, float((num_frames - 1) / (kept - 1) / src_fps)


def measure_stature(vertices: np.ndarray, joints: np.ndarray, names: list[str]) -> dict:
    """Standing height of the shaped SMPL-X body, measured along the body's own up direction.

    The rest pose (zero body pose) of SMPL-X is Y-up, while AMASS motion data is Z-up, so no fixed axis is
    assumed: the up direction is the rest-pose head-above-ankles direction and the stature is the extent of
    the mesh along it. That is the anthropometric measure (crown to sole), independent of the file's frame.
    """
    ankles = [joints[names.index(n)] for n in ('left_ankle', 'right_ankle')]
    up = joints[names.index('head')] - np.mean(ankles, axis=0)
    norm = float(np.linalg.norm(up))
    if norm < 0.3:
        raise ValueError(f'rest-pose head is only {norm:.3f} m above the ankles; body model looks wrong')
    up = up / norm
    projection = (vertices * up).sum(axis=1)   # not vertices @ up: numpy 2.2 warns spuriously on (N, 3)
    return {'stature_m': float(projection.max() - projection.min()),
            'up_direction_rest_pose': up.round(6).tolist(),
            'bounding_box_extent_m': (vertices.max(axis=0) - vertices.min(axis=0)).round(6).tolist()}


def human_height(spec: dict, betas: np.ndarray, rest_vertices: np.ndarray, rest_joints: np.ndarray,
                 names: list[str]) -> dict:
    """GMR's actual_human_height, by the mode the config names. No default: the choice changes leg scaling
    on every sequence, so it is the config's to make.

    - fixed:        value_m, as LAFAN1 configs do (one height for a whole recording set)
    - gmr_beta0:    1.66 + 0.1 * betas[0], the formula inside GMR's own load_smplx_file
    - mesh_stature: crown-to-sole height of this subject's shaped mesh (measure_stature)

    rest_vertices / rest_joints are the shaped, unposed body (smplx's v_shaped and the joints regressed
    from it), so no second forward pass is needed.
    """
    mode = spec['mode']
    detail = {'mode': mode}
    if mode == 'fixed':
        detail['value_m'] = float(spec['value_m'])
    elif mode == 'gmr_beta0':
        detail['value_m'] = float(1.66 + 0.1 * betas.reshape(-1)[0])
        detail['formula'] = '1.66 + 0.1 * betas[0] (general_motion_retargeting.utils.smpl.load_smplx_file)'
    elif mode == 'mesh_stature':
        measured = measure_stature(rest_vertices, rest_joints, names)
        detail.update(measured, value_m=measured['stature_m'],
                      gmr_beta0_for_comparison_m=float(1.66 + 0.1 * betas.reshape(-1)[0]))
    else:
        raise ValueError(f"human_height mode {mode!r}; expected fixed, gmr_beta0 or mesh_stature")
    return detail


def load_amass_smplx(source: Path, start: int, stop: int, target_fps: float, height: dict,
                     body_model_dir: str | None = None, root: Path | None = None) -> MotionSource:
    """An AMASS-style SMPL-X npz as GMR source frames at target_fps.

    GMR's get_smplx_data_offline_fast does the joint-orientation composition and the resampling unmodified;
    this function builds the body model at the file's own beta width and frame count, recomputes the true
    frame time, and audits axes, units and skeleton rigidity.
    """
    import smplx
    import torch
    from smplx.joint_names import JOINT_NAMES
    from smplx.lbs import vertices2joints
    from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast

    with np.load(source, allow_pickle=False) as archive:
        available = list(archive.files)
        missing = [key for key in AMASS_FIELDS if key not in available]
        if missing:
            raise ValueError(f'{source.name} has no {missing}; expected an AMASS-style SMPL-X npz '
                             f'(found {sorted(available)})')
        # Only the fields this pipeline uses. Reading every field would pull in markers_latent_vids, which
        # real AMASS files store as an object array and numpy will not load without allow_pickle; nothing
        # downloaded is ever unpickled here.
        raw = {key: archive[key] for key in AMASS_FIELDS}
        for key in ('surface_model_type', 'mocap_time_length'):
            if key in available:
                raw[key] = archive[key]
    gender, betas = _gender(raw['gender']), np.asarray(raw['betas']).reshape(-1)
    src_fps = float(np.asarray(raw['mocap_frame_rate']).reshape(-1)[0])
    num_frames = int(raw['pose_body'].shape[0])
    if raw['pose_body'].shape[1] != 63:
        raise ValueError(f'pose_body is {raw["pose_body"].shape}, expected (frames, 63) for 21 body joints')
    if raw['root_orient'].shape != (num_frames, 3) or raw['trans'].shape != (num_frames, 3):
        raise ValueError('root_orient and trans must both be (frames, 3)')

    model_dir = smplx_model_dir(body_model_dir, root)
    model_file = require_smplx_model(model_dir, gender)
    body_model = smplx.create(str(model_dir), 'smplx', gender=gender, use_pca=False,
                              num_betas=len(betas), batch_size=num_frames)
    with torch.no_grad():
        output = body_model(
            betas=torch.as_tensor(betas).float().view(1, -1),
            global_orient=torch.as_tensor(raw['root_orient']).float(),
            body_pose=torch.as_tensor(raw['pose_body']).float(),
            transl=torch.as_tensor(raw['trans']).float(),
            left_hand_pose=torch.zeros(num_frames, 45), right_hand_pose=torch.zeros(num_frames, 45),
            jaw_pose=torch.zeros(num_frames, 3), leye_pose=torch.zeros(num_frames, 3),
            reye_pose=torch.zeros(num_frames, 3),
            expression=torch.zeros(num_frames, body_model.num_expression_coeffs),
            # No posed vertices: this layer reads joints, full_pose and the shaped rest body only, and a
            # (frames, 10475, 3) tensor is hundreds of MB on a long sequence.
            return_full_pose=True, return_verts=False)

    names = list(JOINT_NAMES[: len(body_model.parents)])
    parents = body_model.parents.detach().cpu().numpy().astype(int).copy()
    parents[0] = -1                     # smplx marks the root with -1 already; make it explicit for source_fk
    # The shaped, unposed body: rest-pose bone offsets for the FK check and the mesh for the stature.
    rest_vertices = output.v_shaped.detach().numpy()[0]
    with torch.no_grad():
        rest_joints = vertices2joints(body_model.J_regressor, output.v_shaped[:1]).numpy()[0, : len(names)]
    # Independent of GMR's orientation chain: FK over the rest-pose bone offsets and the raw local rotations.
    raw_error = _smplx_fk_error(output, names, parents, rest_joints)

    frames, gmr_aligned_fps = get_smplx_data_offline_fast(raw, body_model, output, tgt_fps=target_fps)
    times, frame_time = resample_grid(num_frames, src_fps, target_fps)
    if len(frames) != len(times):
        raise ValueError(f'GMR returned {len(frames)} frames, the recomputed grid has {len(times)}; '
                         'the resampling in get_smplx_data_offline_fast changed')
    _check_slice(start, stop, len(frames))
    frames = frames[start:stop]
    positions = np.array([[f[n][0] for n in names] for f in frames])
    quaternions = np.array([[f[n][1] for n in names] for f in frames])

    audit = axis_audit(positions, names, root_name='pelvis', head_name='head')
    audit['smplx_model_fk_independent_max_error_m'] = raw_error
    audit['resampled_frame_rigidity'] = _bone_length_drift(positions, parents)
    height_detail = human_height(height, betas, rest_vertices, rest_joints, names)
    return MotionSource(
        kind='smplx', gmr_source='smplx', frames=frames, names=names, parents=parents,
        positions=positions, quaternions=quaternions, frame_time=frame_time, total_frames=len(times),
        human_height_m=float(height_detail['value_m']),
        stance_joints=('left_ankle', 'right_ankle'), ik_foot_roles=('left_foot', 'right_foot'),
        independent_source_max_error_m=raw_error,
        conventions={
            'positions': ('SMPL-X metres Z-up as published (the upright rotation is inside each frame\'s '
                          'root_orient); no axis matrix and no unit change applied, no trajectory recentering'),
            'quaternions': 'wxyz',
            'derivatives': ('numpy.gradient at the resampled frame_time, central interior / one-sided '
                            'endpoints; diagnostics only, tracker must resample and recompute'),
        },
        meta={
            'format': 'AMASS SMPL-X (_stageii npz)',
            'gender': gender, 'num_betas': int(len(betas)), 'betas': betas.round(6).tolist(),
            'body_model_file': str(model_file), 'body_model_sha256': sha256(model_file),
            'source_fps': src_fps, 'target_fps': target_fps,
            'resampling': ('general_motion_retargeting.utils.smpl.get_smplx_data_offline_fast: SLERP on '
                           'rotations, linear on joint positions, over np.linspace(0, N-1, N // '
                           'int(src/tgt)); saved frames are interpolations, not raw mocap samples'),
            'frame_time_s_exact': frame_time,
            'gmr_aligned_fps_reported': float(gmr_aligned_fps),
            'gmr_aligned_fps_error': float(gmr_aligned_fps - 1.0 / frame_time),
            'height_assumption_m': float(height_detail['value_m']), 'height': height_detail,
        },
        audit=audit,
    )


def _smplx_fk_error(output, names: list[str], parents: np.ndarray, rest: np.ndarray) -> float:
    """Max distance between the model's own joints and FK over rest-pose offsets with the raw local poses.

    Checks the piece GMR composes by hand (parent chain, rotation order, quaternion layout) on the frames as
    published, before any resampling. Rotations are the local axis-angle poses; offsets are the shaped
    rest-pose bone vectors, so a wrong axis or unit anywhere shows up as metres of error.
    """
    from scipy.spatial.transform import Rotation

    joints = output.joints.detach().numpy()[:, : len(names)]
    local = output.full_pose.detach().numpy().reshape(len(joints), -1, 3)[:, : len(names)]
    quaternions = np.stack([Rotation.from_rotvec(local[:, j]).as_quat(scalar_first=True)
                            for j in range(len(names))], axis=1)
    offsets = np.broadcast_to(rest - rest[parents], joints.shape).copy()
    offsets[:, 0] = joints[:, 0]         # the root carries the world position, as BVH local offsets do
    return float(np.max(np.abs(source_fk(offsets, quaternions, parents) - joints)))


def _bone_length_drift(positions: np.ndarray, parents: np.ndarray) -> dict:
    """How far parent-child distances move over the clip; a rigid skeleton keeps them constant."""
    children = [j for j, p in enumerate(parents) if p >= 0]
    lengths = np.linalg.norm(positions[:, children] - positions[:, parents[children]], axis=-1)
    spread = lengths.max(axis=0) - lengths.min(axis=0)
    return {'max_bone_length_change_m': float(spread.max()),
            'mean_bone_length_change_m': float(spread.mean()),
            'note': 'resampling interpolates positions and rotations separately, so this is not exactly zero'}


def axis_audit(positions: np.ndarray, names: list[str], root_name: str, head_name: str) -> dict:
    """Which way is up and what the units are, measured from the frames themselves.

    A source read with the wrong axis convention or in centimetres still produces finite,
    plausible-looking arrays, so every run records these numbers. Two independent signals, because
    neither covers every clip:

    - posture: a standing person's head is 0.3-1.0 m above the pelvis along the up axis. Decisive when
      the subject is upright, silent when they are lying down.
    - floor: whatever the body does, its lowest point stays near the ground, so the up axis is the one
      whose per-frame lowest joint varies least over the clip. Decisive when the subject travels, silent
      when they barely move, since then no axis varies much.

    The verdict is deliberately three-valued. A nearly motionless recording of someone lying on the floor
    carries no evidence of gravity's direction at all, and calling that "wrong axis" would be a false
    alarm on perfectly good data (CMU 111_21 is exactly this case). The convention itself is a property
    of the source, fixed in load_amass_smplx / load_lafan1_bvh; this is a check on it, not its origin.
    """
    root, head = positions[:, names.index(root_name)], positions[:, names.index(head_name)]
    offset = (head - root).mean(axis=0)
    upright = bool(0.3 < offset[2] < 1.0 and np.argmax(np.abs(offset)) == 2)
    floor_variation = positions.min(axis=1).std(axis=0)
    order = np.argsort(floor_variation)
    floor_axis = int(order[0])
    best, runner_up = floor_variation[order[0]], floor_variation[order[1]]
    # Two axes equally still means the clip does not discriminate, not an infinitely confident answer.
    still = 1e-6
    margin = float('inf') if best <= still < runner_up else (float(runner_up / best) if best > 0 else 1.0)
    floor_says_z = bool(floor_axis == 2 and margin >= 2.0)
    floor_says_other = bool(floor_axis != 2 and margin >= 2.0)
    units_ok = bool(float(np.abs(positions).max()) < 20.0 and abs(offset[2]) < 1.5)
    if floor_says_other or (upright is False and floor_says_z is False and abs(offset[2]) > 1.0):
        verdict = 'conflicts_with_z_up'
    elif upright or floor_says_z:
        verdict = 'z_up_confirmed'
    else:
        verdict = 'undetermined'
    return {
        'verdict': verdict, 'units_plausible_metres': units_ok,
        'root_joint': root_name, 'head_joint': head_name,
        'mean_head_minus_root_m': offset.round(6).tolist(),
        'head_above_root_m': float(offset[2]), 'subject_upright': upright,
        'floor_pinned_axis_index': floor_axis, 'floor_margin_over_next_axis': round(margin, 3),
        'floor_axis_decisive': bool(margin >= 2.0),
        'lowest_point_std_by_axis_m': floor_variation.round(6).tolist(),
        'joint_extent_m': (positions.reshape(-1, 3).max(axis=0)
                           - positions.reshape(-1, 3).min(axis=0)).round(6).tolist(),
        'rule': ('z_up_confirmed when the subject is upright (head 0.3-1.0 m above the root in z) or the '
                 'floor-pinned axis is z by at least 2x; conflicts_with_z_up when the floor pins another '
                 'axis that clearly; undetermined otherwise, e.g. someone lying still, where the clip '
                 'carries no evidence either way. units_plausible_metres bounds the magnitudes, which a '
                 'centimetre read (x100) would fail.'),
    }
