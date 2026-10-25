"""GVHMR demo output (`hmr4d_results.pt`) -> the AMASS-style SMPL-X npz that sources.load_amass_smplx reads.

What GVHMR gives (tools/demo/demo.py, key `smpl_params_global`): per-frame SMPL-X body parameters in a gravity-aligned
world frame with **y up** (GVHMR's "ay" frame): `body_pose` (F, 63), `betas` (F, 10), `global_orient` (F, 3, axis-angle),
`transl` (F, 3); the demo processes every video at 30 fps. What the entry wants (AMASS convention, z up): `gender`,
`betas` (16,), `root_orient`, `pose_body`, `trans`, `mocap_frame_rate`.

Fixed choices: gender neutral, betas padded 10 -> 16 with zeros, mocap_frame_rate 30.
The y-up -> z-up rotation matrix is GMR's (utils/smpl.py get_gvhmr_data_offline_fast); GMR applies it to joints after
forward kinematics, this adapter applies it at the parameter level so the file is an ordinary AMASS-style file.
Parameter-level rotation needs one term GMR does not: SMPL-X rotates the body about its pelvis joint J0 (betas-dependent,
not at the origin), so for a world rotation R
    root_orient' = R * root_orient,     trans' = R trans + (R - I) J0.
`convert` checks this by running the body model on both parameter sets (joints' == R joints to numerical precision).

Ground mode `feet_p05` (a constant offset): the 5th percentile over frames of the lowest foot vertex is moved to z = 0;
foot vertices are the rest-pose vertices within FOOT_BAND_M of the rest mesh's lowest point. Off with --ground none
(the reconciliation configuration). The offset and foot statistics go into the sidecar json.
Ground mode `static_feet` (per-frame; the production choice of scripts/video_to_reference.py): GVHMR's own
stationary-joint head (`net_outputs.static_conf_logits`, joints L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist)
says per frame which foot is planted; its post-processing (hmr4d/model/gvhmr/utils/postprocess.py, pp_static_joint_cam)
uses sigmoid > 0.8 and corrects only x/z ("do not modify y"). Here the same labels drive the vertical axis: in every
frame with a planted foot, that foot's lowest vertex defines the ground (offset = its height); frames without a planted
foot interpolate linearly between the neighbouring planted frames; the offset curve is low-passed (2nd-order
Butterworth, STATIC_FEET_CUTOFF_HZ, zero phase) and subtracted from trans. A clip with no planted frame at all is left
unmodified and flagged in the sidecar (`static_feet_applied: false`).
Betas: GVHMR predicts betas per frame; GMR's loader takes frame 0. Default here is the per-frame mean (one body for
the clip); --betas first reproduces GMR's choice for the reconciliation run. Recorded in the sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

Y_UP_TO_Z_UP = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])   # GMR utils/smpl.py, get_gvhmr_data_offline_fast
GVHMR_FPS = 30.0            # tools/demo/demo.py writes and reads the working video at 30 fps
NUM_BETAS_OUT = 16          # AMASS SMPL-X width that sources.load_amass_smplx was written for
FOOT_BAND_M = 0.08          # rest-pose vertices this close to the mesh's lowest point count as foot vertices
GROUND_PERCENTILE = 5.0
GROUND_MODES = ('feet_p05', 'static_feet', 'none')
BETAS_MODES = ('mean', 'first')
GVHMR_KEYS = ('body_pose', 'betas', 'global_orient', 'transl')
STATIC_JOINTS = ('L_Ankle', 'L_foot', 'R_Ankle', 'R_foot', 'L_wrist', 'R_wrist')   # GVHMR static_conf head order
STATIC_PROB_THRESHOLD = 0.8      # GVHMR pp_static_joint_cam: static_label = sigmoid(logits) > 0.8
STATIC_FEET_CUTOFF_HZ = 3.0      # low-pass of the per-frame ground offset (same cutoff as the gate's jerk filter)


def load_gvhmr_static_conf(path: Path) -> np.ndarray:
    """GVHMR's per-frame stationary-joint logits, (F, 6) float64, from the demo result file's `net_outputs`."""
    import torch
    pred = torch.load(path, map_location='cpu', weights_only=False)
    try:
        logits = pred['net_outputs']['static_conf_logits']
    except (KeyError, TypeError) as error:
        raise ValueError(f'{path.name}: no net_outputs.static_conf_logits (GVHMR demo result with network outputs needed)') from error
    logits = np.asarray(logits.detach().cpu().numpy() if hasattr(logits, 'detach') else logits, dtype=np.float64)
    logits = logits.reshape(-1, logits.shape[-1])
    if logits.shape[1] != len(STATIC_JOINTS) or not np.isfinite(logits).all():
        raise ValueError(f'{path.name}: static_conf_logits is {logits.shape}, expected (F, {len(STATIC_JOINTS)}) finite')
    return logits


def static_feet_offset(lowest_left: np.ndarray, lowest_right: np.ndarray, planted_left: np.ndarray,
                       planted_right: np.ndarray, fps: float, cutoff_hz: float | None = STATIC_FEET_CUTOFF_HZ):
    """Per-frame ground offset from planted feet.

    lowest_*: (F,) height of each foot's lowest vertex; planted_*: (F,) bool from GVHMR's stationary labels.
    Frames with a planted foot take that foot's height (the lower one if both are planted); other frames interpolate
    linearly between the neighbouring planted frames (held constant before the first / after the last); the curve is
    then low-passed (cutoff_hz None = no filtering, used by the unit tests). Returns (offset (F,), info dict) or
    (None, info) when no frame has a planted foot.
    """
    lowest_left, lowest_right = np.asarray(lowest_left, float), np.asarray(lowest_right, float)
    planted_left, planted_right = np.asarray(planted_left, bool), np.asarray(planted_right, bool)
    frames = lowest_left.shape[0]
    candidate = np.full((frames, 2), np.inf)
    candidate[planted_left, 0] = lowest_left[planted_left]
    candidate[planted_right, 1] = lowest_right[planted_right]
    planted = planted_left | planted_right
    info = {'frames': int(frames), 'planted_frames': int(planted.sum()),
            'planted_left_frames': int(planted_left.sum()), 'planted_right_frames': int(planted_right.sum())}
    if not planted.any():
        return None, info
    raw = np.interp(np.arange(frames), np.flatnonzero(planted), candidate[planted].min(axis=1))
    offset = raw
    if cutoff_hz is not None and frames > 12:
        from scipy.signal import butter, filtfilt
        b, a = butter(2, cutoff_hz / (fps / 2))
        offset = filtfilt(b, a, raw, padlen=min(3 * max(len(a), len(b)), frames - 1))
    info.update({'cutoff_hz': cutoff_hz, 'offset_m': {'min': float(offset.min()), 'max': float(offset.max()),
                 'mean': float(offset.mean())}, 'lowpass_max_change_m': float(np.abs(offset - raw).max())})
    return offset, info


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gvhmr_global_params(path: Path) -> dict:
    """`smpl_params_global` of a GVHMR demo result file as float64 numpy arrays, shapes checked."""
    import torch
    try:
        pred = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:   # the file also holds `net_outputs`; only load result files you produced yourself
        pred = torch.load(path, map_location='cpu', weights_only=False)
    params = pred['smpl_params_global']
    out = {k: np.asarray(params[k].detach().cpu().numpy() if hasattr(params[k], 'detach') else params[k], dtype=np.float64)
           for k in GVHMR_KEYS}
    frames = out['body_pose'].shape[0]
    expected = {'body_pose': (frames, 63), 'betas': (frames, 10), 'global_orient': (frames, 3), 'transl': (frames, 3)}
    for key, shape in expected.items():
        if out[key].shape != shape:
            raise ValueError(f'{path.name}: smpl_params_global[{key!r}] is {out[key].shape}, expected {shape}')
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError(f'{path.name}: non-finite SMPL-X parameters')
    return out


def _body_model(model_dir: Path, num_frames: int):
    import smplx
    from humanoid_motion.retarget.sources import require_smplx_model
    require_smplx_model(model_dir, 'neutral')
    return smplx.create(str(model_dir), 'smplx', gender='neutral', use_pca=False, num_betas=NUM_BETAS_OUT,
                        batch_size=num_frames)


def _forward(model, betas16, root_orient, pose_body, trans, verts: bool):
    import torch
    n = pose_body.shape[0]
    with torch.no_grad():
        return model(betas=torch.as_tensor(betas16).float().view(1, -1).expand(n, -1),
                     global_orient=torch.as_tensor(root_orient).float(), body_pose=torch.as_tensor(pose_body).float(),
                     transl=torch.as_tensor(trans).float(),
                     left_hand_pose=torch.zeros(n, 45), right_hand_pose=torch.zeros(n, 45), jaw_pose=torch.zeros(n, 3),
                     leye_pose=torch.zeros(n, 3), reye_pose=torch.zeros(n, 3),
                     expression=torch.zeros(n, model.num_expression_coeffs), return_verts=verts, return_full_pose=False)


def rotate_params(root_orient: np.ndarray, trans: np.ndarray, pelvis_rest: np.ndarray, rotation: np.ndarray):
    """World rotation applied at the parameter level: root_orient' = R * root_orient, trans' = R t + (R - I) J0."""
    from scipy.spatial.transform import Rotation
    world = Rotation.from_matrix(rotation)
    new_orient = (world * Rotation.from_rotvec(root_orient)).as_rotvec()
    new_trans = trans @ rotation.T + (rotation - np.eye(3)) @ pelvis_rest
    return new_orient, new_trans


def convert(params: dict, model_dir: Path, ground: str = 'feet_p05', betas_mode: str = 'mean',
            rotation: np.ndarray = Y_UP_TO_Z_UP, static_conf: np.ndarray | None = None) -> tuple[dict, dict]:
    """GVHMR global parameters -> (AMASS-style arrays, sidecar report). Runs the SMPL-X body model twice.

    static_conf: GVHMR's (F, 6) stationary-joint logits, required for ground='static_feet'."""
    if ground not in GROUND_MODES or betas_mode not in BETAS_MODES:
        raise ValueError(f'ground {ground!r} (one of {GROUND_MODES}), betas {betas_mode!r} (one of {BETAS_MODES})')
    frames = params['body_pose'].shape[0]
    if ground == 'static_feet':
        if static_conf is None or static_conf.shape != (frames, len(STATIC_JOINTS)):
            raise ValueError(f'ground static_feet needs static_conf of shape ({frames}, {len(STATIC_JOINTS)})')
    betas10 = params['betas'].mean(axis=0) if betas_mode == 'mean' else params['betas'][0]
    betas16 = np.concatenate([betas10, np.zeros(NUM_BETAS_OUT - 10)])
    model = _body_model(model_dir, frames)
    # Original (y-up) forward pass: joints for the equivalence check, rest pelvis for the translation term.
    original = _forward(model, betas16, params['global_orient'], params['body_pose'], params['transl'], verts=False)
    joints_y_up = original.joints.numpy()[:, :22].astype(np.float64)
    # smplx sizes its landmark buffers by the model's batch_size, so the one-frame rest pose needs its own model.
    rest = _forward(_body_model(model_dir, 1), betas16, np.zeros((1, 3)), np.zeros((1, 63)), np.zeros((1, 3)), verts=True)
    pelvis_rest = rest.joints.numpy()[0, 0].astype(np.float64)
    rest_vertices = rest.vertices.numpy()[0].astype(np.float64)
    root_orient, trans = rotate_params(params['global_orient'], params['transl'], pelvis_rest, rotation)
    rotated = _forward(model, betas16, root_orient, params['body_pose'], trans, verts=ground != 'none')
    joints_z_up = rotated.joints.numpy()[:, :22].astype(np.float64)
    equivalence = float(np.abs(joints_z_up - joints_y_up @ rotation.T).max())
    if equivalence > 1e-4:
        raise AssertionError(f'parameter-level rotation disagrees with joint-level rotation by {equivalence:.2e} m')
    report = {
        'frames': int(frames), 'fps': GVHMR_FPS, 'gender': 'neutral',
        'betas_mode': betas_mode, 'betas10': betas10.round(6).tolist(),
        'betas_per_frame_std_max': float(params['betas'].std(axis=0).max()),
        'rotation_y_up_to_z_up': rotation.tolist(), 'pelvis_rest_m': pelvis_rest.round(6).tolist(),
        'rotation_equivalence_max_error_m': equivalence,
        'ground': {'mode': ground},
    }
    if ground != 'none':
        # Rest pose is y-up: the lowest rest vertices are the soles. The clip's foot heights are read after rotation.
        rest_up = rest_vertices[:, 1]
        foot = np.flatnonzero(rest_up <= rest_up.min() + FOOT_BAND_M)
        foot_z = rotated.vertices.numpy()[:, foot, 2].astype(np.float64)   # (F, n_foot)
        lowest = foot_z.min(axis=1)
        report['ground'].update({'foot_vertex_band_m': FOOT_BAND_M, 'foot_vertices': int(foot.size)})
        if ground == 'feet_p05':
            offset = float(np.percentile(lowest, GROUND_PERCENTILE))
            report['ground'].update({'percentile': GROUND_PERCENTILE, 'offset_removed_m': offset})
        else:
            # SMPL-X rest pose: the body's left is +x. Planted = GVHMR's stationary label on the ankle or the foot joint.
            left = foot[rest_vertices[foot, 0] > 0]
            right = foot[rest_vertices[foot, 0] <= 0]
            prob = 1 / (1 + np.exp(-static_conf))
            planted_left = (prob[:, :2] > STATIC_PROB_THRESHOLD).any(axis=1)
            planted_right = (prob[:, 2:4] > STATIC_PROB_THRESHOLD).any(axis=1)
            verts_z = rotated.vertices.numpy()[:, :, 2].astype(np.float64)
            curve, info = static_feet_offset(verts_z[:, left].min(axis=1), verts_z[:, right].min(axis=1),
                                             planted_left, planted_right, GVHMR_FPS)
            report['ground'].update({'static_prob_threshold': STATIC_PROB_THRESHOLD, 'static_joints': list(STATIC_JOINTS),
                                     'left_foot_vertices': int(left.size), 'right_foot_vertices': int(right.size), **info,
                                     'static_feet_applied': curve is not None})
            offset = np.zeros(frames) if curve is None else curve
            report['ground']['offset_per_frame_m'] = [round(float(v), 5) for v in offset]
        trans = trans.copy()
        trans[:, 2] -= offset
        lowest = lowest - offset
        report['ground'].update({
            'lowest_foot_vertex_after_m': {'min': float(lowest.min()), 'p05': float(np.percentile(lowest, 5)),
                                           'median': float(np.median(lowest)), 'max': float(lowest.max())},
            'frames_below_minus_2_5cm': int((lowest < -0.025).sum()), 'frames_above_plus_5cm': int((lowest > 0.05).sum()),
            'frames_above_plus_10cm': int((lowest > 0.10).sum()),
        })
        joints_z_up = joints_z_up.copy()
        joints_z_up[:, :, 2] -= np.asarray(offset).reshape(-1, 1) if np.ndim(offset) else offset
    pelvis_z = joints_z_up[:, 0, 2]
    report['pelvis_height_m'] = {'min': float(pelvis_z.min()), 'max': float(pelvis_z.max()), 'mean': float(pelvis_z.mean())}
    arrays = {
        'gender': np.array('neutral'), 'betas': betas16, 'root_orient': root_orient,
        'pose_body': params['body_pose'].copy(), 'trans': trans, 'mocap_frame_rate': np.array(GVHMR_FPS),
        'surface_model_type': np.array('smplx'), 'mocap_time_length': np.array((frames - 1) / GVHMR_FPS),
    }
    return arrays, report


def write(arrays: dict, report: dict, out_npz: Path) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **arrays)
    report = dict(report, npz=str(out_npz), npz_sha256=sha256(out_npz))
    out_npz.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')


def main(argv=None) -> None:
    from humanoid_motion import ROOT
    from humanoid_motion.retarget.sources import smplx_model_dir
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gvhmr_results', type=Path, required=True, help='hmr4d_results.pt from tools/demo/demo.py')
    parser.add_argument('--out', type=Path, required=True, help='output npz (sidecar json next to it)')
    parser.add_argument('--ground', choices=GROUND_MODES, default='feet_p05')
    parser.add_argument('--betas', choices=BETAS_MODES, default='mean')
    parser.add_argument('--smplx_body_model_dir', type=Path, default=None)
    args = parser.parse_args(argv)
    params = load_gvhmr_global_params(args.gvhmr_results)
    static_conf = load_gvhmr_static_conf(args.gvhmr_results) if args.ground == 'static_feet' else None
    model_dir = smplx_model_dir(str(args.smplx_body_model_dir) if args.smplx_body_model_dir else None, ROOT)
    arrays, report = convert(params, model_dir, ground=args.ground, betas_mode=args.betas, static_conf=static_conf)
    report = dict(report, source={'path': str(args.gvhmr_results), 'sha256': sha256(args.gvhmr_results)},
                  body_model_dir=str(model_dir))
    write(arrays, report, args.out)
    print(json.dumps({k: report[k] for k in ('frames', 'betas_mode', 'rotation_equivalence_max_error_m', 'ground', 'pelvis_height_m')}, indent=2))


if __name__ == '__main__':
    main()
