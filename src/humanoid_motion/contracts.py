"""Named, SI, Z-up tracking interfaces; no simulator imports or inferred orders."""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


CSV_JOINTS = [f'{s}_{j}_joint' for s in ('left', 'right')
              for j in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
CSV_JOINTS += [f'waist_{j}_joint' for j in ('yaw', 'roll', 'pitch')]
CSV_JOINTS += [f'{s}_{j}_joint' for s in ('left', 'right')
               for j in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                         'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
CSV_COLUMNS = ['root_x', 'root_y', 'root_z', 'qx', 'qy', 'qz', 'qw'] + CSV_JOINTS
TRACKING_COMMIT = 'cd65172032893724b445448818c34165846d847d'
URDF_SHA256 = 'dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def named_indices(source: list[str], target: list[str]) -> list[int]:
    """Permutation into target order; reject missing, extra and duplicate joints."""
    if len(set(source)) != len(source) or len(set(target)) != len(target) or set(source) != set(target):
        raise ValueError('Named joint sets must match uniquely')
    return [source.index(name) for name in target]


def check_quaternions(q: np.ndarray) -> None:
    if not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-6, rtol=0):
        raise ValueError('Expected finite unit quaternions; refusing silent normalization')


def angular_velocity_world(q_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """SO(3) central derivative, repeated endpoint values as in pinned converter.

    Left relative rotations express omega in world coordinates. Shortest-arc
    rotation logs make quaternion sign flips irrelevant.
    """
    check_quaternions(q_wxyz)
    if len(q_wxyz) < 3 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Need >=3 samples and positive finite dt')
    rotation = Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]])
    interior = (rotation[2:] * rotation[:-2].inv()).as_rotvec() / (2 * dt)
    return np.concatenate([interior[:1], interior, interior[-1:]])


def resample_csv(csv: np.ndarray, columns: list[str], input_fps: float,
                 output_fps: float = 50.0) -> dict[str, np.ndarray]:
    if columns != CSV_COLUMNS:
        raise ValueError('CSV column manifest differs from pinned named xyzw contract')
    if csv.ndim != 2 or csv.shape[1] != len(columns) or len(csv) < 3 or not np.isfinite(csv).all():
        raise ValueError('Expected >=3 finite CSV rows with exactly 36 columns')
    if not all(np.isfinite(f) and f > 0 for f in (input_fps, output_fps)):
        raise ValueError('FPS must be positive and finite')
    check_quaternions(csv[:, 3:7])
    source_t = np.arange(len(csv), dtype=np.float64) / input_fps
    # Preserve upstream [0, duration) sampling; do not stretch to a grid endpoint.
    times = np.arange(0.0, source_t[-1], 1.0 / output_fps)
    if len(times) < 3:
        raise ValueError('Resampled clip needs >=3 samples for angular differences')
    linear = np.stack([np.interp(times, source_t, csv[:, i]) for i in [0, 1, 2] + list(range(7, 36))], axis=1)
    quat = Slerp(source_t, Rotation.from_quat(csv[:, 3:7]))(times).as_quat()[:, [3, 0, 1, 2]]
    velocity = np.gradient(linear, 1 / output_fps, axis=0)
    return dict(timestamps=times, fps=np.array([output_fps]), joint_names=np.array(CSV_JOINTS),
                root_pos_w=linear[:, :3], root_quat_w=quat, joint_pos=linear[:, 3:],
                root_link_lin_vel_w=velocity[:, :3], root_ang_vel_w=angular_velocity_world(quat, 1 / output_fps),
                joint_vel=velocity[:, 3:])


def load_run(run: Path) -> tuple[dict[str, np.ndarray], dict]:
    metadata = json.loads((run / 'metadata.json').read_text())
    if sha256(run / 'motion.csv') != metadata['outputs']['motion.csv']:
        raise ValueError('CSV hash does not match source run')
    if metadata['upstream_commits']['whole_body_tracking'] != TRACKING_COMMIT:
        raise ValueError('Unexpected tracking revision')
    if metadata['model']['joint_units'] != 'radians' or not metadata['model']['root'].startswith('pelvis free joint'):
        raise ValueError('Unsupported root/joint unit contract')
    named_indices(metadata['model']['joint_names'], CSV_JOINTS)
    data = resample_csv(np.loadtxt(run / 'motion.csv', delimiter=','), metadata['csv']['columns'], metadata['csv']['fps'])
    return data, metadata


def validate_tracker(data: dict, sidecar: dict, joint_names: list[str], body_names: list[str],
                     control_dt: float) -> None:
    """The upstream reader indexes without names: mismatched order MUST fail."""
    if sidecar['joint_names'] != joint_names or sidecar['body_names'] != body_names:
        raise ValueError('Runtime joint/body order differs from motion sidecar')
    if len(set(joint_names)) != len(joint_names) or len(set(body_names)) != len(body_names):
        raise ValueError('Duplicate runtime names')
    fps = np.asarray(data['fps']).reshape(-1)
    if fps.shape != (1,) or not np.isfinite(fps[0]) or fps[0] <= 0 or not np.isclose(fps[0] * control_dt, 1, atol=1e-8, rtol=0):
        raise ValueError('Motion FPS and control frequency differ')
    count = len(data['joint_pos'])
    if count < 3:
        raise ValueError('Clip is too short')
    shapes = {'joint_pos': (count, len(joint_names)), 'joint_vel': (count, len(joint_names))}
    shapes.update({k: (count, len(body_names), d) for k, d in
                   [('body_pos_w', 3), ('body_quat_w', 4), ('body_lin_vel_w', 3), ('body_ang_vel_w', 3)]})
    for name, shape in shapes.items():
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(f'Invalid {name}: expected finite {shape}')
    check_quaternions(data['body_quat_w'])
