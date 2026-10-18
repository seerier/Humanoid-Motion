"""Validate a retargeted CSV and prepare the 50 Hz reference plus independent URDF forward kinematics.

The output is the tracker-independent reference; the tracker's own motion file is written later by
humanoid_motion.tracking.isaac_export, which reads these files.
"""
import json
from pathlib import Path

import numpy as np

from humanoid_motion.contracts import URDF_SHA256, load_run, sha256
from humanoid_motion.urdf_kinematics import UrdfKinematics


def prepare(run: Path, out: Path, urdf: Path) -> dict:
    if sha256(urdf) != URDF_SHA256:
        raise ValueError('Unexpected tracking URDF hash')
    data, meta = load_run(run)
    robot = UrdfKinematics(urdf)
    names = data['joint_names'].tolist()
    if robot.root != 'pelvis' or set(robot.joint_names) != set(names):
        raise ValueError('Unsupported robot')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True, exist_ok=True)
    arrays = {key: [] for key in ('body_pos_w', 'body_quat_w', 'body_link_lin_vel_w',
                                 'body_com_lin_vel_w', 'body_ang_vel_w')}
    for i in range(len(data['timestamps'])):
        state = robot.evaluate(data['root_pos_w'][i], data['root_quat_w'][i],
            dict(zip(names, data['joint_pos'][i])), data['root_link_lin_vel_w'][i],
            data['root_ang_vel_w'][i], dict(zip(names, data['joint_vel'][i])))
        for key, field in [('body_pos_w', 'p'), ('body_link_lin_vel_w', 'v'),
                           ('body_com_lin_vel_w', 'com_v'), ('body_ang_vel_w', 'w')]:
            arrays[key].append([state[name][field] for name in robot.links])
        arrays['body_quat_w'].append([state[name]['r'].as_quat()[[3, 0, 1, 2]] for name in robot.links])
    arrays = {key: np.array(value) for key, value in arrays.items()}
    hard, soft = {}, {}
    for j in robot.joints:
        if j['kind'] == 'fixed':
            continue
        limits = np.array(j['limits'])
        soft_limits = limits.mean() + (limits - limits.mean()) * 0.9
        angles = data['joint_pos'][:, names.index(j['name'])]
        hard[j['name']] = int(np.sum((angles < limits[0]) | (angles > limits[1])))
        soft[j['name']] = int(np.sum((angles < soft_limits[0]) | (angles > soft_limits[1])))
    correction = arrays['body_com_lin_vel_w'] - arrays['body_link_lin_vel_w']
    source_count = len(np.loadtxt(run / 'motion.csv', delimiter=','))
    np.savez_compressed(out / 'generalized_reference.npz', **data)
    np.savez_compressed(out / 'independent_urdf.npz', **arrays, body_names=np.array(robot.links))
    report = dict(schema='humanoid_motion.generalized_reference.v1',
        evidence='resampled kinematic reference and independent URDF FK; no Isaac or physics execution',
        source_csv_sha256=sha256(run / 'motion.csv'), source_metadata_sha256=sha256(run / 'metadata.json'),
        tracking_urdf_sha256=sha256(urdf), upstream_commits=meta['upstream_commits'],
        joint_names=names, independent_body_names=robot.links,
        frames=dict(input=source_count, output=len(data['timestamps'])),
        timing=dict(input_fps=meta['csv']['fps'], output_fps=50,
            input_last_timestamp_s=(source_count - 1) / meta['csv']['fps'],
            output_last_timestamp_s=float(data['timestamps'][-1]),
            endpoint_policy='[0, (N-1)/input_fps); final input timestamp excluded; no time stretching',
            linear_interpolation='piecewise linear root position and joint angles; shortest-arc root SLERP',
            derivative='central position/joint differences, one-sided endpoints; world SO(3) central log with repeated endpoints'),
        coordinates=dict(world='SI metres, radians, seconds; right-handed Z-up; no alignment or recentering',
            root='pelvis link origin', anchor='torso_link', quaternion='wxyz',
            root_linear_velocity='pelvis link origin in world',
            tracker_body_linear_velocity='COM in world; must use runtime COM offsets after fixed-link merging'),
        urdf_hard_limit_violating_joint_frames=hard,
        predicted_soft_limit_violating_joint_frames=soft,
        root_com_velocity_correction_max_m_s=float(np.linalg.norm(correction[:, robot.links.index('pelvis')], axis=-1).max()),
        all_body_com_velocity_correction_max_m_s=float(np.linalg.norm(correction, axis=-1).max()),
        max_joint_speed_rad_s=float(np.abs(data['joint_vel']).max()),
        files={name: sha256(out / name) for name in ('generalized_reference.npz', 'independent_urdf.npz')})
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report

