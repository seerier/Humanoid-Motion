"""Isaac kinematic export of one prepared reference into the BeyondMimic motion format, with an independent audit.

Poses the robot frame by frame (no physics steps) and checks every body pose and velocity against the
independent URDF forward kinematics from humanoid_motion.retarget.reference. Needs Isaac Sim 4.5 / Isaac Lab 2.1
on an NVIDIA GPU, imported after the Isaac AppLauncher.
"""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np

from humanoid_motion.contracts import TRACKING_COMMIT, URDF_SHA256, named_indices, sha256, validate_tracker


def export(args: argparse.Namespace, data: dict, independent: dict, started: float, resources: dict, runtime: tuple | None = None) -> tuple:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationContext
    from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG, G1_ACTION_SCALE
    from whole_body_tracking.robots import g1
    from scipy.spatial.transform import Rotation
    import re

    expected_source = args.tracking_root / 'source/whole_body_tracking/whole_body_tracking/robots/g1.py'
    if Path(g1.__file__).resolve() != expected_source.resolve():
        raise ValueError('Imported tracker does not come from the pinned checkout')
    if runtime is None:
        sim = SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=.02))
        cfg = G1_CYLINDER_CFG.replace(prim_path='/World/Robot')
        cfg.spawn.asset_path = str(args.urdf.resolve())
        # The URDF converter otherwise defaults to a temporary directory outside project control.
        cfg.spawn.usd_dir = str(Path(getattr(args, 'usd_dir', args.output / 'usd')).resolve())
        cfg.spawn.usd_file_name = 'g1.usd'
        robot = Articulation(cfg)
        sim.reset()
        robot.update(.02)
    else:
        sim, robot = runtime
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - started
    joint_names, body_names = robot.joint_names, robot.body_names
    order = named_indices(data['joint_names'].tolist(), joint_names)
    if body_names[0] != 'pelvis' or 'torso_link' not in body_names:
        raise ValueError('Unexpected root/anchor body import')
    independent_names = independent['body_names'].tolist()
    if len(set(body_names)) != len(body_names) or not set(body_names) <= set(independent_names):
        raise ValueError('Unexpected imported body names')
    body_ids = [independent_names.index(name) for name in body_names]
    to_tensor = lambda array: torch.as_tensor(array, dtype=torch.float32, device=sim.device)
    to_numpy = lambda tensor: tensor.detach().cpu().numpy().copy()
    controls = {name: to_numpy(getattr(robot.data, name)[0]).tolist() for name in
        ('joint_pos_limits', 'soft_joint_pos_limits', 'joint_vel_limits', 'joint_effort_limits',
         'joint_stiffness', 'joint_damping', 'joint_armature', 'default_joint_pos')}
    controls['action_scale'] = []
    for name in joint_names:
        matches = [value for pattern, value in G1_ACTION_SCALE.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous action scale for {name}')
        controls['action_scale'].append(matches[0])
    controls['action_offset'] = 'default_joint_pos; policy action is NOT a reference residual'
    com_local = to_numpy(robot.data.com_pos_b[0])
    logs = {key: [] for key in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
                              'body_lin_vel_w', 'body_ang_vel_w')}
    root_write_errors = []
    convert_start = time.perf_counter()
    for i in range(len(data['timestamps'])):
        root = np.concatenate([data['root_pos_w'][i], data['root_quat_w'][i],
                               data['root_link_lin_vel_w'][i], data['root_ang_vel_w'][i]])
        # Input linear velocity is of the pelvis origin. This API converts it to COM velocity.
        robot.write_root_link_state_to_sim(to_tensor(root[None]))
        robot.write_joint_state_to_sim(to_tensor(data['joint_pos'][i, order][None]),
                                      to_tensor(data['joint_vel'][i, order][None]))
        sim.render()
        robot.update(.02)
        for key in logs:
            logs[key].append(to_numpy(getattr(robot.data, key)[0]))
        expected_com = data['root_link_lin_vel_w'][i] + np.cross(data['root_ang_vel_w'][i],
            Rotation.from_quat(data['root_quat_w'][i, [1, 2, 3, 0]]).apply(com_local[0]))
        root_write_errors.append(np.linalg.norm(logs['body_lin_vel_w'][-1][0] - expected_com))
    torch.cuda.synchronize()
    resources['initialization_seconds'] = init_seconds
    resources['frames'] = len(data['timestamps'])
    resources['conversion_seconds'] = time.perf_counter() - convert_start
    resources['frames_per_second'] = len(data['timestamps']) / resources['conversion_seconds']
    logs = {key: np.array(value) for key, value in logs.items()}
    logs['fps'] = data['fps']
    # Runtime COM offsets can differ from URDF offsets when fixed bodies are merged.
    reference_q = independent['body_quat_w'][:, body_ids]
    rotated_com = Rotation.from_quat(reference_q.reshape(-1, 4)[:, [1, 2, 3, 0]]).apply(
        np.broadcast_to(com_local, (len(reference_q), *com_local.shape)).reshape(-1, 3)).reshape(-1, len(body_names), 3)
    expected_com_v = independent['body_link_lin_vel_w'][:, body_ids] + np.cross(
        independent['body_ang_vel_w'][:, body_ids], rotated_com)
    position_error = np.linalg.norm(logs['body_pos_w'] - independent['body_pos_w'][:, body_ids], axis=-1)
    actual_r = Rotation.from_quat(logs['body_quat_w'].reshape(-1, 4)[:, [1, 2, 3, 0]])
    expected_r = Rotation.from_quat(reference_q.reshape(-1, 4)[:, [1, 2, 3, 0]])
    errors = dict(position_m=float(position_error.max()), orientation_rad=float((actual_r * expected_r.inv()).magnitude().max()),
        com_linear_velocity_m_s=float(np.linalg.norm(logs['body_lin_vel_w'] - expected_com_v, axis=-1).max()),
        angular_velocity_rad_s=float(np.linalg.norm(logs['body_ang_vel_w'] - independent['body_ang_vel_w'][:, body_ids], axis=-1).max()),
        root_com_write_m_s=float(max(root_write_errors)),
        joint_position_rad=float(np.abs(logs['joint_pos'] - data['joint_pos'][:, order]).max()),
        joint_velocity_rad_s=float(np.abs(logs['joint_vel'] - data['joint_vel'][:, order]).max()))
    thresholds = dict(position_m=1e-4, orientation_rad=1e-4, com_linear_velocity_m_s=1e-3,
                      angular_velocity_rad_s=1e-3, root_com_write_m_s=1e-4, joint_position_rad=1e-5, joint_velocity_rad_s=1e-5)
    passed = all(errors[key] <= thresholds[key] for key in errors)
    output = args.output / 'motion.npz'
    np.savez_compressed(output, **logs)
    from isaaclab.assets.articulation import articulation_data, articulation
    sidecar = dict(schema='humanoid_motion.isaac_tracking.v1', evidence='Isaac kinematic preprocessing; no physics steps',
        audit_passed=passed, joint_names=joint_names, body_names=body_names,
        root_body='pelvis', anchor_body='torso_link', fps=50,
        coordinates=dict(position='world link origins, metres', quaternion='world wxyz',
                         linear_velocity='world COM, m/s', angular_velocity='world rad/s'),
        tracking_commit=TRACKING_COMMIT, urdf_sha256=URDF_SHA256, motion_sha256=sha256(output),
        prepared_manifest_sha256=sha256(args.prepared / 'manifest.json'),
        runtime_com_offsets_b_m=com_local.tolist(), control=controls, audit_max_errors=errors,
        audit_thresholds=thresholds, body_position_max_error_by_name=dict(zip(body_names, position_error.max(axis=0).tolist())),
        environment={dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()},
        isaac_source_hashes={str(Path(module.__file__).name): sha256(Path(module.__file__)) for module in (articulation_data, articulation)},
        adapter_sha256=sha256(Path(__file__)))
    validate_tracker(logs, sidecar, joint_names, body_names, .02)
    (args.output / 'motion.manifest.json').write_text(json.dumps(sidecar, indent=2) + '\n')
    print(json.dumps({'audit_passed': passed, 'errors': errors, 'output': str(output)}, indent=2))
    if not passed:
        raise RuntimeError('Runtime audit failed; artifacts retained, training must not proceed')
    return sim, robot

