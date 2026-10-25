"""SuperMimic (MorphMimic) policies in MorphMimic's CPU MuJoCo sim2sim: the per-clip player.

`play` runs one exported clip as their deploy.run plays it (sim_node.SimRobot and policy_node.PolicyRunner in one
process instead of two linked by ZMQ): no actuator delay, no noise, no randomization, deterministic policy, motion
interface, simulated to the clip end. The evaluation harness, scoring and output files are in
humanoid_motion.tracking.sim2sim (command line: scripts/evaluate_mujoco.py --policy supermimic).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from humanoid_motion.tracking.sim2sim import mm_state


PREFIX = 'robot/'


def tracked_reference(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tracked-body positions [T, 14, 3] and wxyz quaternions [T, 14, 4] of MorphMimic reference states [T, 240]."""
    return states[:, 58:100].reshape(-1, 14, 3), states[:, 100:156].reshape(-1, 14, 4)


def play(deploy: Path, xml: Path, clip: Path, body_names: list[str], joint_names: list[str]) -> dict:
    """Play one exported clip; return the recorded states (sim world frame) and the clip's reference states."""
    import mujoco
    from deploy.sim2sim.policy_node import ClipReference, PolicyRunner
    from deploy.sim2sim.sim_node import SimRobot
    ref = np.load(clip)
    states = ref['states'].astype(np.float64)
    runner = PolicyRunner(deploy, 'g1', ClipReference(states), 'motion')
    sim = SimRobot(xml, runner.robot)
    sim.reset(ref['init_qpos'], ref['init_qvel'])
    m, d = sim.model, sim.data
    bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + n) for n in body_names]
    joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + n) for n in joint_names]
    if min(bodies) < 0 or min(joints) < 0:
        raise ValueError(f'{xml}: missing library bodies or joints')
    qadr, dadr = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    out = {k: [] for k in ('body_pos_w', 'body_quat_w', 'joint_pos', 'joint_vel', 'actions', 'mm_state')}
    robot = runner.robot

    def capture():
        out['body_pos_w'].append(d.xpos[bodies].copy())
        out['body_quat_w'].append(d.xquat[bodies].copy())
        out['joint_pos'].append(d.qpos[qadr].copy())
        out['joint_vel'].append(d.qvel[dadr].copy())
        out['mm_state'].append(mm_state(m, d, sim, robot))

    capture()
    for step in range(len(states) - 1):          # sim_node.run with delay 0: targets held for `decimation` steps
        quat, gyro, q, dq = sim.measure()
        targets = runner.step(step, quat, gyro, q, dq)
        for _ in range(runner.robot.decimation):
            sim.set_targets(targets)
            sim.step()
        mujoco.mj_forward(m, d)
        out['actions'].append(runner.last_action.copy())
        capture()
    arrays = {k: np.asarray(v) for k, v in out.items()}
    arrays['states'] = states
    arrays['tracked_bodies'] = list(robot.tracked_bodies)
    return arrays
