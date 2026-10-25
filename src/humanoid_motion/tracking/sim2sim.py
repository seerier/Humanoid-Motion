"""The BeyondMimic policy (observation v2, scripts/train.py checkpoints) in CPU MuJoCo, outside Isaac.

It runs on the scene the SuperMimic backend is evaluated on (MorphMimic's export_assets G1 XML: the same PD gains,
torque limits, action scale, default pose and 5 ms x 4 control timing as the Isaac G1, floor friction 1.0), through
MorphMimic's sim2sim robot (deploy.sim2sim.sim_node.SimRobot), so both tracking backends can be compared in one
physics engine. The observation is rebuilt from the simulator state exactly as the Isaac terms compute it
(humanoid_motion.tracking.observations.PolicyV2): the future reference window, then 10-step histories of projected
gravity, pelvis angular velocity, joint positions and velocities relative to the default pose, and raw actions, each
history oldest first and filled with the first observation at reset. Joint order is the Isaac library's.

The second half is the evaluation harness for both backends (scripts/evaluate_mujoco.py): play a clip, move the
recording into the Isaac library's frame, score it with humanoid_motion.evaluation.score_attempt and write the files
scripts/evaluate.py writes. Runs with MorphMimic's interpreter (their deploy/ code, mujoco, onnxruntime, torch).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Same values as humanoid_motion.tracking.observations, which imports Isaac Lab and cannot be imported here.
FUTURE_FRAMES, FUTURE_STRIDE, HISTORY = 10, 5, 10
OBSERVATION_NAMES = {'reference': 'v2', 'robot': 'v2.1'}
GRAVITY = np.array([0., 0., -1.])


def quat_conj(q):
    return q * np.array([1., -1., -1., -1.])


def quat_mul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=-1)


def quat_apply(q, v):
    """Rotate vectors v [..., 3] by wxyz quaternions q [..., 4]."""
    u, w = q[..., 1:], q[..., :1]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def yaw_quat(q):
    """Isaac Lab utils.math.yaw_quat for one wxyz quaternion."""
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)])


def matrix_columns(q):
    """First two columns of the rotation matrices of wxyz quaternions [..., 4], row-major flattened [..., 6]."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    m = np.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(x*z - w*y), 2*(y*z + w*x)], -1)
    return m


class BeyondMimicPolicy:
    """Actor of a scripts/train.py checkpoint: empirical normaliser (x - mean) / (std + 0.01), ELU MLP, mean action."""

    def __init__(self, checkpoint: Path, reference: dict, joint_names: list[str], default_pos: np.ndarray,
                 action_scale: np.ndarray, anchor: int):
        import torch
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        infos = state.get('infos') or {}
        if infos.get('observation') not in OBSERVATION_NAMES.values():
            raise ValueError(f'{checkpoint}: observation {infos.get("observation")!r}, expected v2 or v2.1')
        self.heading = {v: k for k, v in OBSERVATION_NAMES.items()}[infos['observation']]
        weights = state['model_state_dict']
        layers = sorted({int(k.split('.')[1]) for k in weights if k.startswith('actor.')})
        self.layers = [(weights[f'actor.{i}.weight'].double().numpy(), weights[f'actor.{i}.bias'].double().numpy())
                       for i in layers]
        norm = state['obs_norm_state_dict']
        self.mean = norm['_mean'].double().numpy().reshape(-1)
        self.std = norm['_std'].double().numpy().reshape(-1)
        self.ref, self.anchor = reference, anchor
        self.joint_names, self.default, self.scale = joint_names, default_pos, action_scale
        self.history = None
        self.last_action = np.zeros(len(joint_names))

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.mean) / (self.std + 1e-2)
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(x))   # ELU
        return x

    def future(self, step: int, robot_anchor_quat: np.ndarray) -> np.ndarray:
        """observations.future_reference at reference phase `step` (clamped at the clip end)."""
        n = len(self.ref['joint_pos'])
        idx = np.minimum(step + np.arange(FUTURE_FRAMES) * FUTURE_STRIDE, n - 1)
        pos = self.ref['body_pos_w'][idx, self.anchor]
        quat = self.ref['body_quat_w'][idx, self.anchor]
        frame = quat[0] if self.heading == 'reference' else robot_anchor_quat
        inv = np.broadcast_to(quat_conj(yaw_quat(frame)), quat.shape)
        ori = matrix_columns(quat_mul(inv, quat))
        disp = quat_apply(inv, pos - pos[:1])
        return np.concatenate([self.ref['joint_pos'][idx], self.ref['joint_vel'][idx], pos[:, 2:3], ori, disp], 1).ravel()

    def step(self, step: int, pelvis_quat, pelvis_gyro_b, q, dq, robot_anchor_quat) -> np.ndarray:
        """Joint position targets (Isaac library joint order) for control step `step`."""
        proprio = [quat_apply(quat_conj(pelvis_quat), GRAVITY), pelvis_gyro_b, q - self.default, dq, self.last_action]
        if self.history is None:                      # Isaac's CircularBuffer: first push fills every slot
            self.history = [np.tile(p, (HISTORY, 1)) for p in proprio]
        else:
            self.history = [np.vstack([h[1:], p]) for h, p in zip(self.history, proprio)]
        obs = np.concatenate([self.future(step, robot_anchor_quat)] + [h.ravel() for h in self.history])
        action = self.act(obs)
        self.last_action = action
        return self.default + self.scale * action


def start_state(sim, clip_ref, joint_names: list[str], start_joint_pos: np.ndarray | None):
    """Initial (qpos, qvel): MorphMimic's eval-env start of the clip (root pose and all velocities), with the joint
    angles replaced by `start_joint_pos` (reference frame 0, library joint order) unless it is None. Their reset clips
    joint angles to the soft limits; the Isaac evaluator starts from the unclipped reference, which this restores.
    Also returns the largest joint-angle change (rad)."""
    import mujoco
    qpos, qvel = np.array(clip_ref['init_qpos'], np.float64), np.array(clip_ref['init_qvel'], np.float64)
    if start_joint_pos is None:
        return qpos, qvel, 0.
    m = sim.model
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'robot/' + n) for n in joint_names]
    if min(ids) < 0:
        raise ValueError('scene is missing library joints')
    adr = m.jnt_qposadr[ids]
    change = float(np.abs(qpos[adr] - start_joint_pos).max())
    qpos[adr] = start_joint_pos
    return qpos, qvel, change


def mm_state(m, d, sim, robot) -> np.ndarray:
    """MorphMimic's world-model state [240] of the simulated robot: canonical q, qd, then position, wxyz quaternion,
    link-origin linear and angular velocity (world axes) of its 14 tracked bodies; their reward head reads it."""
    import mujoco
    vel = np.zeros((len(sim.bodies), 6))
    for i, b in enumerate(sim.bodies):
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, vel[i], 0)
    return np.concatenate([robot.to_canonical(d.qpos[sim.qadr]), robot.to_canonical_delta(d.qvel[sim.dadr]),
                           d.xpos[sim.bodies].ravel(), d.xquat[sim.bodies].ravel(), vel[:, 3:].ravel(), vel[:, :3].ravel()])


def play_beyondmimic(checkpoint: Path, xml: Path, clip: Path, motion: dict, manifest: dict, body_names: list[str],
                     joint_names: list[str], start_joint_pos: np.ndarray | None = None) -> dict:
    """Play one clip with the BeyondMimic policy in MuJoCo; same recording as supermimic.evaluate.play."""
    import mujoco
    from deploy.sim2sim.obs import Robot
    from deploy.sim2sim.sim_node import SimRobot, body_gyro
    ref = np.load(clip)
    robot = Robot.load('g1')                       # MorphMimic's baked G1 constants (joint order, timing)
    sim = SimRobot(xml, robot)
    qpos, qvel, start_change = start_state(sim, ref, joint_names, start_joint_pos)
    sim.reset(qpos, qvel)
    m, d = sim.model, sim.data
    prefix = 'robot/'
    bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + n) for n in body_names]
    joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + n) for n in joint_names]
    if min(bodies) < 0 or min(joints) < 0:
        raise ValueError(f'{xml}: missing library bodies or joints')
    qadr, dadr = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    pelvis, torso = bodies[body_names.index('pelvis')], bodies[body_names.index(manifest['anchor_body'])]
    control = manifest['control']
    policy = BeyondMimicPolicy(checkpoint, motion, joint_names, np.asarray(control['default_joint_pos']),
                               np.asarray(control['action_scale']), body_names.index(manifest['anchor_body']))
    native = list(robot.native_joint_names)
    to_native = [joint_names.index(n) for n in native]
    out = {k: [] for k in ('body_pos_w', 'body_quat_w', 'joint_pos', 'joint_vel', 'actions', 'mm_state')}

    def capture():
        out['body_pos_w'].append(d.xpos[bodies].copy())
        out['body_quat_w'].append(d.xquat[bodies].copy())
        out['joint_pos'].append(d.qpos[qadr].copy())
        out['joint_vel'].append(d.qvel[dadr].copy())
        out['mm_state'].append(mm_state(m, d, sim, robot))

    capture()
    for step in range(len(ref['states']) - 1):    # sim_node.run with delay 0
        targets = policy.step(step, d.xquat[pelvis].copy(), body_gyro(m, d, pelvis), d.qpos[qadr].copy(),
                              d.qvel[dadr].copy(), d.xquat[torso].copy())
        for _ in range(robot.decimation):
            sim.set_targets(targets[to_native])
            sim.step()
        mujoco.mj_forward(m, d)
        out['actions'].append(policy.last_action.copy())
        capture()
    arrays = {k: np.asarray(v) for k, v in out.items()}
    arrays['states'] = ref['states'].astype(np.float64)
    arrays['tracked_bodies'] = list(robot.tracked_bodies)
    arrays['start_change'] = start_change
    return arrays


# ---- evaluation harness shared by both backends (scripts/evaluate_mujoco.py) ----

FRAME_TOLERANCE_M = 1e-3   # the sim2sim and Isaac references must agree up to one constant translation


def frame_offset(isaac_pos: np.ndarray, isaac_quat: np.ndarray, states: np.ndarray, isaac_names: list[str],
                 tracked: list[str]) -> np.ndarray:
    """The constant translation from the sim2sim reference frame to the Isaac library frame, checked on all frames."""
    pos, quat = states[:, 58:100].reshape(-1, 14, 3), states[:, 100:156].reshape(-1, 14, 4)
    idx = [isaac_names.index(n) for n in tracked]
    if len(pos) != len(isaac_pos):
        raise ValueError(f'reference lengths differ: sim2sim {len(pos)}, Isaac {len(isaac_pos)}')
    offset = (isaac_pos[0, idx] - pos[0]).mean(axis=0)
    residual = np.abs(isaac_pos[:, idx] - (pos + offset)).max()
    # atan2 form: arccos of a float32 dot product near 1 turns rounding into ~1e-3 rad
    diff = quat_mul(quat_conj(isaac_quat[:, idx]), quat)
    angle = (2 * np.arctan2(np.linalg.norm(diff[..., 1:], axis=-1), np.abs(diff[..., 0]))).max()
    if residual > FRAME_TOLERANCE_M or angle > 1e-3:
        raise ValueError(f'sim2sim and Isaac references differ: {residual:.2e} m after translation, {angle:.2e} rad')
    return offset


EVIDENCE = {'supermimic': 'SuperMimic (MorphMimic) ONNX policy in MorphMimic CPU MuJoCo sim2sim; no delay, no noise; nominal',
            'beyondmimic': 'BeyondMimic policy (Isaac-trained) in MorphMimic CPU MuJoCo sim2sim scene; no delay, no noise; nominal'}


def evaluate_clip(task: dict) -> dict:
    """Play, align, score (humanoid_motion.evaluation.score_attempt) and write one clip in evaluate.py's format."""
    import json
    from humanoid_motion import ROOT
    from humanoid_motion.evaluation import completion_counts, pose_errors, score_attempt
    from humanoid_motion.library import atomic_json
    row, assets, output = task['row'], Path(task['assets']), Path(task['output'])
    motion_path = ROOT / row['motion']
    manifest = json.loads(motion_path.with_suffix('.manifest.json').read_text())
    names, joint_names = manifest['body_names'], manifest['joint_names']
    with np.load(motion_path) as z:
        motion = {k: z[k].astype(np.float64) for k in ('body_pos_w', 'body_quat_w', 'joint_pos', 'joint_vel')}
    ref_pos, ref_quat = motion['body_pos_w'], motion['body_quat_w']
    clip = assets / 'clips' / 'g1' / f'{row["id"]}.npz'
    start = motion['joint_pos'][0] if task.get('start', 'reference') == 'reference' else None
    if task['kind'] == 'supermimic':
        from humanoid_motion.tracking.supermimic.evaluate import play
        arrays = play(Path(task['policy']), assets / 'g1.xml', clip, names, joint_names, start)
    else:
        arrays = play_beyondmimic(Path(task['policy']), assets / 'g1.xml', clip, motion, manifest, names, joint_names,
                                  start)
    start_change = arrays.pop('start_change')
    states = arrays.pop('states')
    offset = frame_offset(ref_pos, ref_quat, states, names, arrays.pop('tracked_bodies'))
    extra = {'mm_state': arrays.pop('mm_state'), 'mm_reference': states} if 'mm_state' in arrays else {}
    arrays['body_pos_w'] = arrays['body_pos_w'] + offset
    start_error = float(np.abs(arrays['body_pos_w'][0] - ref_pos[0]).max())
    result = score_attempt(ref_pos, ref_quat, arrays['body_pos_w'], arrays['body_quat_w'], names, 'beyondmimic')
    anchor = score_attempt(ref_pos, ref_quat, arrays['body_pos_w'], arrays['body_quat_w'], names, 'anchor')
    count, n = result['scored_frames'], len(ref_pos)
    clip_dir = output / row['id']
    clip_dir.mkdir()
    frames = np.arange(n)
    np.savez_compressed(clip_dir / 'rollout_full.npz', reference_frame_indices=frames, body_names=np.array(names),
                        **extra, **arrays)
    trace = {k: (v[:count - 1] if k == 'actions' else v[:count]) for k, v in arrays.items()}
    np.savez_compressed(clip_dir / 'rollout.npz', reference_frame_indices=frames[:count], body_names=np.array(names),
                        **trace)
    errors = pose_errors(ref_pos[:count], ref_quat[:count], trace['body_pos_w'], trace['body_quat_w'], names)
    np.savez_compressed(clip_dir / 'errors.npz', **errors)
    reason = 'clip_complete' if result['first_failure_frame'] is None else 'terminated'
    attempt = dict(reference_frame_indices=frames[:count].tolist(), terminal_reason=reason,
                   state_writes_after_initialization=0, assistance_count=0)
    atomic_json(clip_dir / 'completion.json', {**completion_counts([attempt], n), 'attempt': attempt})
    evaluation = dict(clip_id=row['id'], category=row['category'], split=row['split'], seed=0,
                      evaluation_split='training' if row['split'] == 'train' else 'held_out_' + row['split'],
                      complete=result['complete'], completion_fraction=result['completion_fraction'],
                      terminal_reason=reason, last_reference_frame=count - 1, elapsed_simulation_seconds=(count - 1) * .02,
                      failure_terms=result['failure_terms'], first_failure_frame=result['first_failure_frame'],
                      protocol='stop at first failure (scored offline; simulated to the clip end)',
                      policy_sha256=task['policy_sha256'], motion_sha256=row['motion_sha256'],
                      library_sha256=task['library_sha256'], body_names=names,
                      state_writes_after_initialization=0, assistance_count=0, evidence=EVIDENCE[task['kind']],
                      metrics_on_observed_prefix=result['metrics_on_observed_prefix'],
                      anchor_criterion=dict(complete=anchor['complete'], first_failure_frame=anchor['first_failure_frame'],
                                            failure_terms=anchor['failure_terms']),
                      sim2sim_frame_offset_m=offset.tolist(), start_pose_max_error_m=start_error,
                      start=task.get('start', 'reference'), start_joint_change_rad=start_change)
    atomic_json(clip_dir / 'evaluation.json', evaluation)
    return evaluation


def init_worker(morphmimic: str) -> None:
    """One thread per worker process: onnxruntime otherwise starts one thread per core in every worker (their
    PolicyRunner passes no session options), and many workers then contend for the cores."""
    import os
    import sys
    os.environ['OMP_NUM_THREADS'] = '1'
    if morphmimic not in sys.path:
        sys.path.insert(0, morphmimic)
    import onnxruntime as ort
    if not getattr(ort.InferenceSession, 'single_threaded', False):
        session_class = ort.InferenceSession

        def single_threaded(path, sess_options=None, **kwargs):
            if sess_options is None:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = sess_options.inter_op_num_threads = 1
            return session_class(path, sess_options=sess_options, **kwargs)
        single_threaded.single_threaded = True
        ort.InferenceSession = single_threaded
    import torch
    torch.set_num_threads(1)


def main(argv=None):
    """Command line of scripts/evaluate_mujoco.py."""
    import argparse
    import concurrent.futures as futures
    import json
    import os
    from humanoid_motion import ROOT
    from humanoid_motion.contracts import sha256
    from humanoid_motion.library import atomic_json
    parser = argparse.ArgumentParser(description='Evaluate a tracking policy on an imported library in CPU MuJoCo '
                                     '(MorphMimic sim2sim scene); writes the files scripts/evaluate.py writes.')
    parser.add_argument('--policy', choices=('supermimic', 'beyondmimic'), required=True)
    parser.add_argument('--deploy', type=Path, help='supermimic: export_onnx folder (policy.onnx, g1.json)')
    parser.add_argument('--checkpoint', type=Path, help='beyondmimic: scripts/train.py checkpoint')
    parser.add_argument('--assets', type=Path, required=True, help='MorphMimic export_assets folder: g1.xml, clips/g1/')
    parser.add_argument('--library', type=Path, required=True, help='Isaac library <split>.json (reference, names)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clips', nargs='*', help='Only these clip IDs')
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--morphmimic', type=Path, default=ROOT / 'third_party/morphmimic')
    parser.add_argument('--label', default='', help='Free text stored in results.json')
    parser.add_argument('--start', choices=('reference', 'morphmimic'), default='reference',
                        help="reference: reference frame 0's joint angles, as the Isaac evaluator (default); "
                        "morphmimic: MorphMimic's eval-env start, joints clipped to the soft limits")
    args = parser.parse_args(argv)
    policy = args.deploy if args.policy == 'supermimic' else args.checkpoint
    if policy is None:
        parser.error('--policy supermimic needs --deploy, --policy beyondmimic needs --checkpoint')
    weights = policy / 'policy.onnx' if args.policy == 'supermimic' else policy
    init_worker(str(args.morphmimic.resolve()))
    rows = json.loads(args.library.read_text())['clips']
    if args.clips:
        rows = [r for r in rows if r['id'] in set(args.clips)]
    args.output.mkdir(parents=True, exist_ok=False)
    common = dict(kind=args.policy, start=args.start, policy=str(policy.resolve()), assets=str(args.assets.resolve()),
                  output=str(args.output.resolve()), policy_sha256=sha256(weights), library_sha256=sha256(args.library))
    with futures.ProcessPoolExecutor(args.workers, initializer=init_worker,
                                     initargs=(str(args.morphmimic.resolve()),)) as pool:
        results = list(pool.map(evaluate_clip, [dict(common, row=r) for r in rows]))
    atomic_json(args.output / 'results.json', dict(
        checkpoint_sha256=common['policy_sha256'], library_sha256=common['library_sha256'], label=args.label,
        policy_kind=args.policy, policy=str(policy), attempts=len(results),
        completions=sum(r['complete'] for r in results),
        anchor_completions=sum(r['anchor_criterion']['complete'] for r in results), rows=results,
        perturbation_protocol=f'nominal phase0, start {args.start}; one attempt per clip; MuJoCo CPU sim2sim'))
    print(f'{args.output}: {sum(r["complete"] for r in results)}/{len(results)} complete (BeyondMimic criterion), '
          f'{sum(r["anchor_criterion"]["complete"] for r in results)}/{len(results)} (anchor criterion)')
