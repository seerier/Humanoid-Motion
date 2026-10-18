"""Observation v2: a future reference window plus proprioceptive history (BeyondMimic backend).

Imported only after the Isaac AppLauncher. The actor needs no robot world position.
heading='reference' (v2): future torso orientation/displacement in the reference's own current heading frame.
heading='robot' (v2.1): the same quantities in the robot's current heading frame, so the robot-vs-reference
yaw difference is observed. Checkpoints record which one they were trained with (infos['observation']).
"""
import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul, yaw_quat
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import whole_body_tracking.tasks.tracking.mdp as mdp

FUTURE_FRAMES = 10
FUTURE_STRIDE = 5  # 0.1 s at 50 Hz
HISTORY = 10       # 10 x 20 ms
ACTOR_DIM = FUTURE_FRAMES * 68 + HISTORY * 93
CRITIC_DIM = ACTOR_DIM + 3 + 3 + 6 + 14 * 3 + 14 * 6


def future_reference(env, command_name='motion', heading='reference'):
    """Per frame: joint pos (29), joint vel (29), torso height (1), torso orientation (6D) and
    displacement from the current reference torso (3), both in the reference's (or robot's) current heading frame;
    frames 0.0..0.9 s, clamped at clip end."""
    c = env.command_manager.get_term(command_name)
    steps = torch.arange(FUTURE_FRAMES, device=c.device) * FUTURE_STRIDE
    last = (c.offsets + c.lengths - 1)[c.clip_ids]
    idx = torch.minimum(c.time_steps[:, None] + steps[None], last[:, None])
    a = c.motion_anchor_body_index
    pos = c.motion.body_pos_w[idx, a]
    quat = c.motion.body_quat_w[idx, a]
    frame = {'reference': quat[:, 0], 'robot': c.robot_anchor_quat_w}[heading]
    inv_heading = quat_inv(yaw_quat(frame))[:, None].expand(-1, FUTURE_FRAMES, -1)
    ori = matrix_from_quat(quat_mul(inv_heading, quat))[..., :2].reshape(*idx.shape, 6)
    disp = quat_apply(inv_heading, pos - pos[:, :1])
    out = torch.cat((c.motion.joint_pos[idx], c.motion.joint_vel[idx], pos[..., 2:3], ori, disp), -1)
    return out.reshape(env.num_envs, -1)


def clip_end_timeout(env, command_name='motion'):
    """Clip end is a truncation (bootstrapped); a failure on the last frame stays a true terminal."""
    c = env.command_manager.get_term(command_name)
    return (c.phases >= c.lengths[c.clip_ids] - 1) & ~c.last_physical_failure


def _proprio(noise):
    n = (lambda lo: Unoise(n_min=-lo, n_max=lo)) if noise else (lambda lo: None)
    return dict(
        projected_gravity=ObsTerm(func=mdp.projected_gravity, noise=n(.05), history_length=HISTORY),
        base_ang_vel=ObsTerm(func=mdp.base_ang_vel, noise=n(.2), history_length=HISTORY),
        joint_pos=ObsTerm(func=mdp.joint_pos_rel, noise=n(.01), history_length=HISTORY),
        joint_vel=ObsTerm(func=mdp.joint_vel_rel, noise=n(.5), history_length=HISTORY),
        actions=ObsTerm(func=mdp.last_action, history_length=HISTORY))


@configclass
class PolicyV2(ObsGroup):
    future = ObsTerm(func=future_reference, params={'command_name': 'motion'})
    projected_gravity = _proprio(True)['projected_gravity']
    base_ang_vel = _proprio(True)['base_ang_vel']
    joint_pos = _proprio(True)['joint_pos']
    joint_vel = _proprio(True)['joint_vel']
    actions = _proprio(True)['actions']

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CriticV2(ObsGroup):
    future = ObsTerm(func=future_reference, params={'command_name': 'motion'})
    projected_gravity = _proprio(False)['projected_gravity']
    base_ang_vel = _proprio(False)['base_ang_vel']
    joint_pos = _proprio(False)['joint_pos']
    joint_vel = _proprio(False)['joint_vel']
    actions = _proprio(False)['actions']
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={'command_name': 'motion'})
    motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={'command_name': 'motion'})
    body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={'command_name': 'motion'})
    body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={'command_name': 'motion'})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


OBSERVATION_NAMES = {'reference': 'v2', 'robot': 'v2.1'}  # checkpoint infos['observation'] -> heading frame


def apply_v2(cfg, training=True, heading='reference'):
    """Observation v2 groups; clip end as time-out during training (BeyondMimic rewards/failures unchanged)."""
    if heading not in OBSERVATION_NAMES:
        raise ValueError(f'Unknown heading frame: {heading}')
    cfg.observations.policy = PolicyV2()
    cfg.observations.critic = CriticV2()
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.future.params = {'command_name': 'motion', 'heading': heading}
    if training:
        cfg.terminations.time_out = None
        cfg.terminations.clip_end = TerminationTermCfg(func=clip_end_timeout, time_out=True)
    return cfg
