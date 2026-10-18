"""Project adapter for pinned BeyondMimic, imported only after Isaac AppLauncher."""
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from humanoid_motion.library import bounded_probabilities, load_library


class MultiMotionCommand(MotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        manifest, arrays, contract = load_library(Path(cfg.library_file), Path(cfg.project_root))
        if contract['joint_names'] != self.robot.joint_names or contract['body_names'] != self.robot.body_names:
            raise ValueError('Robot runtime names mismatch')
        self.library_manifest = manifest
        self.contract = contract
        self.lengths = torch.tensor([len(a['joint_pos']) for a in arrays], device=self.device, dtype=torch.long)
        self.offsets = torch.cat((torch.zeros(1, device=self.device, dtype=torch.long), self.lengths.cumsum(0)[:-1]))
        fields = {}
        for key in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'):
            data = torch.tensor(np.concatenate([a[key] for a in arrays]), device=self.device, dtype=torch.float32)
            fields[key] = data[:, self.body_indexes] if key.startswith('body_') else data
        self.motion = SimpleNamespace(**fields, time_step_total=int(self.lengths.sum()))
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.nclips, self.nbins = len(arrays), 12
        self.stats = {k: torch.zeros(self.nclips, device=self.device, dtype=torch.long)
                      for k in ('starts', 'transitions', 'failures', 'normal_ends')}
        self.phase_visits = torch.zeros((self.nclips, self.nbins), device=self.device, dtype=torch.long)
        self.phase_failures = torch.zeros_like(self.phase_visits)
        self.failure_ema = torch.full((self.nclips,), .5, device=self.device)
        self.learning_seconds = 0.
        self.eval_mode = bool(getattr(cfg, 'evaluation', False))
        self.eval_clip_ids = None
        self.eval_perturb = False
        self.state_writes = 0
        self._reset_allowed = True
        self.last_advanced_step = -1
        self.last_physical_failure = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_natural_end = self.last_physical_failure.clone()
        self.probabilities = torch.full((self.nclips,), 1/self.nclips, device=self.device)
        self.probability_refresh = -1.

    @property
    def phases(self):
        return self.time_steps - self.offsets[self.clip_ids]

    def _adaptive_sampling(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self.learning_seconds - self.probability_refresh >= 10:
            p = bounded_probabilities(self.failure_ema.cpu().numpy(), self.learning_seconds >= 1800)
            self.probabilities = torch.tensor(p, device=self.device, dtype=torch.float32)
            self.probability_refresh = self.learning_seconds
        clips = torch.multinomial(self.probabilities, len(ids), replacement=True)
        self.clip_ids[ids] = clips
        difficulty = self.phase_failures[clips].float() + 1
        # Uniform coverage remains at least 50% at every phase bin.
        pphase = .5 / self.nbins + .5 * difficulty / difficulty.sum(1, keepdim=True)
        bins = torch.multinomial(pphase, 1).squeeze(1)
        phases = ((bins + torch.rand(len(ids), device=self.device)) / self.nbins * (self.lengths[clips]-1)).long()
        self.time_steps[ids] = self.offsets[clips] + phases
        self.stats['starts'] += torch.bincount(clips, minlength=self.nclips)

    def _resample_command(self, env_ids):
        if not len(env_ids):
            return
        if not self._reset_allowed:
            raise RuntimeError('State write outside explicit environment reset')
        if self.eval_mode:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            chosen = self.eval_clip_ids if self.eval_clip_ids is not None else torch.arange(self.num_envs, device=self.device) % self.nclips
            self.clip_ids[ids] = chosen[ids]
            self.time_steps[ids] = self.offsets[self.clip_ids[ids]]
            if self.eval_perturb:
                # Reuse identical upstream initialization noise without changing the fixed phase/clip.
                self._adaptive_sampling_saved = self._adaptive_sampling
                self._adaptive_sampling = lambda _: None
                try:
                    super()._resample_command(ids)
                finally:
                    self._adaptive_sampling = self._adaptive_sampling_saved
            else:
                root = torch.cat((self.body_pos_w[:, 0], self.body_quat_w[:, 0], self.body_lin_vel_w[:, 0], self.body_ang_vel_w[:, 0]), -1)
                self.robot.write_root_state_to_sim(root[ids], env_ids=ids)
                self.robot.write_joint_state_to_sim(self.joint_pos[ids], self.joint_vel[ids], env_ids=ids)
        else:
            super()._resample_command(env_ids)
        self.state_writes += len(env_ids)
        self._update_command()

    def _update_command(self):
        # Refresh only. Reference advancement occurs after physics, before rewards/dones.
        anchor = self.anchor_pos_w[:, None, :]
        delta_pos = self.robot_anchor_pos_w[:, None, :].clone()
        delta_pos[..., 2] = anchor[..., 2]
        delta_q = yaw_quat(quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w)))[:, None, :]
        delta_q = delta_q.expand(-1, len(self.cfg.body_names), -1)
        self.body_quat_relative_w = quat_mul(delta_q, self.body_quat_w)
        self.body_pos_relative_w = delta_pos + quat_apply(delta_q, self.body_pos_w-anchor)

    def advance(self):
        step = self._env.common_step_counter
        if step <= self.last_advanced_step:
            raise RuntimeError('Reference advanced twice or out of order')
        self.last_advanced_step = step
        self.time_steps = torch.minimum(self.time_steps + 1, self.offsets[self.clip_ids] + self.lengths[self.clip_ids]-1)
        self._update_command()

    def record(self, physical, ended):
        if self.eval_mode:
            return
        clips = self.clip_ids
        bins = torch.minimum(self.phases * self.nbins // self.lengths[clips], torch.tensor(self.nbins-1, device=self.device))
        self.stats['transitions'] += torch.bincount(clips, minlength=self.nclips)
        self.phase_visits.view(-1).add_(torch.bincount(clips*self.nbins+bins, minlength=self.nclips*self.nbins))
        self.phase_failures.view(-1).add_(torch.bincount((clips*self.nbins+bins)[physical], minlength=self.nclips*self.nbins))
        failures = torch.bincount(clips[physical], minlength=self.nclips)
        normals = torch.bincount(clips[ended & ~physical], minlength=self.nclips)
        self.stats['failures'] += failures
        self.stats['normal_ends'] += normals
        total = failures + normals
        mask = total > 0
        alpha = 1 - .95 ** total[mask].float()
        self.failure_ema[mask] = (1-alpha)*self.failure_ema[mask] + alpha * failures[mask]/total[mask]

    def sampler_state(self):
        return dict(stats={k:v.cpu() for k,v in self.stats.items()}, phase_visits=self.phase_visits.cpu(),
                    phase_failures=self.phase_failures.cpu(), failure_ema=self.failure_ema.cpu(),
                    learning_seconds=self.learning_seconds)

    def restore_sampler(self, state):
        for key, value in state['stats'].items():
            self.stats[key].copy_(value.to(self.device))
        for key in ('phase_visits','phase_failures','failure_ema'):
            getattr(self, key).copy_(state[key].to(self.device))
        self.learning_seconds = state['learning_seconds']
        self.probability_refresh = -1.

    def statistics(self):
        data = {k:v.cpu().tolist() for k,v in self.stats.items()}
        data.update(clip_ids=[r['id'] for r in self.library_manifest['clips']],
            phase_visits=self.phase_visits.cpu().tolist(), phase_failures=self.phase_failures.cpu().tolist(),
            failure_rate_ema=self.failure_ema.cpu().tolist(), probabilities=self.probabilities.cpu().tolist(),
            learning_seconds=self.learning_seconds, state_write_env_count=self.state_writes)
        return data


def clip_end(env, command_name='motion'):
    command = env.command_manager.get_term(command_name)
    return command.phases >= command.lengths[command.clip_ids]-1


def install_phase_bridge(env, evaluation=False):
    """Single insertion point into pinned Isaac step, with an explicit reset-write guard."""
    command = env.command_manager.get_term('motion')
    original_compute = env.termination_manager.compute
    original_reset = env._reset_idx
    from whole_body_tracking.tasks.tracking import mdp
    from isaaclab.managers import SceneEntityCfg
    asset_cfg = SceneEntityCfg('robot')
    asset_cfg.resolve(env.scene)

    def compute():
        command.advance()
        command.last_failure_terms = {
            'anchor_pos': mdp.bad_anchor_pos_z_only(env, 'motion', .25),
            'anchor_ori': mdp.bad_anchor_ori(env, asset_cfg, 'motion', .8),
            'ee_body_pos': mdp.bad_motion_body_pos_z_only(env, 'motion', .25,
                ['left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link'])}
        physical = torch.stack(list(command.last_failure_terms.values())).any(0)
        ended = clip_end(env)
        command.last_physical_failure = physical
        command.last_natural_end = ended
        command.record(physical, ended)
        result = original_compute()
        if not evaluation and not torch.equal(result, physical | ended):
            raise RuntimeError('Termination contract differs from recorded failures/ends')
        return result

    def reset(ids):
        command._reset_allowed = True
        try:
            return original_reset(ids)
        finally:
            command._reset_allowed = False

    env.termination_manager.compute = compute
    env._reset_idx = reset
    command._reset_allowed = False
