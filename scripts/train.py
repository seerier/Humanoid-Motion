#!/usr/bin/env python3
"""Train one multi-motion G1 tracking policy with PPO in Isaac Lab (BeyondMimic environment).

Observation v2 (future reference window + proprioceptive history), failure-adaptive clip and phase sampling,
BeyondMimic rewards and failure terminations unchanged; the clip end is a bootstrapped time-out.
Resumes from <output>/latest.pt. --train-hours is cumulative wall time across resumes; training also stops
at --deadline or on SIGTERM/SIGUSR1, saving a checkpoint first.
"""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
from humanoid_motion import ROOT
from humanoid_motion.library import atomic_json, load_library
from humanoid_motion.contracts import sha256, TRACKING_COMMIT, URDF_SHA256
LIBRARY = ROOT / 'outputs/lafan1/runtime/train.json'


class Stop(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--hidden', type=int, nargs='+', default=[1024, 1024, 512, 256])
    parser.add_argument('--train-hours', type=float, default=10., help='Cumulative training wall time, including earlier resumes')
    parser.add_argument('--deadline', type=float, default=float('inf'), help='Epoch seconds; stop and save before this')
    parser.add_argument('--smoke-iterations', type=int, default=0, help='>0: short contract smoke, then exit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--heading', choices=('reference', 'robot'), default='reference',
                        help="Frame of the future orientation/displacement: 'reference' = v2, 'robot' = v2.1")
    parser.add_argument('--library', type=Path, default=LIBRARY, help='Imported Isaac library (import_library.py)')
    parser.add_argument('--usd-template', type=Path, help='Optional verified URDF->USD conversion to copy instead of converting')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    upstream = ROOT / 'third_party/whole_body_tracking'
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != TRACKING_COMMIT \
            or subprocess.check_output(['git', '-C', str(upstream), 'diff', 'HEAD', '--', 'source'], text=True):
        raise ValueError('Pinned upstream changed')
    # Absolute: a relative USD dir makes the asset's relative sublayers/payloads resolve from '/' (2026-10-13 smoke).
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    from humanoid_motion.runtime import monitor_resources
    launcher = None
    try:
        with monitor_resources(args.output / f'resources-{os.getpid()}.json') as resources:
            launcher = AppLauncher(args)
            train(args, resources)
    except BaseException as error:
        import traceback
        traceback.print_exc()
        atomic_json(args.output / f'error-{os.getpid()}.json', dict(error=repr(error), traceback=traceback.format_exc(), epoch=time.time()))
        raise
    finally:
        sys.stdout.flush()
        if launcher is not None:
            from isaaclab.sim import SimulationContext
            sim = SimulationContext.instance()
            if sim is not None:
                sim._disable_app_control_on_stop_handle = True
            launcher.app.close(wait_for_replicator=False)


def train(args, resources):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    import whole_body_tracking.tasks
    from humanoid_motion.tracking.command import MultiMotionCommand, install_phase_bridge
    from humanoid_motion.tracking.checks import check_runtime_contracts
    from humanoid_motion.tracking.observations import apply_v2, ACTOR_DIM, CRITIC_DIM, OBSERVATION_NAMES
    from humanoid_motion import wandb_mirror as wandb_run
    smoke = args.smoke_iterations > 0
    atomic_json(args.output / 'engineering_checks.json', check_runtime_contracts(args.device))
    library = args.library.resolve()
    manifest, arrays, contract = load_library(library, ROOT)
    library_hash = sha256(library)
    del arrays
    cfg = load_cfg_from_registry('Tracking-Flat-G1-v0', 'env_cfg_entry_point')
    agent = load_cfg_from_registry('Tracking-Flat-G1-v0', 'rsl_rl_cfg_entry_point')
    cfg.scene.num_envs = args.num_envs
    cfg.seed = agent.seed = args.seed
    cfg.sim.device = agent.device = args.device
    cfg.commands.motion.motion_file = str(ROOT / manifest['clips'][0]['motion'])
    cfg.commands.motion.library_file = str(library)
    cfg.commands.motion.project_root = str(ROOT)
    cfg.commands.motion.class_type = MultiMotionCommand
    cfg.commands.motion.debug_vis = False
    apply_v2(cfg, training=True, heading=args.heading)
    observation = OBSERVATION_NAMES[args.heading]
    cfg.scene.robot.spawn.asset_path = str(ROOT / 'third_party/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf')
    if sha256(Path(cfg.scene.robot.spawn.asset_path)) != URDF_SHA256:
        raise ValueError('Asset mismatch')
    # Fresh per process: a conversion interrupted after writing .asset_hash must never be reused. Seeding it with
    # a verified conversion of the same URDF/config (same .asset_hash) skips URDF->USD conversion, which can stall
    # on network file systems; a hash mismatch still converts.
    usd_dir = args.output / f'usd-{os.getpid()}'
    if args.usd_template and args.usd_template.joinpath('.asset_hash').exists():
        shutil.copytree(args.usd_template, usd_dir)
    cfg.scene.robot.spawn.usd_dir = str(usd_dir)
    cfg.scene.robot.spawn.usd_file_name = 'g1.usd'
    agent.policy.actor_hidden_dims = list(args.hidden)
    agent.policy.critic_hidden_dims = list(args.hidden)
    agent.logger = 'tensorboard'
    agent.save_interval = 10 ** 9
    dump_yaml(str(args.output / 'env.yaml'), cfg)
    dump_yaml(str(args.output / 'agent.yaml'), agent)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    base = ManagerBasedRLEnv(cfg)
    install_phase_bridge(base)
    wrapped = RslRlVecEnvWrapper(base)
    command = base.command_manager.get_term('motion')
    obs, extra = wrapped.get_observations()
    dims = (obs.shape[-1], extra['observations']['critic'].shape[-1], wrapped.num_actions)
    if dims != (ACTOR_DIM, CRITIC_DIM, 29):
        raise ValueError(f'Observation/action contract changed: {dims}')
    if not (torch.isfinite(obs).all() and torch.isfinite(extra['observations']['critic']).all()):
        raise FloatingPointError('Nonfinite initial observation')
    stop_requested = False

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    budget = args.train_hours * 3600
    steps = agent.num_steps_per_env

    class Runner(OnPolicyRunner):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.updates = 0
            self.train_seconds = 0.
            self.wall_seconds = 0.
            self.resume_count = 0
            self.last_save = time.monotonic()
            self.last_status = 0.
            self.last_keep = 0.
            self.last_tick = time.monotonic()
            self.prev_stats = None

        def snapshot(self, keep=False):
            state = self.alg.policy.state_dict()
            if not all(torch.isfinite(v).all() for v in state.values()):
                raise FloatingPointError('Nonfinite policy; checkpoint not published')
            info = dict(schema='humanoid_motion.checkpoint.v1', updates=self.updates, train_seconds=self.train_seconds,
                        wall_seconds=self.wall_seconds, transitions=self.updates * args.num_envs * steps,
                        library_sha256=library_hash, hidden=list(args.hidden), num_envs=args.num_envs,
                        actor_dim=ACTOR_DIM, critic_dim=CRITIC_DIM, sampler=command.sampler_state(),
                        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                 cuda=torch.cuda.get_rng_state_all()),
                        resume_count=self.resume_count, observation=observation, rewards='BeyondMimic', seed=args.seed,
                        learning_rate=float(self.alg.learning_rate))
            payload = dict(model_state_dict=state, optimizer_state_dict=self.alg.optimizer.state_dict(),
                           obs_norm_state_dict=self.obs_normalizer.state_dict(),
                           privileged_obs_norm_state_dict=self.privileged_obs_normalizer.state_dict(),
                           iter=self.updates, infos=info)
            tmp = args.output / 'latest.pt.tmp'
            torch.save(payload, tmp)
            with tmp.open('rb') as stream:
                os.fsync(stream.fileno())
            os.replace(tmp, args.output / 'latest.pt')
            if keep:
                shutil.copyfile(args.output / 'latest.pt', args.output / f'checkpoint_{self.updates:07d}.pt')
            atomic_json(args.output / 'latest.json', dict(updates=self.updates, train_seconds=self.train_seconds,
                        wall_seconds=self.wall_seconds, epoch=time.time(), sha256=sha256(args.output / 'latest.pt')))
            atomic_json(args.output / 'sampling.json', command.statistics())
            self.last_save = time.monotonic()
            return args.output / 'latest.pt'

        def save(self, path, infos=None):
            return self.snapshot()

        def log(self, locs, *a, **kw):
            super().log(locs, *a, **kw)
            now = time.monotonic()
            elapsed = locs['collection_time'] + locs['learn_time']
            self.updates += 1
            self.train_seconds += elapsed
            self.wall_seconds += now - self.last_tick
            self.last_tick = now
            command.learning_seconds = self.train_seconds
            losses = {k: float(v) for k, v in locs['loss_dict'].items()}
            if not np.isfinite(list(losses.values())).all():
                raise FloatingPointError('Nonfinite PPO loss')
            stats = {k: v.sum().item() for k, v in command.stats.items()}
            delta = {k: stats[k] - (self.prev_stats or {}).get(k, 0) for k in stats}
            self.prev_stats = stats
            ends = delta['failures'] + delta['normal_ends']
            row = dict(update=self.updates, train_seconds=self.train_seconds, wall_seconds=self.wall_seconds,
                       update_seconds=elapsed, transitions_per_second=args.num_envs * steps / elapsed,
                       mean_reward=float(np.mean(locs['rewbuffer'])) if len(locs['rewbuffer']) else None,
                       mean_episode_steps=float(np.mean(locs['lenbuffer'])) if len(locs['lenbuffer']) else None,
                       clip_completion_rate=delta['normal_ends'] / ends if ends else None,
                       losses=losses, learning_rate=float(self.alg.learning_rate), action_std=float(self.alg.policy.std.mean()),
                       gpu_reserved_peak_bytes=torch.cuda.max_memory_reserved())
            with (args.output / 'curve.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            if row['clip_completion_rate'] is not None:
                self.writer.add_scalar('Train/clip_completion_rate', row['clip_completion_rate'], self.updates)
            if now - self.last_save >= 300:
                self.snapshot(keep=now - self.last_keep >= 3600)
                if now - self.last_keep >= 3600:
                    self.last_keep = now
            if now - self.last_status >= 60:
                gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                                               '--format=csv,noheader'], text=True).strip()
                atomic_json(args.output / 'status.json', dict(stage='smoke' if smoke else 'training', job_id=os.environ.get('SLURM_JOB_ID'),
                            budget_remaining_s=budget - self.wall_seconds,
                            deadline_remaining_s=None if args.deadline == float('inf') else args.deadline - time.time(),
                            gpu=gpu, **row))
                print('PROGRESS', json.dumps(row), flush=True)
                self.last_status = now
            if smoke and self.updates >= args.smoke_iterations:
                saved = self.snapshot()
                recovered = torch.load(saved, map_location=args.device, weights_only=False)
                for k, v in self.alg.policy.state_dict().items():
                    if not torch.equal(v, recovered['model_state_dict'][k]):
                        raise ValueError('Checkpoint round-trip tensor mismatch')
                before = command.statistics()
                command.restore_sampler(recovered['infos']['sampler'])
                if command.statistics() != before:
                    raise ValueError('Sampler round-trip mismatch')
                atomic_json(args.output / 'smoke.json', dict(passed=True, num_envs=args.num_envs, hidden=list(args.hidden),
                            actor_dim=ACTOR_DIM, critic_dim=CRITIC_DIM, updates=self.updates,
                            transitions_per_second=self.updates * args.num_envs * steps / self.train_seconds,
                            seconds_per_update=self.train_seconds / self.updates,
                            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            normal_ends=int(command.stats['normal_ends'].sum()), failures=int(command.stats['failures'].sum())))
                raise Stop('smoke complete')
            if stop_requested or self.wall_seconds >= budget or time.time() >= args.deadline:
                raise Stop('budget, deadline or signal')

    runner = Runner(wrapped, agent.to_dict(), log_dir=str(args.output), device=args.device)
    resume = args.output / 'latest.pt'
    if resume.exists() and not smoke:
        infos = runner.load(str(resume))
        if (infos['library_sha256'] != library_hash or infos['hidden'] != list(args.hidden)
                or infos.get('observation', 'v2') != observation):
            raise ValueError('Resume provenance mismatch')
        runner.updates, runner.train_seconds, runner.wall_seconds = infos['updates'], infos['train_seconds'], infos['wall_seconds']
        # rsl_rl keeps the adaptive learning rate in PPO.learning_rate (initialised from the config, 1e-3) and writes it into
        # the optimizer every minibatch, so restoring the optimizer alone resumes at 1e-3 and wrecks a policy trained at
        # ~1.5e-5. Older checkpoints lack infos['learning_rate']; their optimizer holds the last value.
        runner.alg.learning_rate = float(infos.get('learning_rate', runner.alg.optimizer.param_groups[0]['lr']))
        print('RESUME learning_rate', runner.alg.learning_rate, flush=True)
        runner.resume_count = infos['resume_count'] + 1
        command.restore_sampler(infos['sampler'])
        random.setstate(infos['rng']['python'])
        np.random.set_state(infos['rng']['numpy'])
        torch.set_rng_state(infos['rng']['torch'])
        torch.cuda.set_rng_state_all(infos['rng']['cuda'])
        wrapped.reset()
    runner.current_learning_iteration = runner.updates
    runner.last_tick = time.monotonic()
    atomic_json(args.output / f'provenance-{runner.resume_count}.json', dict(library_sha256=library_hash,
                job=os.environ.get('SLURM_JOB_ID'), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT / 'src/humanoid_motion/tracking/observations.py',
                             ROOT / 'src/humanoid_motion/tracking/command.py', ROOT / 'src/humanoid_motion/library.py']},
                tracker_commit=TRACKING_COMMIT, seed=args.seed, num_envs=args.num_envs, hidden=list(args.hidden),
                resume_count=runner.resume_count, smoke=smoke, agent=agent.to_dict()))
    mirror = None if smoke else wandb_run.start(args.output, 'train', dict(num_envs=args.num_envs, hidden=list(args.hidden),
                                                seed=args.seed, library_sha256=library_hash, observation=observation, rewards='BeyondMimic',
                                                clip_end='time_out', job=os.environ.get('SLURM_JOB_ID'), resume_count=runner.resume_count,
                                                agent=agent.to_dict()))
    failed = True
    try:
        try:
            runner.learn(num_learning_iterations=10 ** 7, init_at_random_ep_len=False)
        except Stop:
            pass
        if not smoke:
            runner.snapshot(keep=True)
            if runner.wall_seconds >= budget:
                atomic_json(args.output / 'complete.json', dict(updates=runner.updates, train_seconds=runner.train_seconds,
                            wall_seconds=runner.wall_seconds, epoch=time.time()))
        if runner.writer is not None:
            runner.writer.flush()
        failed = False
    finally:
        if mirror is not None:
            wandb_run.finish(mirror, failed)
        wrapped.close()


if __name__ == '__main__':
    main()
