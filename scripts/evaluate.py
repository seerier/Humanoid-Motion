#!/usr/bin/env python3
"""Evaluate a tracking policy on an imported library: one fixed-start attempt per clip, all clips in parallel.

No auto-reset and no policy-dependent clip selection. By default an attempt stops at its first failure (any
BeyondMimic failure term). Writes results.json plus, per clip, evaluation.json, rollout.npz (robot states) and
errors.npz (per-frame tracking errors). Any policy whose rollouts are saved in this format can be compared.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from humanoid_motion import ROOT
from humanoid_motion.library import atomic_json, load_library
from humanoid_motion.contracts import sha256
from humanoid_motion.evaluation import pose_errors, completion_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--perturb', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--continue-after-failure', action='store_true',
                        help='Diagnostic: keep simulating to the clip end and record the first failure frame')
    parser.add_argument('--usd-template', type=Path, help='Optional verified URDF->USD conversion to copy instead of converting')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from humanoid_motion.runtime import monitor_resources
    launcher = None
    try:
        with monitor_resources(args.output/'resources.json'):
            launcher = AppLauncher(args)
            evaluate(args)
    except BaseException as error:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        atomic_json(args.output / ('error-' + Path(__file__).stem + '-' + str(os.getpid()) + '.json'), dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        if launcher is not None:
            from isaaclab.sim import SimulationContext
            sim = SimulationContext.instance()
            if sim is not None:
                sim._disable_app_control_on_stop_handle = True
            launcher.app.close(wait_for_replicator=False)


def evaluate(args):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner
    import whole_body_tracking.tasks
    from humanoid_motion.tracking.command import MultiMotionCommand, install_phase_bridge
    manifest, motions, contract = load_library(args.library, ROOT)
    rows = manifest['clips']
    selected = list(range(len(rows)))
    if args.perturb:
        selected = [i for i,r in enumerate(rows) if r['category']=='walk'][:6]
        selected += [i for i,r in enumerate(rows) if r['category']=='aiming'][:4]
        selected += [i for i,r in enumerate(rows) if r['category']=='dance'][:2]
        if len(selected) != 12:
            raise ValueError('Perturbation suite needs 12 fixed basic test clips')
    cfg = load_cfg_from_registry('Tracking-Flat-G1-v0','env_cfg_entry_point')
    agent = load_cfg_from_registry('Tracking-Flat-G1-v0','rsl_rl_cfg_entry_point')
    cfg.scene.num_envs = len(selected)
    # Each environment is separated enough for every reference trajectory in this batch.
    extent = max(float(np.ptp(motions[i]['body_pos_w'][:,0,:2], axis=0).max()) for i in selected)
    cfg.scene.env_spacing = max(5., 2*extent+4.)
    cfg.seed = agent.seed = 0
    cfg.sim.device = agent.device = args.device
    cfg.scene.robot.spawn.asset_path = str((ROOT/'third_party/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf').resolve())
    usd_dir = args.output.resolve()/'usd'  # relative USD dirs break the asset's relative payload paths
    if args.usd_template and args.usd_template.joinpath('.asset_hash').exists():
        # Same URDF/config hash: skips URDF->USD conversion, which can stall on network file systems.
        import shutil
        shutil.copytree(args.usd_template, usd_dir)
    cfg.scene.robot.spawn.usd_dir = str(usd_dir)
    cfg.scene.robot.spawn.usd_file_name = 'g1.usd'
    cfg.commands.motion.class_type = MultiMotionCommand
    cfg.commands.motion.motion_file = str(ROOT/rows[0]['motion'])
    cfg.commands.motion.library_file = str(args.library.resolve())
    cfg.commands.motion.project_root = str(ROOT)
    cfg.commands.motion.evaluation = True
    cfg.commands.motion.debug_vis = False
    cfg.events = {}
    infos = torch.load(args.checkpoint, map_location='cpu', weights_only=False).get('infos') or {}
    if infos.get('observation') in ('v2', 'v2.1'):
        from humanoid_motion.tracking.observations import apply_v2, OBSERVATION_NAMES
        apply_v2(cfg, training=False, heading={v: k for k, v in OBSERVATION_NAMES.items()}[infos['observation']])
        agent.policy.actor_hidden_dims = list(infos['hidden'])
        agent.policy.critic_hidden_dims = list(infos['hidden'])
    cfg.observations.policy.enable_corruption = False
    cfg.terminations = {}
    base = ManagerBasedRLEnv(cfg)
    install_phase_bridge(base, evaluation=True)
    wrapped = RslRlVecEnvWrapper(base)
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=args.device)
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy()
    command = base.command_manager.get_term('motion')
    command.eval_clip_ids = torch.tensor(selected, device=args.device)
    command.eval_perturb = args.perturb
    robot = base.scene['robot']
    all_results = []
    to_np = lambda v:v.detach().cpu().numpy().copy()
    try:
        for seed in (0, 1, 2) if args.perturb else (0,):
            torch.manual_seed(seed)
            np.random.seed(seed)
            obs, _ = wrapped.reset()
            command._update_command()
            obs, _ = wrapped.get_observations()
            initial_writes = command.state_writes
            arrays = {k:[] for k in ('body_pos_w','body_quat_w','body_link_lin_vel_w','joint_pos','joint_vel','applied_torque','actions')}
            last = np.zeros(len(selected), dtype=int)
            failed = np.zeros(len(selected), dtype=bool)
            reasons = [[] for _ in selected]
            first_failure = [None] * len(selected)
            alive = np.ones(len(selected), dtype=bool)
            max_frames = max(len(motions[i]['joint_pos']) for i in selected)

            def capture():
                for key in arrays:
                    if key == 'actions':
                        continue
                    value = to_np(getattr(robot.data, key))
                    if key == 'body_pos_w':
                        value -= to_np(base.scene.env_origins)[:,None,:]
                    arrays[key].append(value)

            capture()
            with torch.inference_mode():
                for frame in range(1, max_frames):
                    action = policy(obs)
                    arrays['actions'].append(to_np(action))
                    obs, _, done, _ = wrapped.step(action)
                    if bool(done.any()) or command.state_writes != initial_writes:
                        raise RuntimeError('Hidden reset/state write in evaluation')
                    if not torch.isfinite(robot.data.joint_pos).all():
                        raise FloatingPointError('Nonfinite simulation during evaluation')
                    capture()
                    bad = to_np(command.last_physical_failure)
                    for j,i in enumerate(selected):
                        if alive[j]:
                            last[j] = min(frame, len(motions[i]['joint_pos'])-1)
                            if bad[j] and first_failure[j] is None:
                                first_failure[j] = int(last[j])
                                reasons[j] = [key for key,values in command.last_failure_terms.items() if bool(values[j])]
                            failed[j] = first_failure[j] is not None
                            # Diagnostic mode records past the first failure; the reference holds its last frame at clip end.
                            alive[j] = (args.continue_after_failure or not bad[j]) and frame < len(motions[i]['joint_pos'])-1
                    if not alive.any():
                        break
            arrays = {k:np.asarray(v) for k,v in arrays.items()}
            for j,i in enumerate(selected):
                row, motion = rows[i], motions[i]
                count = int(last[j])+1
                reason = 'terminated' if failed[j] and not args.continue_after_failure else 'clip_complete'
                complete = not failed[j] and count == len(motion['joint_pos'])  # unchanged meaning: no failure at all
                clip_dir = args.output / (row['id'] + (f'_seed{seed}' if args.perturb else ''))
                frames = np.arange(count)
                trace = {k:v[:count-(k=='actions'),j] for k,v in arrays.items()}
                errors = pose_errors(motion['body_pos_w'][:count],motion['body_quat_w'][:count],trace['body_pos_w'],trace['body_quat_w'],contract['body_names'])
                result = dict(clip_id=row['id'], category=row['category'], split=row['split'], seed=seed,
                    evaluation_split='training' if row['split']=='train' else 'held_out_'+row['split'],
                    complete=bool(complete), completion_fraction=int(last[j])/(len(motion['joint_pos'])-1),
                    terminal_reason=reason, last_reference_frame=int(last[j]), elapsed_simulation_seconds=float(last[j]*.02),
                    failure_terms=reasons[j], first_failure_frame=first_failure[j],
                    protocol='diagnostic: simulation continues after the first failure' if args.continue_after_failure else 'stop at first failure',
                    policy_sha256=sha256(args.checkpoint), motion_sha256=row['motion_sha256'],
                    library_sha256=sha256(args.library), body_names=contract['body_names'],
                    state_writes_after_initialization=0, assistance_count=0,
                    evidence='own PPO policy physical attempt; no hidden resets; nominal' if not args.perturb else 'initialization perturbation only; no ongoing pushes',
                    metrics_on_observed_prefix={k:dict(mean=float(v.mean()),max=float(v.max())) for k,v in errors.items()})
                all_results.append(result)
                if not args.summary_only:
                    clip_dir.mkdir()
                    attempt=dict(reference_frame_indices=frames.tolist(), terminal_reason=reason,state_writes_after_initialization=0,assistance_count=0)
                    atomic_json(clip_dir/'evaluation.json', result)
                    atomic_json(clip_dir/'completion.json', {**completion_counts([attempt],len(motion['joint_pos'])),'attempt':attempt})
                    np.savez_compressed(clip_dir/'rollout.npz',reference_frame_indices=frames,**trace)
                    np.savez_compressed(clip_dir/'errors.npz',**errors)
            atomic_json(args.output/'results.json',dict(checkpoint_sha256=sha256(args.checkpoint),library_sha256=sha256(args.library),
                attempts=len(all_results),completions=sum(r['complete'] for r in all_results),rows=all_results,
                perturbation_protocol='original pose/velocity/joint reset ranges, fixed phase0; seeds0,1,2' if args.perturb else 'nominal phase0; one attempt per clip'))
    finally:
        wrapped.close()

if __name__ == '__main__':
    main()
