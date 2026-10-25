#!/usr/bin/env python3
"""MorphMimic's normalised return of saved sim2sim rollouts: the SuperMimic paper's benchmark definition.

Per control step t = 1..T-1, MorphMimic's analytic reward head (morphmimic.model.rewards_torch.tracking_reward:
mjlab's tracking terms plus action-rate and joint-limit penalties, each times dt) of the recorded robot state against
reference frame t, with the policy's actions; summed up to and including the first failure of MorphMimic's own
training termination (anchor height or tilt, the 'anchor' criterion) and divided by the best possible return
R_MAX * (T - 1), as deploy/sim2sim/clip_playback.ClipPlayback reports it. This is not the per-step average that the
training log prints as `nret`. Reads <evaluation>/<clip>/rollout_full.npz and evaluation.json written by
humanoid_motion.tracking.supermimic.evaluate; writes <evaluation>/returns.json. Runs in MorphMimic's environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from humanoid_motion.library import atomic_json


def soft_joint_limits():
    """Canonical G1 soft joint limits [29, 2] as mjlab sets them: range midpoint +- factor * half range."""
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
    from morphmimic.data.seed_convert import make_model
    from morphmimic.robots import get_robot
    spec, model = get_robot('g1'), make_model('g1')
    factor = get_g1_robot_cfg().articulation.soft_joint_pos_limit_factor
    ranges = np.array([model.joint(name).range for name in spec.joint_names])
    mid, half = ranges.mean(axis=1), (ranges[:, 1] - ranges[:, 0]) / 2
    return np.stack([mid - factor * half, mid + factor * half], axis=1), float(factor)


def clip_return(rollout: dict, end: int, limits) -> tuple[float, dict]:
    """Normalised return of one rollout whose episode ends at frame `end` (failure frame or the last frame)."""
    import torch
    from morphmimic.constants import R_MAX
    from morphmimic.model.rewards_torch import tracking_reward
    state = torch.as_tensor(rollout['mm_state'][1:end + 1], dtype=torch.float64)
    ref = torch.as_tensor(rollout['mm_reference'][1:end + 1], dtype=torch.float64)
    actions = np.asarray(rollout['actions'], np.float64)
    action = torch.as_tensor(actions[:end])
    prev = torch.as_tensor(np.concatenate([np.zeros((1, actions.shape[1])), actions[:end - 1]]))
    total, terms = tracking_reward(state, ref, action, prev, torch.as_tensor(limits))
    frames = len(rollout['mm_reference'])
    return float(total.sum()) / (R_MAX * (frames - 1)), {k: float(v.sum()) for k, v in terms.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--evaluation', type=Path, required=True)
    args = parser.parse_args()
    limits, factor = soft_joint_limits()
    rows = []
    for path in sorted(args.evaluation.glob('*/rollout_full.npz')):
        evaluation = json.loads((path.parent / 'evaluation.json').read_text())
        with np.load(path) as z:
            rollout = {k: z[k] for k in ('mm_state', 'mm_reference', 'actions')}
        failure = evaluation['anchor_criterion']['first_failure_frame']
        end = len(rollout['mm_reference']) - 1 if failure is None else failure
        value, terms = clip_return(rollout, end, limits)
        rows.append(dict(clip_id=evaluation['clip_id'], category=evaluation['category'], normalised_return=value,
                         failed=failure is not None, terms=terms))
    values = np.array([r['normalised_return'] for r in rows])
    summary = dict(schema='humanoid_motion.morphmimic_returns.v1', evaluation=str(args.evaluation),
                   definition=__doc__.split('\n\n')[0], soft_joint_limit_factor=factor, clips=len(rows),
                   mean_normalised_return=float(values.mean()) if len(values) else None, rows=rows)
    atomic_json(args.evaluation / 'returns.json', summary)
    if not rows:
        raise SystemExit(f'{args.evaluation}: no rollout_full.npz with MorphMimic states')
    print(f'{args.evaluation}: mean normalised return {summary["mean_normalised_return"]:.3f} over {len(rows)} clips')


if __name__ == '__main__':
    main()
