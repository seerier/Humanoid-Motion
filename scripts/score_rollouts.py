#!/usr/bin/env python3
"""Score saved rollouts of any tracking backend with one offline function (humanoid_motion.evaluation.score_attempt).

Reads <evaluation>/<clip>/rollout.npz (recorded world states, written by scripts/evaluate.py or
humanoid_motion.tracking.supermimic.evaluate) and the reference motions of the Isaac library, and writes
<evaluation>/scores_<criterion>.json in the results.json row format, so scripts/summarize_results.py can read it:

    python scripts/score_rollouts.py --library outputs/lafan1/runtime/test.json --evaluation outputs/eval/test
    python scripts/summarize_results.py C=outputs/eval/test/scores_beyondmimic.json ...

--check compares the offline scores with the evaluator's own online results.json (same criterion, beyondmimic).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from humanoid_motion import ROOT
from humanoid_motion.evaluation import CRITERIA, score_attempt
from humanoid_motion.library import atomic_json


def score_directory(library: Path, evaluation: Path, criterion: str, full: bool = False,
                    allow_missing: bool = False) -> dict:
    """Score every clip of the library; `full` uses rollout_full.npz (recorded past failures). A clip without a
    recording is an error unless `allow_missing` (then it is listed in the output and left out of the counts)."""
    manifest = json.loads(library.read_text())
    rows, missing = [], []
    for row in manifest['clips']:
        clip_dir = evaluation / row['id']
        path = clip_dir / ('rollout_full.npz' if full else 'rollout.npz')
        if not path.exists():
            missing.append(row['id'])
            continue
        motion = ROOT / row['motion']
        names = json.loads(motion.with_suffix('.manifest.json').read_text())['body_names']
        with np.load(motion) as ref, np.load(path) as rollout:
            if not np.array_equal(rollout['reference_frame_indices'], np.arange(len(rollout['body_pos_w']))):
                raise ValueError(f'{path}: recording is not time-matched to reference frames 0..n-1')
            if 'body_names' in rollout.files and [str(n) for n in rollout['body_names']] != names:
                raise ValueError(f'{path}: body order differs from the library')
            if criterion != 'beyondmimic' and len(rollout['body_pos_w']) < len(ref['body_pos_w']):
                raise ValueError(f'{path}: recording stopped early (at a BeyondMimic failure); score the full '
                                 'recording (--full, or an evaluation run with --continue-after-failure)')
            result = score_attempt(ref['body_pos_w'].astype(np.float64), ref['body_quat_w'].astype(np.float64),
                                   rollout['body_pos_w'].astype(np.float64),
                                   rollout['body_quat_w'].astype(np.float64), names, criterion)
        rows.append(dict(clip_id=row['id'], category=row['category'], split=row['split'], **result))
    if missing and not allow_missing:
        raise FileNotFoundError(f'{len(missing)} of {len(manifest["clips"])} clips have no recording in {evaluation} '
                                f'(e.g. {missing[0]}); a summary-only or perturbation run, or rollout_full.npz of an '
                                'Isaac run? Pass --allow-missing to score the rest')
    return dict(schema='humanoid_motion.offline_scores.v1', criterion=criterion, library=str(library),
                evaluation=str(evaluation), recording='rollout_full.npz' if full else 'rollout.npz',
                attempts=len(rows), completions=sum(r['complete'] for r in rows), missing=missing, rows=rows)


def check(scores: dict, evaluation: Path) -> dict:
    online = {r['clip_id']: r for r in json.loads((evaluation / 'results.json').read_text())['rows']}
    mismatched, metric_diff = [], 0.
    for row in scores['rows']:
        other = online[row['clip_id']]
        if row['complete'] != other['complete'] or row['first_failure_frame'] != other['first_failure_frame']:
            mismatched.append(dict(clip=row['clip_id'], offline=(row['complete'], row['first_failure_frame']),
                                   online=(other['complete'], other['first_failure_frame'])))
        for key, value in row['metrics_on_observed_prefix'].items():
            metric_diff = max(metric_diff, abs(value['mean'] - other['metrics_on_observed_prefix'][key]['mean']))
    return dict(clips=len(scores['rows']), online_completions=sum(online[r['clip_id']]['complete'] for r in scores['rows']),
                offline_completions=scores['completions'], mismatched=mismatched, max_metric_mean_difference=metric_diff)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--library', type=Path, required=True, help='Isaac library <split>.json (references)')
    parser.add_argument('--evaluation', type=Path, required=True, help='Directory with <clip>/rollout.npz')
    parser.add_argument('--criterion', choices=sorted(CRITERIA), default='beyondmimic')
    parser.add_argument('--full', action='store_true', help='Score rollout_full.npz (recorded past the first failure)')
    parser.add_argument('--check', action='store_true', help='Compare with the online results.json of a nominal run '
                        '(a --continue-after-failure run reports metrics over the whole clip, not the scored prefix)')
    parser.add_argument('--allow-missing', action='store_true', help='Score the clips that have a recording')
    args = parser.parse_args()
    scores = score_directory(args.library, args.evaluation, args.criterion, args.full, args.allow_missing)
    if args.check:
        scores['online_check'] = check(scores, args.evaluation)
    out = args.evaluation / f'scores_{args.criterion}{"_full" if args.full else ""}.json'
    atomic_json(out, scores)
    done = [r for r in scores['rows'] if r['complete']]
    mean = lambda key: 100 * float(np.mean([r['metrics_on_observed_prefix'][key]['mean'] for r in done])) if done else float('nan')
    print(f'{out}: {scores["completions"]}/{scores["attempts"]} complete ({args.criterion}); on completed clips '
          f'aligned body {mean("aligned_body_position_m"):.1f} cm, world root {mean("world_root_position_m"):.1f} cm')
    if args.check:
        print(json.dumps(scores['online_check'], indent=2))


if __name__ == '__main__':
    main()
