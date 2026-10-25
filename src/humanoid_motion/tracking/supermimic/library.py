#!/usr/bin/env python3
"""Prepared 50 Hz references -> MorphMimic motion library: one npz per clip, manifest.csv and a subset yaml.

Both trackers read the same frames. Root pose and joint angles come from prepared/generalized_reference.npz
(contracts.resample_csv, the frames the Isaac library is built from); body poses, finite-difference velocities
and clip statistics come from MorphMimic's own converter functions on mjlab's G1 model. MorphMimic's name and
feasibility filters are evaluated and recorded but do not drop clips, so the training set stays identical to the
Isaac one. Each clip is checked against the independent URDF forward kinematics of the same frames.

Run with MorphMimic's interpreter and this package on PYTHONPATH:
  third_party/morphmimic/.venv/bin/python -m humanoid_motion.tracking.supermimic.library \
    --split outputs/lafan1_v2/train.json --out outputs/v0.3/supermimic/library/lafan1_train
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from humanoid_motion import ROOT
from humanoid_motion.contracts import check_quaternions, named_indices
from humanoid_motion.library import atomic_json

FK_TOLERANCE_M = 1e-4


def morphmimic():
    from morphmimic.data import seed_convert
    from morphmimic.robots import get_robot
    return seed_convert, get_robot('g1')


def convert_reference(reference: Path, out: Path) -> tuple[dict, dict]:
    """Write MorphMimic's npz for one prepared reference; return (its clip_stats, the arrays)."""
    convert, spec = morphmimic()
    with np.load(reference, allow_pickle=False) as archive:
        names = [str(n) for n in archive['joint_names']]
        fps = float(np.asarray(archive['fps']).reshape(-1)[0])
        root_pos, root_quat, joint_pos = (np.asarray(archive[k], dtype=np.float64)
                                          for k in ('root_pos_w', 'root_quat_w', 'joint_pos'))
    if fps != convert.FPS:
        raise ValueError(f'{reference}: {fps} Hz, MorphMimic needs {convert.FPS} Hz')
    check_quaternions(root_quat)
    joint_pos = joint_pos[:, named_indices(names, list(spec.joint_names))]
    model = convert.make_model('g1')
    fk = convert.forward_kinematics(model, root_pos, root_quat, joint_pos,
                                    zmin_body_ids=convert.zmin_bodies(model, spec.zmin_exclude_regex),
                                    feet=convert.foot_geoms(model, spec.foot_geom_regex))
    # Same steps and keys as morphmimic.data.seed_convert.convert_clip after its CSV parsing and resampling.
    q = fk.body_quat_w
    body_quat_w = np.where(q[..., :1] < 0.0, -q, q)
    joint_vel, body_lin_vel_w, body_ang_vel_w = convert.finite_diff_velocities(
        joint_pos, fk.body_pos_w, body_quat_w, convert.FPS)
    f32 = np.float32
    motion = {'fps': np.array(convert.FPS, dtype=np.int64), 'joint_pos': joint_pos.astype(f32),
              'joint_vel': joint_vel.astype(f32), 'body_pos_w': fk.body_pos_w.astype(f32),
              'body_quat_w': body_quat_w.astype(f32), 'body_lin_vel_w': body_lin_vel_w.astype(f32),
              'body_ang_vel_w': body_ang_vel_w.astype(f32), 'body_names': np.array(convert.body_names(model)),
              'joint_names': np.array(spec.joint_names), 'z_min': fk.z_min.astype(f32),
              'foot_dist': fk.foot_dist.astype(f32), 'robot': np.array(spec.name),
              'root_body': np.array(spec.root_body)}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.tmp')
    with open(tmp, 'wb') as stream:
        np.savez(stream, **motion)
    os.replace(tmp, out)
    return convert.clip_stats(out.stem, motion, 'g1'), dict(motion, body_pos_f64=fk.body_pos_w,
                                                            body_quat_f64=body_quat_w)


def urdf_difference(motion: dict, independent: Path) -> dict:
    """Largest tracked-body pose difference between MorphMimic's FK and the independent URDF FK, same frames."""
    from morphmimic.constants import TRACKED_BODY_NAMES
    names = [str(n) for n in motion['body_names']]
    with np.load(independent, allow_pickle=False) as archive:
        urdf_names = [str(n) for n in archive['body_names']]
        urdf_pos, urdf_quat = archive['body_pos_w'], archive['body_quat_w']
    if len(urdf_pos) != len(motion['body_pos_f64']):
        raise ValueError(f'{independent}: {len(urdf_pos)} frames, expected {len(motion["body_pos_f64"])}')
    ours = [names.index(n) for n in TRACKED_BODY_NAMES]
    theirs = [urdf_names.index(n) for n in TRACKED_BODY_NAMES]
    position = np.linalg.norm(motion['body_pos_f64'][:, ours] - urdf_pos[:, theirs], axis=-1)
    dot = np.abs(np.sum(motion['body_quat_f64'][:, ours] * urdf_quat[:, theirs], axis=-1))
    angle = 2 * np.arccos(np.clip(dot, 0, 1))
    worst = int(np.argmax(position.max(axis=0)))
    return dict(position_m=float(position.max()), angle_rad=float(angle.max()),
                worst_body=TRACKED_BODY_NAMES[worst])


def build(split: Path, out: Path, root: Path = ROOT, limit: int | None = None) -> dict:
    """Convert every clip of a prepared split manifest (train/validation/test.json) into `out`."""
    import pandas as pd
    from morphmimic.data.filters import name_filter_reject
    from morphmimic.data.manifest import MANIFEST_COLUMNS, build_subset, load_manifest
    source = json.loads(split.read_text())
    clips = source['clips'][:limit]
    if len({row['id'] for row in clips}) != len(clips):
        raise ValueError('Duplicate clip IDs')
    if (out / 'manifest.csv').exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    rows, checks = [], []
    for row in clips:
        prepared = root / row['prepared']
        session = Path(row['source']).stem
        rel = f'{session}/{row["id"]}.npz'
        stats, motion = convert_reference(prepared / 'generalized_reference.npz', out / rel)
        check = dict(id=row['id'], **urdf_difference(motion, prepared / 'independent_urdf.npz'))
        checks.append(check)
        rows.append(dict(name=row['id'], path=rel, session=session, frames=stats['frames'],
                         seconds=stats['seconds'], name_reject=name_filter_reject(row['id']) or '',
                         feas_reject=stats['feas_reject'] or '', category=row.get('category', ''),
                         package=source.get('group', split.stem)))
    df = pd.DataFrame(rows)[list(MANIFEST_COLUMNS)]
    df.to_csv(out / 'manifest.csv', index=False)
    # Every clip goes into the subset (MorphMimic's filters are recorded, not applied); their subset format.
    subset = out / 'subsets' / f'{split.stem}.yaml'
    build_subset(load_manifest(str(out / 'manifest.csv')), None, 0, str(subset))
    worst = max(checks, key=lambda c: c['position_m'])
    summary = dict(schema='humanoid_motion.supermimic_library.v1', split=str(split), subset=str(subset),
                   clips=len(df), frames=int(df['frames'].sum()), hours=float(df['seconds'].sum() / 3600),
                   name_filter_would_reject=int((df['name_reject'] != '').sum()),
                   feasibility_filter_would_reject=int((df['feas_reject'] != '').sum()),
                   rejected_examples=df[(df['name_reject'] != '') | (df['feas_reject'] != '')]
                   [['name', 'name_reject', 'feas_reject']].head(20).to_dict('records'),
                   urdf_fk_max_position_m=worst['position_m'], urdf_fk_worst=worst,
                   urdf_fk_max_angle_rad=max(c['angle_rad'] for c in checks),
                   urdf_fk_tolerance_m=FK_TOLERANCE_M, urdf_fk_passed=worst['position_m'] < FK_TOLERANCE_M)
    atomic_json(out / 'build_summary.json', summary)
    atomic_json(out / 'fk_checks.json', dict(clips=checks))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--split', type=Path, required=True, help='Prepared split manifest, e.g. train.json')
    parser.add_argument('--out', type=Path, required=True, help='New library directory')
    parser.add_argument('--root', type=Path, default=ROOT, help='Directory the manifest paths are relative to')
    parser.add_argument('--limit', type=int, help='Convert only the first N clips (smoke test)')
    args = parser.parse_args()
    summary = build(args.split, args.out, args.root, args.limit)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rejected_examples'}, indent=2))
    if not summary['urdf_fk_passed']:
        raise SystemExit(f'MorphMimic FK differs from the URDF FK by {summary["urdf_fk_max_position_m"]:.2e} m')


if __name__ == '__main__':
    main()
