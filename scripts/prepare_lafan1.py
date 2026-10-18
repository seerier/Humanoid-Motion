#!/usr/bin/env python3
"""Batch LAFAN1 -> G1 reference library (CPU; parallel over recordings).

1. GMR retargets each BVH recording once, frame 0 to the end, warm-started frame to frame; frame 0 may iterate
   to convergence. 2. One vertical ground offset per recording, estimated from human-stance soles. 3. 6 s
   windows (180 frames at 30 Hz) are cut afterwards, starting at frame 300. 4. A symbolic audit with thresholds
   fixed in advance rejects bad windows; no human in the loop.
5. Actor x category stratified split: one contiguous test block (~15 %) and validation block (~5 %) per
   recording, each with a one-window gap. obstacles/push* are retargeted and audited but kept out of all splits.
Every stage caches its outputs on disk; rerunning resumes. The full dataset is hours of CPU time: run it on
a many-core machine (e.g. a cluster node) with --workers.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from humanoid_motion.contracts import sha256
from humanoid_motion.retarget.reference import prepare

OUT = ROOT / 'outputs/lafan1'
URDF = ROOT / 'third_party/whole_body_tracking/audited_asset/unitree_description/urdf/g1/main.urdf'
# Windows held out from every split because earlier experiments used them as unseen test clips.
EXCLUDED = {'walk1_subject1': [(600, 780)], 'walk1_subject2': [(300, 480)], 'dance1_subject1': [(300, 480)]}
NAME = re.compile(r'([A-Za-z]+?)(\d+)_subject([1-5])\.bvh')
CONTEXT = ('obstacles', 'push', 'pushAndFall', 'pushAndStumble')  # need scenery / external forces
GMR_XML = ROOT / 'third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml'
WINDOW, FIRST_START = 180, 300   # source frames at 30 Hz
SEED = 20261013
# Audit thresholds, fixed before any result was seen.
PENETRATION_M, FLOAT_M, FLOAT_FRAMES = .05, .05, 15
TRUNK_DEG, FLIP_ROBOT_M, FLIP_HUMAN_M = 25., .08, .02
JOINT_SPEED_RAD_S, PIN_TOL_RAD, PIN_FRACTION, PIN_JOINTS = 40., 1e-3, .9, 3
STANCE_RESIDUAL_M = .02
FLIP_PAIRS = {'left_elbow_link': 'LeftForeArm', 'right_elbow_link': 'RightForeArm', 'left_knee_link': 'LeftLeg',
              'right_knee_link': 'RightLeg', 'left_wrist_yaw_link': 'LeftHand', 'right_wrist_yaw_link': 'RightHand'}


def recordings() -> list[dict]:
    rows = []
    with zipfile.ZipFile(ROOT / 'data/lafan1.zip') as archive:
        for member in sorted(archive.namelist()):
            match = NAME.fullmatch(Path(member).name)
            if not match:
                continue
            raw = archive.read(member)
            path = ROOT / 'data' / Path(member).name
            if path.exists() and sha256(path) != hashlib.sha256(raw).hexdigest():
                raise ValueError(f'Source mismatch: {path.name}')
            if not path.exists():
                path.write_bytes(raw)
            rows.append(dict(stem=path.stem, source=path.name, source_sha256=sha256(path), category=match.group(1),
                             subject=int(match.group(3)), frames=int(re.search(rb'Frames:\s*(\d+)', raw).group(1))))
    if len(rows) != 77:
        raise ValueError(f'Expected 77 LAFAN1 recordings, found {len(rows)}')
    return rows


def retarget_recording(rec: dict, root: Path, velocity_limit: bool, max_frames: int | None) -> dict:
    out = root / rec['stem'] / 'gmr'
    if not (out / 'diagnostics.json').exists():
        cfg = json.loads((ROOT / 'configs/walk1_s1.json').read_text())
        stop = min(rec['frames'], max_frames or rec['frames'])
        cfg.update(source_file='data/' + rec['source'], frame_start=0, frame_end_exclusive=stop, first_frame_max_iter=200,
                   use_velocity_limit=velocity_limit, output_dir=str(out.relative_to(ROOT)),
                   selection_reason='Whole recording from frame 0 (warm-started IK); windows are cut afterwards.')
        out.parent.mkdir(parents=True, exist_ok=True)
        config = out.parent / 'config.json'
        config.write_text(json.dumps(cfg, indent=2))
        with (out.parent / 'retarget.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'scripts/retarget_clip.py'), '--config', str(config), '--output', str(out)],
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
    d = json.loads((out / 'diagnostics.json').read_text())
    return dict(stem=rec['stem'], gmr=str(out.relative_to(ROOT)), frames=d['frames_saved'],
                seconds=d['execution']['retarget_and_fk_seconds'])


def ground_offset(ref: dict) -> dict:
    """Median lowest-sole height over human-stance frames; one constant for the whole recording."""
    soles = ref['foot_collision_bottom_z'].min(axis=2)          # frames x 2 feet
    stance = ref['stance_mask'].astype(bool)
    values = soles[stance]
    floor = float(np.median(values)) if values.size else float(np.percentile(soles.min(axis=1), 5))
    return dict(offset_m=floor, stance_samples=int(values.size), fallback=not values.size)


def run_lengths(mask: np.ndarray) -> int:
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def trunk_mismatch_deg(ref: dict) -> np.ndarray:
    sn, bn = list(ref['source_names']), list(ref['body_names'])
    v = ref['source_positions'][:, sn.index('Spine2')] - ref['source_positions'][:, sn.index('Hips')]
    w = ref['body_positions'][:, bn.index('torso_link')] - ref['body_positions'][:, bn.index('pelvis')]
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((v * w).sum(1), -1, 1)))


def flip_frames(ref: dict) -> np.ndarray:
    """Frames where a robot elbow/knee/wrist jumps although its human target barely moves (IK branch flip)."""
    bn, tn = list(ref['body_names']), list(ref['target_names'])
    robot = ref['body_positions'][:, [bn.index(b) for b in FLIP_PAIRS]]
    human = ref['target_positions'][:, [tn.index(h) for h in FLIP_PAIRS.values()]]
    dr = np.linalg.norm(np.diff(robot, axis=0), axis=2)
    dh = np.linalg.norm(np.diff(human, axis=0), axis=2)
    return ((dr > FLIP_ROBOT_M) & (dh < FLIP_HUMAN_M)).any(axis=1)


def slice_window(rec_gmr: Path, start: int, stop: int, offset: float, out: Path) -> None:
    """Write a GMR-run directory for one window, ground-corrected, that contracts.load_run accepts."""
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((rec_gmr / 'metadata.json').read_text())
    with np.load(rec_gmr / 'reference.npz') as data:
        ref = dict(data)
    frames = len(ref['qpos'])
    window = {}
    for key, value in ref.items():
        window[key] = value[start:stop].copy() if value.ndim and len(value) == frames else value
    window['qpos'][:, 2] -= offset
    window['body_positions'][..., 2] -= offset
    window['foot_collision_bottom_z'] -= offset
    window['timestamps'] = window['timestamps'] - window['timestamps'][0]
    np.savez_compressed(out / 'reference.npz', **window)
    csv = np.loadtxt(rec_gmr / 'motion.csv', delimiter=',')[start:stop]
    csv[:, 2] -= offset   # root_z column
    np.savetxt(out / 'motion.csv', csv, delimiter=',', fmt='%.17g')
    meta['source'].update(frame_start=start, frame_end_exclusive=stop)
    meta['outputs'] = {name: sha256(out / name) for name in ('reference.npz', 'motion.csv')}
    meta['derived_from'] = dict(recording_gmr=str(rec_gmr.relative_to(ROOT)), frames=[start, stop],
                                ground_offset_m=offset, recording_metadata_sha256=sha256(rec_gmr / 'metadata.json'),
                                rule='rows [start, stop) of the whole-recording retarget; root z lowered by ground_offset_m')
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')


def audit(window: dict, rec_diag: dict) -> dict:
    import mujoco
    from humanoid_motion.urdf_kinematics import UrdfKinematics
    gmr, prepared = ROOT / window['gmr'], ROOT / window['prepared']
    with np.load(gmr / 'reference.npz') as data:
        ref = dict(data)
    with np.load(prepared / 'generalized_reference.npz') as data:
        gen = dict(data)
    model = mujoco.MjModel.from_xml_path(str(GMR_XML))
    hinge = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    limits = model.jnt_range[hinge]
    q = ref['qpos'][:, 7:]
    excess = float(np.maximum(np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0).max())
    pinned = int(((np.minimum(np.abs(q - limits[:, 0]), np.abs(q - limits[:, 1])) < PIN_TOL_RAD).mean(0) >= PIN_FRACTION).sum())
    robot = UrdfKinematics(URDF)
    names = gen['joint_names'].tolist()
    urdf_excess = max(float(np.maximum(j['limits'][0] - gen['joint_pos'][:, names.index(j['name'])],
                                       gen['joint_pos'][:, names.index(j['name'])] - j['limits'][1]).max())
                      for j in robot.joints if j['kind'] != 'fixed')
    soles = ref['foot_collision_bottom_z'].min(axis=2)
    stance = ref['stance_mask'].astype(bool)
    floating = max(run_lengths(stance[:, j] & (soles[:, j] > FLOAT_M)) for j in range(2))
    trunk = trunk_mismatch_deg(ref)
    flips = flip_frames(ref)
    speed = float(np.abs(gen['joint_vel']).max())
    quat = float(np.abs(np.linalg.norm(ref['qpos'][:, 3:7], axis=1) - 1).max())
    stance_median = float(np.median(soles[stance])) if stance.any() else None
    metrics = dict(min_sole_m=float(soles.min()), floating_frames=floating, trunk_median_deg=float(np.median(trunk)),
                   trunk_p95_deg=float(np.percentile(trunk, 95)), flip_frames=int(flips.sum()), joint_speed_max=speed,
                   pinned_joints=pinned, ik_error_median=float(np.median(ref['ik_error'])),
                   stance_sole_median_m=stance_median, gmr_limit_excess_rad=excess, urdf_limit_excess_rad=urdf_excess)
    rules = [
        (bool(np.isfinite(ref['qpos']).all()) and quat <= 1e-6 and excess <= 1e-6 and urdf_excess <= 1e-6, 'numeric_or_joint_limit'),
        (rec_diag['robot_fk_independent_position_error_m']['max'] <= 1e-6 and rec_diag['source_fk_independent_max_error_m'] <= 1e-5
         and rec_diag['csv_named_roundtrip_max_error'] <= 1e-8, 'recording_fk_or_csv'),
        (metrics['min_sole_m'] >= -PENETRATION_M, 'penetration_gt_5cm'),
        (floating < FLOAT_FRAMES, 'floating_in_stance_gt_0.5s'),
        (metrics['trunk_median_deg'] <= TRUNK_DEG, 'trunk_mismatch_gt_25deg'),
        (metrics['flip_frames'] == 0, 'ik_flip'),
        (speed <= JOINT_SPEED_RAD_S, 'joint_speed_gt_40rad_s'),
        (pinned < PIN_JOINTS, 'joints_pinned_at_limit'),
    ]
    reasons = [name for ok, name in rules if not ok]
    return dict(metrics=metrics, rejection_reasons=reasons, usable=not reasons,
                stance_residual_ok=stance_median is None or abs(stance_median) <= STANCE_RESIDUAL_M)


def build_window(window: dict, rec_gmr: Path, offset: float, rec_diag: dict) -> dict:
    done = ROOT / window['gmr'] / 'audit.json'
    if done.exists():
        return {**window, **json.loads(done.read_text())}
    gmr, prepared = ROOT / window['gmr'], ROOT / window['prepared']
    if not (gmr / 'metadata.json').exists():
        slice_window(rec_gmr, window['start'], window['stop'], offset, gmr)
    if not (prepared / 'manifest.json').exists():
        prepare(gmr, prepared, URDF)
    result = audit(window, rec_diag)
    result.update(audit_version=5, prepared_manifest_sha256=sha256(prepared / 'manifest.json'), frames=299, fps=50)
    done.write_text(json.dumps(result, indent=2) + '\n')
    return {**window, **result}


def assign_split(rec: dict, windows: list[dict]) -> None:
    """One contiguous test block (~15 %) and validation block (~5 %) per recording, each padded by a gap window."""
    n = len(windows)
    for w in windows:
        w['split'] = 'excluded_context' if rec['category'] in CONTEXT else 'train'
    if rec['category'] in CONTEXT or n == 0:
        return
    test, val = max(1, round(.15 * n)), max(1, round(.05 * n))
    rng = np.random.default_rng(int(hashlib.sha256(f'{SEED}:{rec["stem"]}'.encode()).hexdigest()[:8], 16))
    if n < test + val + 4:   # only truncated smoke runs; every full LAFAN1 recording has >= 15 windows
        for w in windows:
            w['split_note'] = 'too short for a padded split; all train'
        return
    for _ in range(1000):
        a, b = int(rng.integers(0, n - test + 1)), int(rng.integers(0, n - val + 1))
        if b + val + 1 <= a or a + test + 1 <= b:   # at least one gap window between the blocks
            break
    else:
        raise RuntimeError(f'No split placement found: {rec["stem"]}')
    for block, start, length in (('test', a, test), ('validation', b, val)):
        for i in range(start, start + length):
            windows[i]['split'] = block
        for i in (start - 1, start + length):
            if 0 <= i < n and windows[i]['split'] == 'train':
                windows[i]['split'] = 'gap'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)))
    parser.add_argument('--recordings', nargs='*', help='Smoke: only these recording stems')
    parser.add_argument('--max-frames', type=int, help='Smoke: truncate recordings')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    started = time.monotonic()
    recs = recordings()
    if args.recordings:
        recs = [r for r in recs if r['stem'] in args.recordings]
    recs.sort(key=lambda r: -r['frames'])     # longest first: they bound the wall clock
    out = args.out
    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(retarget_recording, r, out / 'recordings', False, args.max_frames): r['stem'] for r in recs}
        retargeted = {}
        for i, job in enumerate(futures.as_completed(jobs), 1):
            retargeted[jobs[job]] = job.result()
            print(f'retargeted {i}/{len(jobs)} {jobs[job]} {retargeted[jobs[job]]["seconds"]:.0f}s', flush=True)
    print('STAGE retarget done', round(time.monotonic() - started), flush=True)

    windows, grounds, rec_diags = [], {}, {}
    for rec in recs:
        gmr = out / 'recordings' / rec['stem'] / 'gmr'
        with np.load(gmr / 'reference.npz') as data:
            grounds[rec['stem']] = ground_offset(dict(data))
        rec_diags[rec['stem']] = json.loads((gmr / 'diagnostics.json').read_text())
        rows = []
        for start in range(FIRST_START, retargeted[rec['stem']]['frames'] - WINDOW + 1, WINDOW):
            if any(start < b and start + WINDOW > a for a, b in EXCLUDED.get(rec['stem'], [])):
                continue
            wid = f"{rec['stem']}_f{start}_{start + WINDOW}"
            rows.append(dict(id=wid, source=rec['source'], source_sha256=rec['source_sha256'], start=start, stop=start + WINDOW,
                             category=rec['category'], subject=rec['subject'],
                             gmr=str((out / 'windows' / wid / 'gmr').relative_to(ROOT)),
                             prepared=str((out / 'windows' / wid / 'prepared').relative_to(ROOT))))
        assign_split(rec, rows)
        windows += rows
    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(build_window, w, out / 'recordings' / w['source'].removesuffix('.bvh') / 'gmr',
                            grounds[w['source'].removesuffix('.bvh')]['offset_m'], rec_diags[w['source'].removesuffix('.bvh')])
                for w in windows]
        built = []
        for i, job in enumerate(futures.as_completed(jobs), 1):
            built.append(job.result())
            if i % 100 == 0 or i == len(jobs):
                print(f'windows {i}/{len(jobs)} elapsed {time.monotonic() - started:.0f}s', flush=True)
    built.sort(key=lambda w: w['id'])
    print('STAGE windows done', round(time.monotonic() - started), flush=True)

    summary, reasons = {}, {}
    for w in built:
        cell = summary.setdefault(w['split'], {}).setdefault(w['category'], dict(windows=0, usable=0))
        cell['windows'] += 1
        cell['usable'] += bool(w['usable'])
        for r in w['rejection_reasons']:
            reasons[r] = reasons.get(r, 0) + 1
    minutes = {split: round(sum(c['usable'] for c in cats.values()) * 6 / 60, 1) for split, cats in summary.items()}
    result = dict(schema='humanoid_motion.lafan1_library.v1', complete=not (args.recordings or args.max_frames),
                  workers=args.workers,
                  wall_seconds=time.monotonic() - started, thresholds=dict(
                      penetration_m=PENETRATION_M, floating_m=FLOAT_M, floating_frames_30hz=FLOAT_FRAMES, trunk_deg=TRUNK_DEG,
                      flip_robot_m=FLIP_ROBOT_M, flip_human_m=FLIP_HUMAN_M, joint_speed_rad_s=JOINT_SPEED_RAD_S,
                      pinned=dict(tolerance_rad=PIN_TOL_RAD, fraction=PIN_FRACTION, joints=PIN_JOINTS)),
                  ground=grounds, windows=len(built), usable=sum(w['usable'] for w in built), usable_minutes_by_split=minutes,
                  rejection_reasons=reasons, stance_residual_failures=sum(not w['stance_residual_ok'] for w in built),
                  by_split_category=summary, split_rule=dict(
                      seed=SEED, test_fraction=.15, validation_fraction=.05, gap_windows=1, excluded_categories=list(CONTEXT)),
                  records=built)
    out.mkdir(parents=True, exist_ok=True)
    name = 'library.json' if result['complete'] else 'smoke.json'
    (out / name).write_text(json.dumps(result, indent=2) + '\n')
    if result['complete']:
        for split in ('train', 'validation', 'test'):
            clips = [w for w in built if w['split'] == split and w['usable']]
            (out / f'{split}.json').write_text(json.dumps(dict(
                schema='humanoid_motion.library.prepared.v1', seed=SEED, group=split, clips=clips,
                selection='whole-recording GMR, per-recording ground offset, symbolic audit, padded block split'), indent=2) + '\n')
        (out / 'frozen.json').write_text(json.dumps({s: sha256(out / f'{s}.json') for s in ('train', 'validation', 'test')}, indent=2) + '\n')
    print('DONE', out / name, json.dumps({k: result[k] for k in ('windows', 'usable', 'usable_minutes_by_split', 'rejection_reasons', 'wall_seconds')}), flush=True)


if __name__ == '__main__':
    main()
