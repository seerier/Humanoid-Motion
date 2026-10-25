"""Quality gate for one clip: four numbers, a pass/fail each, nothing changed, nothing dropped.

Inputs are what the pipeline already wrote: the retarget run directory of humanoid_motion.retarget.gmr (reference.npz,
diagnostics.json, metadata.json) and the adapter's npz. Each check names what it is about (G1 or the human):

  g1_joint_limits    G1: a (frame, joint) sample is "at the limit" when it lies outside the central 98 % of the joint's
                     range in the GMR G1 model (PHUMA / LIMMT keep 98 %); the number is the fraction of frames with any
                     such sample. Pass: <= 1 % of frames.
  g1_foot_penetration G1: fraction of frames where the lowest sole collision sphere is more than 2.5 cm below z = 0
                     (PHUMA's 2.5 cm contact tolerance, applied to the floor the adapter aligned). Pass: <= 1 %.
  human_pelvis_height human: SMPL-X pelvis joint height over the clip must stay inside 0.6-1.5 m (PHUMA). Pass: every frame.
  human_root_jerk    human: third time derivative of the pelvis joint after PHUMA's 4th-order Butterworth low-pass at 3 Hz
                     (the filter is used for the measurement only; the motion itself is not filtered). Pass: max < 50 m/s^3.
                     The unfiltered maximum is reported next to it.

Thresholds are this project's choices, built on the published values cited above (PHUMA, LIMMT); a failed gate routes
the clip (reported as "did not pass"), it never edits or removes it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

JOINT_LIMIT_FRACTION = 0.98
JOINT_LIMIT_MAX_FRAMES = 0.01
PENETRATION_M = 0.025
PENETRATION_MAX_FRAMES = 0.01
PELVIS_RANGE_M = (0.6, 1.5)
JERK_MAX = 50.0
JERK_FILTER_HZ = 3.0


def butter_lowpass(x: np.ndarray, fps: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    b, a = butter(order, cutoff_hz / (0.5 * fps))
    return filtfilt(b, a, x, axis=0)


def jerk(positions: np.ndarray, fps: float) -> np.ndarray:
    dt = 1.0 / fps
    v = np.gradient(positions, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)
    return np.linalg.norm(j, axis=-1)


def joint_ranges(xml_path: Path, joint_names: list[str]) -> np.ndarray:
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    return np.array([m.jnt_range[m.joint(n).id] for n in joint_names])


def evaluate(run_dir: Path, root: Path) -> dict:
    meta = json.loads((run_dir / 'metadata.json').read_text())
    diag = json.loads((run_dir / 'diagnostics.json').read_text())
    with np.load(run_dir / 'reference.npz', allow_pickle=False) as f:
        qpos, source_positions, names = f['qpos'], f['source_positions'], f['source_names'].tolist()
        sole_z = f['foot_collision_bottom_z']            # (frames, 2 feet, spheres)
        timestamps = f['timestamps']
    fps = 1.0 / float(np.mean(np.diff(timestamps)))
    frames = len(qpos)
    joint_names = meta['model']['joint_names']
    xml = root / meta['model']['xml_path']
    ranges = joint_ranges(xml, joint_names)
    mid, half = ranges.mean(axis=1), 0.5 * (ranges[:, 1] - ranges[:, 0])
    q = qpos[:, 7:]
    at_limit = np.abs(q - mid) > JOINT_LIMIT_FRACTION * half          # (frames, joints)
    limit_frames = at_limit.any(axis=1)
    worst = {joint_names[j]: int(at_limit[:, j].sum()) for j in np.flatnonzero(at_limit.sum(axis=0))}
    lowest = sole_z.min(axis=(1, 2))
    penetration_frames = lowest < -PENETRATION_M
    pelvis = source_positions[:, names.index('pelvis')]
    pelvis_z = pelvis[:, 2]
    raw_jerk = jerk(pelvis, fps)
    filtered_jerk = jerk(butter_lowpass(pelvis, fps, JERK_FILTER_HZ), fps) if frames > 15 else raw_jerk
    checks = {
        'g1_joint_limits': {
            'subject': 'G1', 'rule': f'frames with any joint outside the central {JOINT_LIMIT_FRACTION:.0%} of its range',
            'value': float(limit_frames.mean()), 'threshold': JOINT_LIMIT_MAX_FRAMES, 'frames': int(limit_frames.sum()),
            'joints_hit': worst, 'pass': bool(limit_frames.mean() <= JOINT_LIMIT_MAX_FRAMES)},
        'g1_foot_penetration': {
            'subject': 'G1', 'rule': f'frames with the lowest sole sphere more than {PENETRATION_M * 100:.1f} cm below z = 0',
            'value': float(penetration_frames.mean()), 'threshold': PENETRATION_MAX_FRAMES,
            'frames': int(penetration_frames.sum()), 'deepest_m': float(lowest.min()),
            'pass': bool(penetration_frames.mean() <= PENETRATION_MAX_FRAMES)},
        'human_pelvis_height': {
            'subject': 'human', 'rule': f'pelvis joint height within {PELVIS_RANGE_M} m on every frame',
            'value': [float(pelvis_z.min()), float(pelvis_z.max())], 'threshold': list(PELVIS_RANGE_M),
            'frames_outside': int(((pelvis_z < PELVIS_RANGE_M[0]) | (pelvis_z > PELVIS_RANGE_M[1])).sum()),
            'pass': bool(pelvis_z.min() >= PELVIS_RANGE_M[0] and pelvis_z.max() <= PELVIS_RANGE_M[1])},
        'human_root_jerk': {
            'subject': 'human', 'rule': f'max pelvis jerk after a 4th-order Butterworth {JERK_FILTER_HZ} Hz low-pass < {JERK_MAX} m/s^3',
            'value': float(filtered_jerk.max()), 'threshold': JERK_MAX, 'unfiltered_max': float(raw_jerk.max()),
            'unfiltered_p99': float(np.percentile(raw_jerk, 99)), 'pass': bool(filtered_jerk.max() < JERK_MAX)},
    }
    result = {
        'run_dir': str(run_dir), 'frames': frames, 'fps': fps, 'checks': checks,
        'passed': all(c['pass'] for c in checks.values()), 'n_pass': sum(c['pass'] for c in checks.values()),
        'score': float(np.mean([c['pass'] for c in checks.values()])),
        'retarget_diagnostics': {k: diag.get(k) for k in ('joint_limit_violations_gt_1e-6_rad', 'joint_limit_max_excess_rad',
                                                          'ik_task_error_norm', 'root_z_range_m', 'feet', 'floor_limit')},
        'note': 'routing only: a failed gate marks the clip, it never edits or drops it',
    }
    return result


def main(argv=None) -> None:
    from humanoid_motion import ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dir', type=Path, required=True, help='retarget run directory (reference.npz, metadata.json)')
    parser.add_argument('--out', type=Path, default=None, help='json to write (default <run_dir>/gate.json)')
    args = parser.parse_args(argv)
    result = evaluate(args.run_dir, ROOT)
    out = args.out or args.run_dir / 'gate.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    for name, c in result['checks'].items():
        print(f"{name:22s} {c['subject']:6s} {'PASS' if c['pass'] else 'FAIL'}  value={c['value']}  threshold={c['threshold']}")
    print('passed' if result['passed'] else 'did not pass', f"({result['n_pass']}/4) -> {out}")


if __name__ == '__main__':
    main()
