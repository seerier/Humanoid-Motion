"""Write the GMR retargeting config (humanoid_motion.retarget.gmr) for one adapter npz (production or reconciliation).

Production: human_height mesh_stature (the subject's shaped-mesh stature instead of GMR's 1.66 + 0.1 b0); the IK floor
constraint floor_0mm (humanoid_motion.retarget.gmr add_floor_limit: kind floor, gain 0.85, minimum distance 0,
detection 0.1 m) at floor_height 0, which is where the adapter put the ground; first_frame_max_iter 200 and the stance
proxy numbers of the LAFAN1 config (configs/walk1_s1.json).
Reconciliation (gate 6): exactly what GMR's scripts/gvhmr_to_robot.py does: human_height gmr_beta0 (1.66 + 0.1 b0),
no floor constraint, no first-frame iteration cap change, offset_to_ground off. Solver daqp / damping 0.5 are GMR's
own defaults in both. The adapter npz for the reconciliation run must itself be made with --ground none --betas first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FLOOR_0MM = {'floor_height_m': 0.0, 'kind': 'floor', 'gain': 0.85, 'minimum_distance_m': 0.0, 'detection_distance_m': 0.1}


def build(npz: Path, root: Path, mode: str, output_dir: Path, reason: str) -> dict:
    with np.load(npz, allow_pickle=False) as f:
        frames = int(f['pose_body'].shape[0])
    cfg = {
        'source_kind': 'smplx', 'source_file': str(npz.resolve().relative_to(root.resolve())), 'target_fps': 30,
        'frame_start': 0, 'frame_end_exclusive': frames, 'robot': 'unitree_g1', 'solver': 'daqp', 'damping': 0.5,
        'use_velocity_limit': False, 'offset_to_ground': False,
        'stance_height_above_p05_m': 0.04, 'stance_speed_m_s': 0.2,
        'output_dir': str(output_dir.resolve().relative_to(root.resolve())), 'selection_reason': reason,
    }
    if mode == 'production':
        cfg.update({'human_height': {'mode': 'mesh_stature'}, 'first_frame_max_iter': 200, 'floor_limit': dict(FLOOR_0MM)})
    elif mode == 'reconcile':
        cfg.update({'human_height': {'mode': 'gmr_beta0'}})
    else:
        raise ValueError(mode)
    return cfg


def main(argv=None) -> None:
    from humanoid_motion import ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--npz', type=Path, required=True)
    parser.add_argument('--mode', choices=('production', 'reconcile'), required=True)
    parser.add_argument('--output_dir', type=Path, required=True, help='retarget run directory (must not exist yet)')
    parser.add_argument('--config_out', type=Path, required=True)
    parser.add_argument('--reason', default='video clip (GVHMR -> SMPL-X adapter)')
    args = parser.parse_args(argv)
    cfg = build(args.npz, ROOT, args.mode, args.output_dir, args.reason)
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(cfg, indent=2) + '\n')
    print(json.dumps(cfg, indent=2))


if __name__ == '__main__':
    main()
