"""Process reconciliation: our path (adapter -> sources.load_amass_smplx -> humanoid_motion.retarget.gmr) against
GMR's own GVHMR path, run headless in the call order of GMR's scripts/gvhmr_to_robot.py:

    load_gvhmr_pred_file(pt, body_models) -> get_gvhmr_data_offline_fast(tgt_fps=30) -> GeneralMotionRetargeting(
        actual_human_height, src_human='smplx', tgt_robot='unitree_g1').retarget(frame) per frame

Both sides use GMR's unmodified IK, so with identical input frames the joint angles must agree to solver precision.
Verdict thresholds: joint angles < 1e-3 rad per frame, root position < 1 mm.

Second check: the adapter npz read back through load_amass_smplx gives SMPL-X joints; GVHMR's own body model
(scripts/dump_gvhmr_joints.py, run in the GVHMR environment) gives joints in GVHMR's y-up frame; after
GMR's rotation matrix the two must differ only by numerical precision.

Known upstream detail: GMR's load_gvhmr_pred_file pads betas to 16 but builds smplx with the default 10, which smplx
rejects. The fallback replicates its lines with 10 betas and says so in the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

Y_UP_TO_Z_UP = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def gmr_reference(pt: Path, body_models: Path, gmr_root: Path) -> tuple[np.ndarray, dict]:
    """qpos per frame from GMR's own GVHMR path; dict says which loader produced the frames."""
    sys.path.insert(0, str(gmr_root))
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.utils import smpl as gmr_smpl
    note = {}
    try:
        smplx_data, body_model, smplx_output, height = gmr_smpl.load_gvhmr_pred_file(str(pt), str(body_models))
        note['loader'] = 'general_motion_retargeting.utils.smpl.load_gvhmr_pred_file (unmodified)'
    except Exception as error:   # betas padded to 16 against a 10-beta model, see module docstring
        import smplx, torch
        note['loader'] = ('load_gvhmr_pred_file failed; its lines replayed with 10 betas and batch_size=N (same gender, '
                          'frame rate, height formula, same forward call): ' + repr(error)[:300])
        pred = torch.load(pt, map_location='cpu', weights_only=False)
        g = pred['smpl_params_global']
        smplx_data = {'pose_body': g['body_pose'].numpy(), 'betas': g['betas'][0].numpy(),
                      'root_orient': g['global_orient'].numpy(), 'trans': g['transl'].numpy(),
                      'mocap_frame_rate': torch.tensor(30)}
        n = smplx_data['pose_body'].shape[0]
        # GMR builds the model with smplx's default batch_size=1, so its expression default is (1, 10) while betas /
        # poses are (N, ...) and torch.cat fails for any N > 1; the model here is sized to the clip (second departure).
        body_model = smplx.create(str(body_models), 'smplx', gender='neutral', use_pca=False, batch_size=n)
        smplx_output = body_model(
            betas=torch.tensor(smplx_data['betas']).float().view(1, -1), global_orient=torch.tensor(smplx_data['root_orient']).float(),
            body_pose=torch.tensor(smplx_data['pose_body']).float(), transl=torch.tensor(smplx_data['trans']).float(),
            left_hand_pose=torch.zeros(n, 45).float(), right_hand_pose=torch.zeros(n, 45).float(),
            jaw_pose=torch.zeros(n, 3).float(), leye_pose=torch.zeros(n, 3).float(), reye_pose=torch.zeros(n, 3).float(),
            return_full_pose=True)
        height = 1.66 + 0.1 * float(smplx_data['betas'][0])
    frames, aligned_fps = gmr_smpl.get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
    note.update(actual_human_height=float(height), aligned_fps=float(aligned_fps), frames=len(frames))
    retarget = GeneralMotionRetargeting(actual_human_height=float(height), src_human='smplx', tgt_robot='unitree_g1', verbose=False)
    qpos = np.array([retarget.retarget(frame) for frame in frames])
    return qpos, note


def compare_qpos(ours: np.ndarray, ref: np.ndarray) -> dict:
    n = min(len(ours), len(ref))
    d_joint = np.abs(ours[:n, 7:] - ref[:n, 7:])
    d_root = np.linalg.norm(ours[:n, :3] - ref[:n, :3], axis=1)
    q1, q2 = ours[:n, 3:7], ref[:n, 3:7]
    d_quat = 2 * np.degrees(np.arccos(np.clip(np.abs((q1 * q2).sum(1)), 0, 1)))
    return {'frames_compared': int(n), 'frames_ours': int(len(ours)), 'frames_reference': int(len(ref)),
            'joint_abs_diff_rad': {'max': float(d_joint.max()), 'p99': float(np.percentile(d_joint, 99)), 'mean': float(d_joint.mean())},
            'root_pos_diff_m': {'max': float(d_root.max()), 'mean': float(d_root.mean())},
            'root_quat_diff_deg': {'max': float(d_quat.max())},
            'pass_joint_lt_1e-3_rad': bool(d_joint.max() < 1e-3), 'pass_root_lt_1mm': bool(d_root.max() < 1e-3)}


def compare_joints(adapter_npz: Path, gvhmr_joints_npz: Path, root: Path) -> dict:
    from humanoid_motion.retarget.sources import load_amass_smplx
    with np.load(gvhmr_joints_npz) as f:
        gv = f['joints_y_up'].astype(np.float64)
    frames = gv.shape[0]
    src = load_amass_smplx(adapter_npz, 0, frames, target_fps=30.0, height={'mode': 'gmr_beta0'}, root=root)
    ours = src.positions[:, :22]
    ref = gv @ Y_UP_TO_Z_UP.T
    report = json.loads(adapter_npz.with_suffix('.json').read_text())
    offset = report.get('ground', {}).get('offset_removed_m', 0.0)
    ref = ref.copy(); ref[:, :, 2] -= offset
    d = np.linalg.norm(ours - ref, axis=-1)
    return {'frames': int(frames), 'ground_offset_applied_m': float(offset), 'betas_mode': report.get('betas_mode'),
            'joint_pos_diff_m': {'max': float(d.max()), 'mean': float(d.mean())},
            'note': ('adapter npz joints via sources.load_amass_smplx vs GVHMR make_smplx("supermotion") joints '
                     'rotated by GMR matrix; betas differ from GVHMR per-frame betas when betas_mode is mean, '
                     'so compare the --betas first adapter output for the strict test')}


def main(argv=None) -> None:
    from humanoid_motion import ROOT
    from humanoid_motion.retarget.sources import smplx_model_dir
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gvhmr_results', type=Path, required=True)
    parser.add_argument('--our_run_dir', type=Path, required=True, help='retarget run made with the reconcile config')
    parser.add_argument('--adapter_npz_recon', type=Path, required=True, help='adapter output with --ground none --betas first')
    parser.add_argument('--gvhmr_joints', type=Path, default=None, help='npz from scripts/dump_gvhmr_joints.py')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    gmr_root = ROOT / 'third_party/GMR'
    body_models = smplx_model_dir(None, ROOT)
    ref_qpos, note = gmr_reference(args.gvhmr_results, body_models, gmr_root)
    with np.load(args.our_run_dir / 'reference.npz', allow_pickle=False) as f:
        our_qpos = f['qpos']
    result = {'reference_path': note, 'qpos': compare_qpos(our_qpos, ref_qpos)}
    if args.gvhmr_joints:
        result['joints_vs_gvhmr_body_model'] = compare_joints(args.adapter_npz_recon, args.gvhmr_joints, ROOT)
    np.savez(args.out.with_suffix('.npz'), reference_qpos=ref_qpos, our_qpos=our_qpos)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
