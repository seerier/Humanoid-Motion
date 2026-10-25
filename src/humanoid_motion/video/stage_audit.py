"""Video front end: per-stage audit of one clip directory, so a bad end result is attributed to the stage that broke it.

Stages, in pipeline order, each with its own evidence and a pass / flag verdict:
  1 reconstruction  GVHMR in-camera SMPL-X joints reprojected with GVHMR's own intrinsics against the ViTPose 2-D
                    keypoints GVHMR itself detected (COCO-17), normalised by the detection box height. A large error
                    means the body model does not sit on the person in the image (wrong person, tracking loss, bad fit).
  2 interface       the adapter reconciliation record (reconcile.json): our SMPL-X entry reproduces GVHMR's joints.
  3 retarget        GMR output against GVHMR's human: per-limb direction angle (human bone vs robot bone, same frame),
                    GMR's own IK task residuals (diagnostics.json), and joint-limit pinning per frame from the saved
                    joint angles against the MuJoCo joint ranges. A limb that points 90 deg away from the human's, or a
                    joint sitting on its limit for most of the clip, is a retargeting failure even when the human is fine.
  4 ground          adapter ground report and quality gate (feet in the air / below the floor).
  5 tracking        optional: evaluator results.json files (--eval label=path), completion and first failure.
Verdict = the first stage with a flag. Thresholds are constants below and are copied into the output.

Clip directory layout: the one scripts/video_to_reference.py writes (human/smplx.npz, retarget/g1/, gate.json,
source.json naming the GVHMR result). Outputs <clip_dir>/stage_audit.json and stage_audit.png (per-frame curves):
    .venv/bin/python -m humanoid_motion.video.stage_audit --clip_dir outputs/video/<clip>
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from humanoid_motion import ROOT

# COCO-17 index -> SMPL-X joint index (first 22 joints of the SMPL-X joint set)
COCO_TO_SMPLX = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
COCO_NAMES = {5: 'L_shoulder', 6: 'R_shoulder', 7: 'L_elbow', 8: 'R_elbow', 9: 'L_wrist', 10: 'R_wrist',
              11: 'L_hip', 12: 'R_hip', 13: 'L_knee', 14: 'R_knee', 15: 'L_ankle', 16: 'R_ankle'}
KP_CONF_MIN = 0.3
RECON_MEAN_MAX = {'default': 0.08, 'hip': 0.12}        # mean 2-D error / box height
RECON_BAD_FRAC_MAX, RECON_BAD_ERR = 0.10, 0.15          # fraction of frames with error above RECON_BAD_ERR

# human bone (SMPL-X joint pair) and the robot bone (MuJoCo body pair) it should point like
LIMBS = {
    'L_upper_arm': ((16, 18), ('left_shoulder_roll_link', 'left_elbow_link')),
    'L_forearm': ((18, 20), ('left_elbow_link', 'left_wrist_yaw_link')),
    'R_upper_arm': ((17, 19), ('right_shoulder_roll_link', 'right_elbow_link')),
    'R_forearm': ((19, 21), ('right_elbow_link', 'right_wrist_yaw_link')),
    'L_thigh': ((1, 4), ('left_hip_roll_link', 'left_knee_link')),
    'L_shank': ((4, 7), ('left_knee_link', 'left_ankle_roll_link')),
    'R_thigh': ((2, 5), ('right_hip_roll_link', 'right_knee_link')),
    'R_shank': ((5, 8), ('right_knee_link', 'right_ankle_roll_link')),
}
LIMB_MEAN_MAX_DEG, LIMB_P95_MAX_DEG = 30.0, 45.0
IK_ROT_MAX_DEG, IK_POS_MAX_M = 45.0, 0.20
PIN_TOL_RAD, PIN_FRAC_MAX = 1e-3, 0.20
AIRBORNE_FRAC_MAX = 0.10                                 # frames with the lowest foot more than 10 cm up


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def summary(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), p95=float(np.percentile(x, 95)), max=float(x.max()))


def smplx_joints(model_dir: Path, betas, root_orient, pose_body, trans) -> np.ndarray:
    """SMPL-X joints (F, 22, 3) for per-frame or shared betas (10 or 16 numbers)."""
    import smplx
    import torch
    from humanoid_motion.retarget.sources import require_smplx_model
    n = pose_body.shape[0]
    betas = np.asarray(betas, np.float32)
    if betas.ndim == 1:
        betas = np.broadcast_to(betas, (n, betas.shape[0]))
    nb = betas.shape[1]
    require_smplx_model(model_dir, 'neutral')
    model = smplx.create(str(model_dir), 'smplx', gender='neutral', use_pca=False, num_betas=nb, batch_size=n)
    with torch.no_grad():
        out = model(betas=torch.as_tensor(np.ascontiguousarray(betas)), global_orient=torch.as_tensor(root_orient).float(),
                    body_pose=torch.as_tensor(pose_body).float(), transl=torch.as_tensor(trans).float(),
                    left_hand_pose=torch.zeros(n, 45), right_hand_pose=torch.zeros(n, 45), jaw_pose=torch.zeros(n, 3),
                    leye_pose=torch.zeros(n, 3), reye_pose=torch.zeros(n, 3),
                    expression=torch.zeros(n, model.num_expression_coeffs), return_verts=False, return_full_pose=False)
    return out.joints[:, :22].numpy()


def stage_reconstruction(gvhmr_results: Path, model_dir: Path) -> dict:
    import torch
    res = torch.load(gvhmr_results, map_location='cpu', weights_only=False)
    pre = gvhmr_results.parent / 'preprocess'
    kp = torch.load(pre / 'vitpose.pt', map_location='cpu', weights_only=False).numpy()         # (F, 17, 3) x, y, conf
    bbx = torch.load(pre / 'bbx.pt', map_location='cpu', weights_only=False)['bbx_xyxy'].numpy()
    box_h = np.maximum(bbx[:, 3] - bbx[:, 1], 1.0)
    p = {k: v.numpy() for k, v in res['smpl_params_incam'].items()}
    K = res['K_fullimg'].numpy()
    joints = smplx_joints(model_dir, p['betas'], p['global_orient'], p['body_pose'], p['transl'])   # camera frame
    uvw = np.einsum('fij,fkj->fki', K, joints)
    uv = uvw[..., :2] / np.maximum(uvw[..., 2:3], 1e-6)
    per_joint, curves = {}, {}
    flags = []
    for ci, si in COCO_TO_SMPLX.items():
        ok = kp[:, ci, 2] > KP_CONF_MIN
        err = np.linalg.norm(uv[:, si] - kp[:, ci, :2], axis=1) / box_h
        name = COCO_NAMES[ci]
        e = err[ok] if ok.any() else np.array([np.nan])
        bad = float((e > RECON_BAD_ERR).mean()) if ok.any() else float('nan')
        per_joint[name] = dict(frames_used=int(ok.sum()), mean=float(np.nanmean(e)), p95=float(np.nanpercentile(e, 95)),
                               frac_above_bad=bad)
        curves[name] = np.where(ok, err, np.nan)
        lim = RECON_MEAN_MAX['hip' if 'hip' in name else 'default']
        if ok.any() and (per_joint[name]['mean'] > lim or bad > RECON_BAD_FRAC_MAX):
            flags.append(f'{name}: mean {per_joint[name]["mean"]:.3f} of box height, {100 * bad:.0f}% frames above {RECON_BAD_ERR}')
    limb = np.nanmax(np.stack([curves[n] for n in curves]), axis=0)
    return dict(rule=f'SMPL-X in-camera joints reprojected with K_fullimg vs ViTPose (conf > {KP_CONF_MIN}); error / '
                     f'detection box height; flag when mean > {RECON_MEAN_MAX} or > {100 * RECON_BAD_FRAC_MAX:.0f}% '
                     f'frames above {RECON_BAD_ERR}',
                frames=int(len(kp)), per_joint=per_joint, flags=flags, verdict='flag' if flags else 'pass',
                curve_max_limb_error=limb)


def stage_interface(clip_dir: Path, source_dir: Path) -> dict:
    for d in (clip_dir, source_dir):
        f = d / 'reconcile.json'
        if f.exists():
            r = json.loads(f.read_text())
            q = r.get('qpos', {})
            j = r.get('joints_vs_gvhmr_body_model', {})
            flags = []
            if q and not q.get('pass_root_lt_1mm', True):
                flags.append('root position differs from GVHMR replay by > 1 mm')
            if q and q.get('joint_abs_diff_rad', {}).get('p99', 0) > 1e-3:
                flags.append('joint angles differ from GVHMR replay (p99 > 1e-3 rad)')
            return dict(rule='reconcile.json from the production run: adapter npz replayed through GVHMR\'s own body '
                             'model; root within 1 mm and joint p99 within 1e-3 rad', file=str(f.relative_to(ROOT)),
                        qpos=q, joints_vs_gvhmr_body_model=j, flags=flags, verdict='flag' if flags else 'pass')
    return dict(rule='reconcile.json', flags=[], verdict='not run', note='no reconcile.json in clip or source dir')


def robot_fk(run_dir: Path):
    import mujoco
    import humanoid_motion.retarget.sources  # noqa: F401  (puts the pinned GMR checkout on sys.path)
    from general_motion_retargeting import ROBOT_XML_DICT
    m = pickle.load(open(run_dir / 'motion.pkl', 'rb'))
    root_pos, root_rot, dof = np.asarray(m['root_pos']), np.asarray(m['root_rot']), np.asarray(m['dof_pos'])
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT['unitree_g1']))
    data = mujoco.MjData(model)
    assert model.nq == 7 + dof.shape[1], (model.nq, dof.shape)
    hinge = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    names = [model.joint(i).name for i in hinge]
    lo, hi = model.jnt_range[hinge, 0], model.jnt_range[hinge, 1]
    body_names = [model.body(i).name for i in range(model.nbody)]
    xpos = np.zeros((len(dof), model.nbody, 3))
    for f in range(len(dof)):
        q = np.concatenate([root_pos[f], root_rot[f][[3, 0, 1, 2]], dof[f]])      # GMR stores xyzw, MuJoCo wants wxyz
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        xpos[f] = data.xpos
    pinned = (dof <= lo + PIN_TOL_RAD) | (dof >= hi - PIN_TOL_RAD)
    return xpos, body_names, dof, names, pinned, float(m['fps'])


def stage_retarget(clip_dir: Path, model_dir: Path) -> dict:
    run_dir = clip_dir / 'retarget' / 'g1'
    diag = json.loads((run_dir / 'diagnostics.json').read_text())
    with np.load(clip_dir / 'human' / 'smplx.npz', allow_pickle=False) as f:
        hj = smplx_joints(model_dir, f['betas'], f['root_orient'], f['pose_body'], f['trans'])
    xpos, body_names, dof, joint_names, pinned, fps = robot_fk(run_dir)
    n = min(len(hj), len(xpos))
    limbs, curves, flags = {}, {}, []
    for name, ((a, b), (ra, rb)) in LIMBS.items():
        hv = unit(hj[:n, b] - hj[:n, a])
        rv = unit(xpos[:n, body_names.index(rb)] - xpos[:n, body_names.index(ra)])
        ang = np.degrees(np.arccos(np.clip((hv * rv).sum(1), -1, 1)))
        limbs[name] = dict(**summary(ang), frac_above_45deg=float((ang > 45).mean()), first_frame=float(ang[0]))
        curves[name] = ang
        if ang.mean() > LIMB_MEAN_MAX_DEG or np.percentile(ang, 95) > LIMB_P95_MAX_DEG:
            flags.append(f'{name} points {ang.mean():.0f} deg (mean) away from the human\'s, first frame {ang[0]:.0f} deg')
    pin_frac = {j: float(pinned[:, i].mean()) for i, j in enumerate(joint_names)}
    for j, fr in pin_frac.items():
        if fr > PIN_FRAC_MAX:
            flags.append(f'{j} on its joint limit in {100 * fr:.0f}% of frames')
    ik_rot = {k: v['mean'] for k, v in diag.get('ik_task_rotation_error_deg', {}).items()}
    ik_pos = {k: v['mean'] for k, v in diag.get('world_target_body_position_residual_m', {}).items()}
    for k, v in ik_rot.items():
        if v > IK_ROT_MAX_DEG:
            flags.append(f'GMR orientation task {k}: {v:.0f} deg mean residual')
    for k, v in ik_pos.items():
        if ('wrist' in k or 'elbow' in k or 'ankle' in k or 'knee' in k) and v > IK_POS_MAX_M:
            flags.append(f'GMR position task {k}: {v:.2f} m mean residual')
    return dict(rule=f'per-limb direction angle human vs robot (flag mean > {LIMB_MEAN_MAX_DEG} or p95 > {LIMB_P95_MAX_DEG} '
                     f'deg); joint on its MuJoCo range limit within {PIN_TOL_RAD} rad in > {100 * PIN_FRAC_MAX:.0f}% '
                     f'frames; GMR task residuals (rotation > {IK_ROT_MAX_DEG} deg, limb position > {IK_POS_MAX_M} m)',
                frames_compared=int(n), limb_angle_deg=limbs, joint_limit_pinned_fraction=pin_frac,
                ik_task_rotation_error_deg_mean=ik_rot, ik_position_residual_m_mean=ik_pos,
                ik_task_error_norm_mean=diag.get('ik_task_error_norm', {}).get('mean'),
                flags=flags, verdict='flag' if flags else 'pass', curves=curves, curve_pinned_count=pinned.sum(1))


def stage_ground(clip_dir: Path) -> dict:
    out, flags = {}, []
    ref = clip_dir / 'retarget' / 'g1' / 'reference.npz'
    with np.load(ref, allow_pickle=False) as f:
        sole_z = f['foot_collision_bottom_z']                     # (frames, 2 feet, spheres), GMR output + FK
    lowest = sole_z.min(axis=(1, 2))
    airborne = float((lowest > 0.10).mean())
    penetration = float((lowest < -0.025).mean())
    out['robot_lowest_sole_m'] = dict(min=float(lowest.min()), median=float(np.median(lowest)), max=float(lowest.max()))
    out['airborne_fraction_gt_10cm'] = airborne
    out['penetration_fraction_gt_2_5cm'] = penetration
    if airborne > AIRBORNE_FRAC_MAX:
        flags.append(f'lowest sole more than 10 cm above the floor in {100 * airborne:.0f}% of frames (reference floats)')
    if penetration > 0.01:
        flags.append(f'lowest sole more than 2.5 cm below the floor in {100 * penetration:.0f}% of frames')
    rep = clip_dir / 'human' / 'smplx.json'
    if rep.exists():
        g = json.loads(rep.read_text()).get('ground', {})
        out['adapter_ground'] = {k: v for k, v in g.items() if k != 'offset_per_frame_m'}
    gate = clip_dir / 'gate.json'
    if gate.exists():
        gj = json.loads(gate.read_text())
        out['gate'] = {k: dict(value=v.get('value'), threshold=v.get('threshold'), passed=v.get('pass'))
                       for k, v in gj.get('checks', {}).items()}
    return dict(rule=f'lowest sole collision sphere of the retargeted robot per frame (reference.npz): flag when > '
                     f'{100 * AIRBORNE_FRAC_MAX:.0f}% of frames are more than 10 cm up or > 1% more than 2.5 cm down',
                **out, flags=flags, verdict='flag' if flags else 'pass', curve_lowest_sole=lowest)


def stage_tracking(clip_id: str, evals: list[str]) -> dict:
    rows, flags = {}, []
    for spec in evals:
        label, path = spec.split('=', 1)
        r = json.loads(Path(path).read_text())
        entry = next((e for e in r.get('rows', []) if e.get('clip_id') == clip_id), None)
        if entry is None:
            rows[label] = dict(note=f'no row for {clip_id} in {path}')
            continue
        fps = 50.0
        ff = entry.get('first_failure_frame')
        keep = dict(complete=entry.get('complete'), completion_fraction=entry.get('completion_fraction'),
                    first_failure_time_s=(ff / fps if ff is not None else None), failure_terms=entry.get('failure_terms'),
                    protocol=entry.get('protocol'))
        m = entry.get('metrics_on_observed_prefix', {})
        if 'aligned_body_position_m' in m:
            keep['aligned_body_position_m_mean'] = m['aligned_body_position_m'].get('mean')
        rows[label] = keep
        if keep['complete'] is False:
            flags.append(f'{label}: fails at {keep["first_failure_time_s"]:.1f} s ({", ".join(keep["failure_terms"] or [])})')
    return dict(rule='evaluator results.json rows (50 Hz reference frames): completion and first failure per policy',
                policies=rows, flags=flags, verdict=('flag' if flags else 'pass') if rows else 'not run')


def plot(path: Path, s1: dict | None, s3: dict, s4: dict, fps: float) -> None:
    from PIL import Image, ImageDraw, ImageFont
    W, H, M = 1600, 1100, 70
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 16)
    except OSError:
        fnt = ImageFont.load_default()
    rows = [('stage 1  reconstruction: max 2-D error over limb joints / box height (flag line 0.15)',
             {'max over limb joints': s1['curve_max_limb_error']} if s1 else {}, 0.4, RECON_BAD_ERR),
            ('stage 3  retarget: limb direction angle human vs robot, deg (flag line 45)', s3['curves'], 180.0, LIMB_P95_MAX_DEG),
            ('stage 3  retarget: number of joints on a range limit', {'pinned joints': s3['curve_pinned_count']}, 8.0, None),
            ('stage 4  ground: lowest sole sphere height of the retargeted robot, m (flag line 0.10)',
             {'lowest sole': s4['curve_lowest_sole']}, 0.3, 0.10)]
    rh = (H - 2 * M) // len(rows)
    palette = ['#d62728', '#ff7f0e', '#1f77b4', '#17becf', '#2ca02c', '#98df8a', '#9467bd', '#c5b0d5']
    for r, (title, series, ymax, hline) in enumerate(rows):
        y0 = M + r * rh
        y1 = y0 + rh - 40
        d.rectangle([M, y0, W - M, y1], outline='#888')
        d.text((M, y0 - 20), title, fill='black', font=fnt)
        if hline is not None:
            yy = y1 - (hline / ymax) * (y1 - y0)
            d.line([(M, yy), (W - M, yy)], fill='#bbb', width=1)
        for i, (name, arr) in enumerate(series.items()):
            arr = np.asarray(arr, float)
            n = len(arr)
            pts = [(M + (W - 2 * M) * k / max(n - 1, 1), y1 - (min(float(v), ymax) / ymax) * (y1 - y0))
                   for k, v in enumerate(arr) if np.isfinite(v)]
            if len(pts) > 1:
                d.line(pts, fill=palette[i % len(palette)], width=2)
            d.text((W - M - 200, y0 + 5 + 18 * i), name, fill=palette[i % len(palette)], font=fnt)
        d.text((M, y1 + 4), '0 s', fill='black', font=fnt)
        n = max((len(np.asarray(a)) for a in series.values()), default=1)
        d.text((W - M - 60, y1 + 4), f'{n / fps:.0f} s', fill='black', font=fnt)
        d.text((M - 60, y0), f'{ymax:g}', fill='black', font=fnt)
    im.save(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clip_dir', type=Path, required=True)
    ap.add_argument('--eval', action='append', default=[], help='label=path/to/evaluator results.json (repeatable)')
    ap.add_argument('--out', type=Path, default=None, help='output stem; default <clip_dir>/stage_audit')
    ap.add_argument('--skip_reconstruction', action='store_true')
    args = ap.parse_args(argv)
    clip_dir = args.clip_dir.resolve()
    out = args.out or clip_dir / 'stage_audit'
    source = json.loads((clip_dir / 'source.json').read_text()) if (clip_dir / 'source.json').exists() else {}
    source_dir = Path(source.get('source_clip_dir', clip_dir))
    clip_id = clip_dir.name
    gvhmr_results = Path(source.get('gvhmr_results', source_dir / 'gvhmr' / source_dir.name / 'hmr4d_results.pt'))
    from humanoid_motion.retarget.sources import smplx_model_dir
    model_dir = smplx_model_dir(None, ROOT)
    stages = {}
    s1 = None
    if not args.skip_reconstruction and gvhmr_results.exists():
        s1 = stage_reconstruction(gvhmr_results, model_dir)
        stages['1_reconstruction'] = {k: v for k, v in s1.items() if not k.startswith('curve')}
    else:
        stages['1_reconstruction'] = dict(verdict='not run', flags=[], note=f'{gvhmr_results} missing or skipped')
    stages['2_interface'] = stage_interface(clip_dir, source_dir)
    s3 = stage_retarget(clip_dir, model_dir)
    stages['3_retarget'] = {k: v for k, v in s3.items() if not k.startswith('curve')}
    s4 = stage_ground(clip_dir)
    stages['4_ground'] = {k: v for k, v in s4.items() if not k.startswith('curve')}
    stages['5_tracking'] = stage_tracking(clip_id, args.eval)
    first_flag = next((k for k, v in stages.items() if v.get('verdict') == 'flag'), None)
    verdict = (f'first flagged stage: {first_flag}: ' + '; '.join(stages[first_flag]['flags'])) if first_flag else 'no stage flagged'
    fps = float(json.loads((clip_dir / 'retarget' / 'g1' / 'metadata.json').read_text()).get('fps', 30.0)) \
        if (clip_dir / 'retarget' / 'g1' / 'metadata.json').exists() else 30.0
    plot(out.with_suffix('.png'), s1, s3, s4, fps)
    result = dict(clip=clip_id, clip_dir=str(clip_dir), source_clip_dir=str(source_dir), verdict=verdict,
                  stages=stages, figure=str(out.with_suffix('.png')),
                  thresholds=dict(RECON_MEAN_MAX=RECON_MEAN_MAX, RECON_BAD_ERR=RECON_BAD_ERR, RECON_BAD_FRAC_MAX=RECON_BAD_FRAC_MAX,
                                  LIMB_MEAN_MAX_DEG=LIMB_MEAN_MAX_DEG, LIMB_P95_MAX_DEG=LIMB_P95_MAX_DEG,
                                  IK_ROT_MAX_DEG=IK_ROT_MAX_DEG, IK_POS_MAX_M=IK_POS_MAX_M, PIN_FRAC_MAX=PIN_FRAC_MAX,
                                  AIRBORNE_FRAC_MAX=AIRBORNE_FRAC_MAX))
    out.with_suffix('.json').write_text(json.dumps(result, indent=2, default=float) + '\n')
    print(f'== {clip_id}')
    ev = {}
    if s1:
        ev['1_reconstruction'] = 'worst joint mean 2-D error %.3f of box height' % max(j['mean'] for j in s1['per_joint'].values())
    ev['3_retarget'] = 'worst limb %s %.0f deg mean; IK norm %.2f' % (max(s3['limb_angle_deg'], key=lambda k: s3['limb_angle_deg'][k]['mean']),
                        max(v['mean'] for v in s3['limb_angle_deg'].values()), s3['ik_task_error_norm_mean'] or float('nan'))
    ev['4_ground'] = 'airborne %.0f%%, penetration %.0f%%' % (100 * s4['airborne_fraction_gt_10cm'], 100 * s4['penetration_fraction_gt_2_5cm'])
    ev['5_tracking'] = '; '.join(f"{k}: {'complete' if v.get('complete') else 'fail'} {v.get('aligned_body_position_m_mean', float('nan')):.3f} m" for k, v in stages['5_tracking'].get('policies', {}).items() if 'complete' in v)
    for k, v in stages.items():
        print(f'  {k:18s} {v.get("verdict", "?"):8s} [{ev.get(k, "")}] ' + ('; '.join(v.get('flags', [])) or v.get('note', '')))
    result['evidence'] = ev
    out.with_suffix('.json').write_text(json.dumps(result, indent=2, default=float) + '\n')
    print('  VERDICT', verdict)
    print('  wrote', out.with_suffix('.json'), out.with_suffix('.png'))


if __name__ == '__main__':
    main()
