"""Video front end: the camera that filmed a clip, in the G1 reference frame, so a render can view the robot from where
the video was shot.

GVHMR gives the body twice per frame: in camera coordinates (`smpl_params_incam`, OpenCV axes: x right, y down,
z forward) and in its gravity-aligned world (`smpl_params_global`, y up). The pelvis pose in both gives the camera pose
in that world, per frame:
    R_wc = R_w R_c^T,     centre = p_w - R_wc p_c        (R: root orientation, p: pelvis joint)
For a static camera these are one pose up to GVHMR's own inconsistency between its two heads, so the camera is the
per-frame median (centre; axes normalised and re-orthogonalised) and the spread is reported. The pose then takes the
transforms the body took:
  - the adapter's y-up -> z-up rotation and its ground offset (mean over frames for a per-frame offset);
  - GMR's root scaling about the origin. The scale is fitted from the human and G1 pelvis paths (least squares through
    the origin, horizontal axes) and applied to the camera centre in all three axes, so the floor stays at z = 0 and
    the robot, that much smaller than the human, fills the frame about as the human does.
The vertical field of view comes from GVHMR's intrinsics (K_fullimg; unless its demo was given a focal length, it
assumes one equal to the image diagonal). MuJoCo's free camera has no roll; the camera's roll is reported, not used.
The file also keeps the person's pelvis in camera coordinates per frame (unscaled, GVHMR's 30 fps), so a render can
place the camera where the robot appears as the person does in the video (scripts/render_video.py --camera video):
GVHMR's world path and its in-camera path drift apart by tens of centimetres, and a tracked robot drifts on its own.

    .venv/bin/python -m humanoid_motion.video.camera --clip_dir outputs/video/<clip>     # writes <clip_dir>/camera.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_motion import ROOT


def camera_from_poses(p_w: np.ndarray, r_w: Rotation, p_c: np.ndarray, r_c: Rotation) -> dict:
    """Camera pose in the world from the same body seen in the world (p_w, r_w) and in the camera (p_c, r_c), per
    frame: median centre and axes (columns x right, y down, z forward), with the per-frame spread."""
    r_wc = r_w * r_c.inv()
    centres = p_w - r_wc.apply(p_c)
    axes = r_wc.as_matrix()                                                # (F, 3, 3), columns = camera axes
    u, _, vt = np.linalg.svd(np.median(axes, axis=0))
    rot = u @ vt                                                           # nearest rotation to the median axes
    forward = axes[:, :, 2]
    return dict(centre=np.median(centres, axis=0), axes=rot,
                centre_p5_p95_m=np.percentile(centres, 95, axis=0) - np.percentile(centres, 5, axis=0),
                forward_max_deg=float(np.degrees(np.arccos(np.clip(forward @ rot[:, 2], -1., 1.))).max()))


def gvhmr_pelvis(params: dict, model_dir: Path) -> tuple[np.ndarray, Rotation]:
    """Pelvis joint (F, 3) and root orientation of one GVHMR parameter set, with its per-frame betas."""
    import torch
    from humanoid_motion.video.gvhmr_adapter import NUM_BETAS_OUT, _body_model
    p = {k: np.asarray(v.detach().cpu().numpy() if hasattr(v, 'detach') else v, np.float64) for k, v in params.items()}
    n = p['body_pose'].shape[0]
    model = _body_model(model_dir, n)
    betas = np.concatenate([p['betas'], np.zeros((n, NUM_BETAS_OUT - p['betas'].shape[1]))], axis=1)
    with torch.no_grad():
        out = model(betas=torch.as_tensor(betas).float(), global_orient=torch.as_tensor(p['global_orient']).float(),
                    body_pose=torch.as_tensor(p['body_pose']).float(), transl=torch.as_tensor(p['transl']).float(),
                    left_hand_pose=torch.zeros(n, 45), right_hand_pose=torch.zeros(n, 45), jaw_pose=torch.zeros(n, 3),
                    leye_pose=torch.zeros(n, 3), reye_pose=torch.zeros(n, 3),
                    expression=torch.zeros(n, model.num_expression_coeffs))
    return out.joints[:, 0].numpy().astype(np.float64), Rotation.from_rotvec(p['global_orient'])


def ground_offset(sidecar: dict, frames: int) -> np.ndarray:
    """The adapter's ground offset per frame (subtracted from z), for any of its ground modes."""
    ground = sidecar.get('ground', {})
    if 'offset_per_frame_m' in ground:
        return np.asarray(ground['offset_per_frame_m'], np.float64)
    return np.full(frames, float(ground.get('offset_removed_m', 0.0)))


def estimate(clip_dir: Path, model_dir: Path) -> dict:
    import torch
    from humanoid_motion.video.gvhmr_adapter import Y_UP_TO_Z_UP
    source = json.loads((clip_dir / 'source.json').read_text()) if (clip_dir / 'source.json').exists() else {}
    results = Path(source.get('gvhmr_results', clip_dir / 'gvhmr' / clip_dir.name / 'hmr4d_results.pt'))
    res = torch.load(results, map_location='cpu', weights_only=False)
    p_w, r_w = gvhmr_pelvis(res['smpl_params_global'], model_dir)
    p_c, r_c = gvhmr_pelvis(res['smpl_params_incam'], model_dir)
    cam = camera_from_poses(p_w, r_w, p_c, r_c)
    offset = ground_offset(json.loads((clip_dir / 'human' / 'smplx.json').read_text()), len(p_w))
    # into the adapter's frame: rotate, then lower by the ground offset (the human pelvis path checks the same transform)
    centre = Y_UP_TO_Z_UP @ cam['centre'] - np.array([0., 0., offset.mean()])
    axes = Y_UP_TO_Z_UP @ cam['axes']
    human = p_w @ Y_UP_TO_Z_UP.T
    human[:, 2] -= offset
    with np.load(clip_dir / 'retarget' / 'g1' / 'reference.npz') as ref:
        names = [str(n) for n in ref['body_names']]
        robot = ref['body_positions'][:, names.index('pelvis')].astype(np.float64)
    h, r = human[:, :2].ravel(), robot[:, :2].ravel()
    scale = float(h @ r / (h @ h))                                         # GMR scales the root about the origin
    k = res['K_fullimg'][0].numpy().astype(np.float64)
    from humanoid_motion.video.gvhmr_adapter import GVHMR_FPS
    width, height = int(round(2 * k[0, 2])), int(round(2 * k[1, 2]))
    return {
        'frame': 'G1 reference world (z up, floor at z = 0)',
        'position_m': (scale * centre).round(4).tolist(),
        'forward': axes[:, 2].round(5).tolist(), 'up': (-axes[:, 1]).round(5).tolist(),
        'fovy_deg': round(float(np.degrees(2 * np.arctan(k[1, 2] / k[1, 1]))), 3), 'image_size': [width, height],
        'roll_deg': round(float(np.degrees(np.arcsin(np.clip(axes[2, 0], -1., 1.)))), 3),
        'scale': round(scale, 4),
        'scale_fit_rms_m': round(float(np.sqrt(np.mean((scale * h - r) ** 2))), 4),
        'spread': {'centre_p5_p95_m': cam['centre_p5_p95_m'].round(3).tolist(),
                   'forward_max_deg': round(cam['forward_max_deg'], 2)},
        'ground_offset_mean_m': round(float(offset.mean()), 4),
        'fps': GVHMR_FPS, 'human_pelvis_in_camera_m': p_c.round(4).tolist(),
        'gvhmr_results': str(results),
        'method': 'median over frames of the camera pose from GVHMR in-camera vs world pelvis poses; adapter rotation '
                  'and mean ground offset; GMR root scale fitted on the pelvis path',
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clip_dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None, help='default <clip_dir>/camera.json')
    args = ap.parse_args(argv)
    from humanoid_motion.retarget.sources import smplx_model_dir
    clip_dir = args.clip_dir.resolve()
    camera = estimate(clip_dir, smplx_model_dir(None, ROOT))
    out = args.out or clip_dir / 'camera.json'
    out.write_text(json.dumps(camera, indent=1) + '\n')
    print(json.dumps(camera, indent=1))


if __name__ == '__main__':
    main()
