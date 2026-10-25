#!/usr/bin/env python3
"""GVHMR's own SMPL-X forward pass on `smpl_params_global` -> body joints, for humanoid_motion.video.reconcile.

Run inside the GVHMR conda environment with cwd = third_party/GVHMR (make_smplx uses relative input paths), e.g.
  cd third_party/GVHMR && $GVHMR_PYTHON ../../scripts/dump_gvhmr_joints.py --gvhmr_results <clip>/hmr4d_results.pt --out ...
Writes joints_y_up (F, 22, 3) from make_smplx("supermotion"), the body model GVHMR renders with; the adapter's
z-up joints must equal these after GMR's rotation matrix. Nothing in GVHMR is modified.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from hmr4d.utils.smplx_utils import make_smplx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gvhmr_results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    pred = torch.load(args.gvhmr_results, map_location='cpu')
    params = {k: v.float() for k, v in pred['smpl_params_global'].items()}
    model = make_smplx('supermotion')
    with torch.no_grad():
        out = model(**params)
    joints = out.joints[:, :22].numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, joints_y_up=joints, betas=params['betas'].numpy(), transl=params['transl'].numpy(),
             global_orient=params['global_orient'].numpy())
    print('frames', joints.shape[0], 'pelvis y range', joints[:, 0, 1].min(), joints[:, 0, 1].max())


if __name__ == '__main__':
    main()
