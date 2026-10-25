#!/usr/bin/env python3
"""Export a motion-interface-only MorphMimic checkpoint with MorphMimic's own exporter (deploy.export_onnx).

Their exporter always builds the policy with the five-point (keypoint) goal encoder and reads that encoder's input
normaliser; a run trained without --kp-interface has neither, so the exporter stops with a KeyError. This adapter
writes a copy of the checkpoint with a five-point encoder initialised as their Policy initialises it (torch seed 0)
and a fresh (identity) normaliser added, then runs their exporter, unmodified, on the copy. The motion interface, the
only one such a run trained and the one evaluated here, is untouched; their exporter's parity check compares its
ONNX actions with PyTorch. The five-point output of such an export is untrained and must not be used. A checkpoint
that already has both is exported as it is.

  third_party/morphmimic/.venv/bin/python -m humanoid_motion.tracking.supermimic.export <ckpt_N.pt> <out_dir>
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from humanoid_motion import ROOT


def with_keypoint_interface(ckpt: dict) -> tuple[dict, bool]:
    """The checkpoint with the five-point encoder and normaliser their exporter expects; True if anything was added."""
    import torch
    from morphmimic.constants import KP_FEAT_DIM
    from morphmimic.model.policy import Policy, PolicyCfg
    from morphmimic.train.normalizers import RunningNormalizer
    prefix = 'kp_encoder.'
    has_encoder = any(k.startswith(prefix) for k in ckpt['policy'])
    if has_encoder and 'kp_norm' in ckpt:
        return ckpt, False
    cfg, morph = ckpt['cfg'], ckpt.get('morph')
    torch.manual_seed(0)
    fresh = Policy(PolicyCfg(          # the same construction as deploy.export_onnx._build_policy
        enc_hidden=tuple(cfg['policy_hidden']), dec_hidden=tuple(cfg['policy_hidden']), act=cfg['act'],
        hist_len=int(cfg['hist_len']), morph_dim=0 if morph is None else int(morph['actor'].numel()),
        film=False if morph is None else bool(morph['film']), keypoint=True))
    policy = dict(ckpt['policy'])
    if not has_encoder:
        policy.update({k: v for k, v in fresh.state_dict().items() if k.startswith(prefix)})
    out = dict(ckpt, policy=policy)
    out.setdefault('kp_norm', RunningNormalizer(KP_FEAT_DIM).state_dict())
    return out, True


def export(checkpoint: Path, out_dir: Path, staging: Path | None = None,
           morphmimic: Path = ROOT / 'third_party/morphmimic') -> None:
    """`staging`: where the augmented copy of the checkpoint goes (default: next to out_dir)."""
    import torch
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    ckpt, added = with_keypoint_interface(ckpt)
    source = checkpoint
    if added:
        staging = staging or out_dir.with_name(out_dir.name + '.staging')
        staging.mkdir(parents=True, exist_ok=True)
        source = staging / checkpoint.name          # keeps the ckpt_<it>.pt name their exporter parses
        torch.save(ckpt, source)
        print(f'[export] {checkpoint.name}: motion-only run; added an untrained five-point encoder for the exporter')
    subprocess.run([sys.executable, '-m', 'deploy.export_onnx', str(source), str(out_dir)], cwd=morphmimic, check=True)
    if added:
        (out_dir / 'FIVE_POINT_OUTPUT_UNTRAINED.txt').write_text(
            f'{checkpoint} was trained with the motion interface only. The five-point (keypoint) output of this\n'
            'policy.onnx comes from a placeholder encoder added by humanoid_motion.tracking.supermimic.export so that\n'
            'MorphMimic\'s exporter runs; do not use it. The motion output is the trained policy.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('checkpoint', type=Path, help='MorphMimic ckpt_<it>.pt of a single-robot run')
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--staging', type=Path, help='Directory for the augmented checkpoint copy (large)')
    args = parser.parse_args()
    export(args.checkpoint.resolve(), args.out_dir.resolve(), args.staging)


if __name__ == '__main__':
    main()
