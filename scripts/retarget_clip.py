#!/usr/bin/env python3
"""Retarget one LAFAN1 BVH window to the Unitree G1 with GMR and audit the result (CPU, no physics)."""
import argparse
import json
from pathlib import Path

from humanoid_motion import ROOT
from humanoid_motion.retarget.gmr import retarget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/walk1_s1.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    diagnostics = retarget(json.loads(args.config.read_text()), args.output)
    print(json.dumps(diagnostics, indent=2))


if __name__ == '__main__':
    main()
