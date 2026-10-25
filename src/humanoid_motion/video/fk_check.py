"""Recompute every body position of an Isaac tracking motion.npz with an independent forward kinematics and compare
with the npz's own body_pos_w (Isaac PhysX forward kinematics at 50 Hz).

Independent FK: humanoid_motion.geometry.xml_forward_kinematics over GMR's G1 MuJoCo XML (SciPy rotations, XML body
transforms; no MuJoCo, no Isaac). Inputs per frame: the npz root pose (body_pos_w / body_quat_w of `pelvis`) and the
29 joint angles mapped from Isaac's joint order to the XML's. Isaac's joint and body order come from a `<npz>.names.json`
next to the file or, for scripts/import_library.py output, the `motion.manifest.json` sidecar. Bodies are matched by
name; the error is the max over frames and shared bodies. Verdict: < 1 mm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def runtime_names(npz: Path) -> dict:
    """Isaac joint_names / body_names of a motion npz: `<npz>.names.json`, else the import's motion.manifest.json."""
    sidecar = Path(str(npz) + '.names.json')
    if not sidecar.exists():
        sidecar = npz.with_suffix('.manifest.json')
    names = json.loads(sidecar.read_text())
    return {'joint_names': names['joint_names'], 'body_names': names['body_names']}


def check(npz: Path, xml: Path, stride: int = 1) -> dict:
    import mujoco
    from humanoid_motion.geometry import xml_forward_kinematics
    names = runtime_names(npz)
    with np.load(npz) as f:
        joint_pos, body_pos, body_quat, fps = f['joint_pos'], f['body_pos_w'], f['body_quat_w'], float(f['fps'][0])
    m = mujoco.MjModel.from_xml_path(str(xml))
    xml_joints = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    order = [names['joint_names'].index(n) for n in xml_joints]
    isaac_bodies = names['body_names']
    pelvis = isaac_bodies.index('pelvis')
    shared = [b for b in isaac_bodies if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) >= 0]
    errors = np.zeros((0, len(shared)))
    per_frame = []
    for k in range(0, len(joint_pos), stride):
        qpos = np.concatenate([body_pos[k, pelvis], body_quat[k, pelvis], joint_pos[k, order]])
        fk = xml_forward_kinematics(xml, qpos, xml_joints)
        per_frame.append([np.linalg.norm(fk[b][0] - body_pos[k, isaac_bodies.index(b)]) for b in shared])
    errors = np.array(per_frame)
    worst = int(errors.max(axis=0).argmax())
    return {'npz': str(npz), 'xml': str(xml), 'fps': fps, 'frames_checked': int(len(errors)), 'stride': stride,
            'bodies_compared': len(shared), 'bodies_isaac': len(isaac_bodies), 'bodies_not_in_xml': [b for b in isaac_bodies if b not in shared],
            'max_error_m': float(errors.max()), 'mean_error_m': float(errors.mean()), 'worst_body': shared[worst],
            'per_body_max_m': {b: float(e) for b, e in zip(shared, errors.max(axis=0))}, 'pass_lt_1mm': bool(errors.max() < 1e-3)}


def main(argv=None) -> None:
    from humanoid_motion import ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--npz', type=Path, required=True)
    parser.add_argument('--xml', type=Path, default=ROOT / 'third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = check(args.npz, args.xml, args.stride)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('frames_checked', 'bodies_compared', 'max_error_m', 'worst_body', 'pass_lt_1mm', 'bodies_not_in_xml')}, indent=2))


if __name__ == '__main__':
    main()
