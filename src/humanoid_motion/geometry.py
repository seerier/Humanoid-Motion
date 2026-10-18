"""Independent XML forward kinematics and explicit reference-format boundaries."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def xml_forward_kinematics(xml_path: Path, qpos: np.ndarray, joint_names: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """FK using XML body transforms + scalar hinges, independently of MuJoCo FK.

    Restricted deliberately to the audited G1 model: radian compiler, quaternion
    body rotations, zero-position hinge axes. Reject unsupported joint schemas.
    """
    root = ET.parse(xml_path).getroot()
    assert root.find('compiler').get('angle') == 'radian'
    values = dict(zip(joint_names, qpos[7:], strict=True))
    result = {}

    def visit(body: ET.Element, parent_p: np.ndarray, parent_r: Rotation) -> None:
        local_p = np.fromstring(body.get('pos', '0 0 0'), sep=' ')
        local_r = Rotation.from_quat(np.fromstring(body.get('quat', '1 0 0 0'), sep=' '), scalar_first=True)
        p, r = parent_p + parent_r.apply(local_p), parent_r * local_r
        free = [joint for joint in body.findall('joint') if joint.get('type') == 'free']   # <joint type="free"/> == <freejoint/>
        if body.find('freejoint') is not None or free:
            p, r = qpos[:3].copy(), Rotation.from_quat(qpos[3:7], scalar_first=True)
        for joint in body.findall('joint'):
            if joint in free:
                continue
            assert joint.get('type', 'hinge') == 'hinge'
            assert np.allclose(np.fromstring(joint.get('pos', '0 0 0'), sep=' '), 0)
            axis = np.fromstring(joint.get('axis', '0 0 1'), sep=' ')
            r = r * Rotation.from_rotvec(axis / np.linalg.norm(axis) * values[joint.get('name')])
        result[body.get('name')] = p, r.as_quat(scalar_first=True)
        for child in body.findall('body'):
            visit(child, p, r)

    for body in root.find('worldbody').findall('body'):
        visit(body, np.zeros(3), Rotation.identity())
    return result


def named_csv(qpos: np.ndarray, joint_names: list[str], expected_names: list[str]) -> np.ndarray:
    if (len(set(joint_names)) != len(joint_names)
            or len(set(expected_names)) != len(expected_names)
            or set(joint_names) != set(expected_names)):
        raise ValueError('Joint-name contract mismatch')
    if qpos.ndim != 2 or qpos.shape[1] != 7 + len(joint_names):
        raise ValueError('qpos columns do not match the free-root and named-joint contract')
    indices = [joint_names.index(name) for name in expected_names]
    return np.concatenate([qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:][:, indices]], axis=1)


def source_fk(local_positions: np.ndarray, local_quaternions: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Independent source FK with SciPy, returns original BVH coordinates."""
    positions, rotations = [], []
    for j, parent in enumerate(parents):
        r = Rotation.from_quat(local_quaternions[:, j], scalar_first=True)
        if parent < 0:
            positions.append(local_positions[:, j])
            rotations.append(r)
        else:
            positions.append(positions[parent] + rotations[parent].apply(local_positions[:, j]))
            rotations.append(rotations[parent] * r)
    return np.stack(positions, axis=1)
