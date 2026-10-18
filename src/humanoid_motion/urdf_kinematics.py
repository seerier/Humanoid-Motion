"""Independent G1 URDF pose/twist propagation, using explicit named joints."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


class UrdfKinematics:
    def __init__(self, path: Path):
        tree = ET.parse(path).getroot()
        self.links = [node.attrib['name'] for node in tree.findall('link')]
        self.com = {}
        for link in tree.findall('link'):
            origin = link.find('inertial/origin')
            self.com[link.attrib['name']] = np.fromstring('0 0 0' if origin is None else origin.get('xyz', '0 0 0'), sep=' ')
        self.joints = []
        for node in tree.findall('joint'):
            kind = node.attrib['type']
            if kind not in ('revolute', 'fixed'):
                raise ValueError(f'Unsupported joint type {kind}')
            origin, axis, limit = node.find('origin'), node.find('axis'), node.find('limit')
            def vector(key: str) -> np.ndarray:
                return np.fromstring('0 0 0' if origin is None else origin.get(key, '0 0 0'), sep=' ')
            axis_v = np.fromstring('1 0 0' if axis is None else axis.get('xyz'), sep=' ')
            self.joints.append(dict(name=node.attrib['name'], kind=kind,
                parent=node.find('parent').attrib['link'], child=node.find('child').attrib['link'],
                p=vector('xyz'), r=Rotation.from_euler('xyz', vector('rpy')),
                axis=axis_v / np.linalg.norm(axis_v),
                limits=None if kind == 'fixed' else [float(limit.attrib[k]) for k in ('lower', 'upper')]))
        roots = set(self.links) - {j['child'] for j in self.joints}
        if len(roots) != 1:
            raise ValueError('Expected a single URDF root')
        self.root = roots.pop()
        self.joint_names = [j['name'] for j in self.joints if j['kind'] == 'revolute']

    def evaluate(self, root_pos: np.ndarray, root_quat: np.ndarray, angles: dict[str, float],
                 root_velocity: np.ndarray, root_omega: np.ndarray,
                 velocities: dict[str, float]) -> dict[str, dict]:
        if set(angles) != set(self.joint_names) or set(velocities) != set(self.joint_names):
            raise ValueError('URDF joint name mismatch')
        state = {self.root: dict(p=root_pos, r=Rotation.from_quat(root_quat[[1, 2, 3, 0]]),
                                v=root_velocity, w=root_omega)}
        pending = list(self.joints)
        while pending:
            ready = [j for j in pending if j['parent'] in state]
            if not ready:
                raise ValueError('Disconnected URDF')
            for j in ready:
                parent = state[j['parent']]
                offset = parent['r'].apply(j['p'])
                rotation = parent['r'] * j['r']
                omega = parent['w'].copy()
                if j['kind'] == 'revolute':
                    omega += rotation.apply(j['axis']) * velocities[j['name']]
                    rotation = rotation * Rotation.from_rotvec(j['axis'] * angles[j['name']])
                state[j['child']] = dict(p=parent['p'] + offset, r=rotation,
                    v=parent['v'] + np.cross(parent['w'], offset), w=omega)
                pending.remove(j)
        for name, body in state.items():
            offset = body['r'].apply(self.com[name])
            body['com_p'] = body['p'] + offset
            body['com_v'] = body['v'] + np.cross(body['w'], offset)
        return state
