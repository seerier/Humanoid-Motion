"""CMU Graphics Lab motion capture (Acclaim ASF skeleton + AMC motion) as GMR source frames.

GMR has no ASF/AMC loader, and the AMASS release of the CMU database leaves out some trials (the standing backflips
87_03, 87_04, 88_01, for example). This module reads the original files and hands GMR frames in the shape and the
joint conventions of its LAFAN1 loader, so GMR's own bvh_lafan1 IK table retargets them unmodified.

Acclaim conventions, as the CMU files use them:

- Units: ASF lengths are inches divided by ':units length' (0.45 in CMU files), so metres = value * 0.0254 / length.
  Angles in degrees. Y up; the rest pose is a T-pose facing +Z with the subject's left on +X.
- Each bone has a rest direction (world frame), a length, an 'axis' rotation C (Euler angles, order XYZ) and up to
  three rotational degrees of freedom. With R(a) = Rz(a_z) Ry(a_y) Rx(a_x) (rotations about the fixed X, then Y, then
  Z axes), a bone's world rotation is parent_world * C R(dof) C^-1, and its end lies at parent_end + length *
  world * direction. The root carries its own translation and R(root rx, ry, rz). At the rest pose every world
  rotation is the identity.

LAFAN1 conventions, which GMR's bvh_lafan1 rotation offsets assume (read from LAFAN1's own T-pose frames, where
every file starts): a joint's frame has its x axis along the bone towards the child, and its y axis on a fixed body
direction: forward for the pelvis, spine, neck, head, thighs and shanks; up for the feet and toes; backward for the
clavicles, arms, forearms and hands. Z is x cross y. Here each CMU bone gets that frame at the CMU rest pose, with x
on the CMU bone's own rest direction, and keeps it rigidly as the bone moves. Positions are joint origins, as in a
BVH: the start of the matching CMU bone. GMR's foot targets (LeftFootMod: ankle position, toe orientation) are built
as its LAFAN1 loader builds them.

One departure: the toe frames follow the foot bone, with the toe joint held at its rest angle (toes straight ahead of
the foot, as when standing). The recorded CMU toe angles are unreliable: standing still at the end of 88_01, the AMC
toe joints sit 30-60 degrees from rest, and since GMR's foot target takes the toe orientation, plain use of them
drives the G1 ankle pitch to its limits.

Euler branch flips (two_dof_branch): some AMC files switch a two-angle joint to the other Euler branch, both angles
shifted by 180 degrees, which is the same rotation only together with the 180 degree middle angle the joint does not
have. In 88_01 the left foot's (rx, rz) goes from about (-14, 3) to (-194, 183) after the flip and stays there,
turning the foot backwards on the shank. two_dof_branch puts such frames back on the branch with both angles within
90 degrees; the loader counts them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

INCH_M = 0.0254
# Acclaim Y-up -> Z-up with the matrix GMR's LAFAN1 loader uses: (x, y, z) -> (x, -z, y). The CMU rest pose then
# faces -y with its left on +x, as LAFAN1's T-pose frames do.
AXIS_MATRIX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
LEFT, FORWARD, UP = np.array([1., 0, 0]), np.array([0., -1, 0]), np.array([0., 0, 1])

# LAFAN1 joint -> (CMU bone whose start is the joint position, CMU bone whose rest direction is the frame's x axis,
# CMU bone whose rotation moves the frame, y axis reference direction). 'root' is the CMU root (pelvis). Order and
# parents are LAFAN1's (GMR's lafan_vendor reader).
LAFAN1_JOINTS = {
    'Hips': ('root', 'root', 'root', FORWARD),
    'LeftUpLeg': ('lfemur', 'lfemur', 'lfemur', FORWARD), 'LeftLeg': ('ltibia', 'ltibia', 'ltibia', FORWARD),
    'LeftFoot': ('lfoot', 'lfoot', 'lfoot', UP), 'LeftToe': ('ltoes', 'ltoes', 'lfoot', UP),
    'RightUpLeg': ('rfemur', 'rfemur', 'rfemur', FORWARD), 'RightLeg': ('rtibia', 'rtibia', 'rtibia', FORWARD),
    'RightFoot': ('rfoot', 'rfoot', 'rfoot', UP), 'RightToe': ('rtoes', 'rtoes', 'rfoot', UP),
    'Spine': ('lowerback', 'lowerback', 'lowerback', FORWARD), 'Spine1': ('upperback', 'upperback', 'upperback', FORWARD),
    'Spine2': ('thorax', 'thorax', 'thorax', FORWARD), 'Neck': ('lowerneck', 'lowerneck', 'lowerneck', FORWARD),
    'Head': ('head', 'head', 'head', FORWARD),
    'LeftShoulder': ('lclavicle', 'lclavicle', 'lclavicle', -FORWARD),
    'LeftArm': ('lhumerus', 'lhumerus', 'lhumerus', -FORWARD),
    'LeftForeArm': ('lradius', 'lradius', 'lradius', -FORWARD), 'LeftHand': ('lwrist', 'lhand', 'lhand', -FORWARD),
    'RightShoulder': ('rclavicle', 'rclavicle', 'rclavicle', -FORWARD),
    'RightArm': ('rhumerus', 'rhumerus', 'rhumerus', -FORWARD),
    'RightForeArm': ('rradius', 'rradius', 'rradius', -FORWARD), 'RightHand': ('rwrist', 'rhand', 'rhand', -FORWARD),
}
LAFAN1_PARENTS = {'Hips': None, 'LeftUpLeg': 'Hips', 'LeftLeg': 'LeftUpLeg', 'LeftFoot': 'LeftLeg', 'LeftToe': 'LeftFoot',
                  'RightUpLeg': 'Hips', 'RightLeg': 'RightUpLeg', 'RightFoot': 'RightLeg', 'RightToe': 'RightFoot',
                  'Spine': 'Hips', 'Spine1': 'Spine', 'Spine2': 'Spine1', 'Neck': 'Spine2', 'Head': 'Neck',
                  'LeftShoulder': 'Spine2', 'LeftArm': 'LeftShoulder', 'LeftForeArm': 'LeftArm', 'LeftHand': 'LeftForeArm',
                  'RightShoulder': 'Spine2', 'RightArm': 'RightShoulder', 'RightForeArm': 'RightArm',
                  'RightHand': 'RightForeArm'}


@dataclass
class Bone:
    name: str
    direction: np.ndarray        # unit rest direction, ASF world frame
    length: float                # ASF units
    axis_deg: np.ndarray         # C as XYZ Euler angles
    dof: list[str]               # e.g. ['rx', 'ry', 'rz']


@dataclass
class Skeleton:
    bones: dict[str, Bone]
    children: dict[str, list[str]]
    metres_per_unit: float
    root_order: list[str]
    root_axis_deg: np.ndarray
    root_position: np.ndarray


def euler_xyz(degrees: np.ndarray) -> np.ndarray:
    """Rz(z) Ry(y) Rx(x) for (..., 3) angles in degrees: rotations about the fixed X, Y, then Z axes."""
    x, y, z = np.radians(np.moveaxis(np.asarray(degrees, float), -1, 0))
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    return np.stack([np.stack([cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx], -1),
                     np.stack([sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], -1),
                     np.stack([-sy, cy * sx, cy * cx], -1)], -2)


def parse_asf(path: Path) -> Skeleton:
    lines = [line.split('#', 1)[0].strip() for line in Path(path).read_text().splitlines()]
    lines = [line for line in lines if line]
    bones, children, section, current = {}, {}, None, None
    length_unit, angle_unit = 1.0, 'deg'
    root_order, root_axis, root_position = [], np.zeros(3), np.zeros(3)
    for line in lines:
        if line.startswith(':'):
            word = line.split()
            section = word[0]
            continue
        word = line.split()
        if section == ':units':
            if word[0] == 'length':
                length_unit = float(word[1])
            elif word[0] == 'angle':
                angle_unit = word[1]
        elif section == ':root':
            if word[0] == 'order':
                root_order = [w.lower() for w in word[1:]]
            elif word[0] == 'axis' and word[1].upper() != 'XYZ':
                raise ValueError(f'root axis order {word[1]}; only XYZ is supported')
            elif word[0] == 'position':
                root_position = np.array(word[1:4], float)
            elif word[0] == 'orientation':
                root_axis = np.array(word[1:4], float)
        elif section == ':bonedata':
            if word[0] == 'begin':
                current = {'dof': []}
            elif word[0] == 'end':
                bones[current['name']] = Bone(current['name'], current['direction'] / np.linalg.norm(current['direction']),
                                              current['length'], current['axis'], current['dof'])
            elif word[0] == 'name':
                current['name'] = word[1]
            elif word[0] == 'direction':
                current['direction'] = np.array(word[1:4], float)
            elif word[0] == 'length':
                current['length'] = float(word[1])
            elif word[0] == 'axis':
                if word[4].upper() != 'XYZ':
                    raise ValueError(f"bone {current.get('name')} axis order {word[4]}; only XYZ is supported")
                current['axis'] = np.array(word[1:4], float)
            elif word[0] == 'dof':
                current['dof'] = [w.lower() for w in word[1:]]
        elif section == ':hierarchy':
            if word[0] not in ('begin', 'end'):
                children.setdefault(word[0], []).extend(word[1:])
    if angle_unit.lower() != 'deg':
        raise ValueError(f'ASF angle unit {angle_unit}; only degrees are supported')
    if root_order[:3] != ['tx', 'ty', 'tz'] or sorted(root_order[3:]) != ['rx', 'ry', 'rz']:
        raise ValueError(f'root order {root_order}; expected TX TY TZ and three rotations')
    return Skeleton(bones, children, INCH_M / length_unit, root_order, root_axis, root_position)


def parse_amc(path: Path) -> list[dict[str, np.ndarray]]:
    frames, current = [], None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or line.startswith(':'):
            continue
        word = line.split()
        if len(word) == 1 and word[0].isdigit():
            current = {}
            frames.append(current)
            continue
        if current is None:
            raise ValueError(f'{path}: values before the first frame number: {line!r}')
        current[word[0]] = np.array(word[1:], float)
    return frames


def two_dof_branch(skeleton: Skeleton, frames: list[dict[str, np.ndarray]]):
    """Frames with every angle wrapped to (-180, 180], and two-angle joints whose angles both exceed 90 degrees moved
    to the other branch (both minus 180); returns the frames and the number of moved frames per bone."""
    moved, out = {}, []
    for frame in frames:
        row = dict(frame)
        for name, bone in skeleton.bones.items():
            if not bone.dof or name not in frame:
                continue
            values = (np.asarray(frame[name], float) + 180.) % 360. - 180.
            if len(bone.dof) == 2 and np.all(np.abs(values) > 90.):
                values = values % 360. - 180.
                moved[name] = moved.get(name, 0) + 1
            row[name] = values
        out.append(row)
    return out, moved


def forward_kinematics(skeleton: Skeleton, frames: list[dict[str, np.ndarray]]):
    """World rotations (bone -> (T, 3, 3)) and bone start / end points in metres, ASF frame (Y up)."""
    count = len(frames)
    root_values = np.stack([f['root'] for f in frames])
    index = {name: k for k, name in enumerate(skeleton.root_order)}
    root_t = root_values[:, [index['tx'], index['ty'], index['tz']]]
    root_r = root_values[:, [index['rx'], index['ry'], index['rz']]]
    c_root = euler_xyz(skeleton.root_axis_deg)
    rotation = {'root': c_root @ euler_xyz(root_r) @ c_root.T}
    start = {'root': (skeleton.root_position + root_t) * skeleton.metres_per_unit}
    end = {'root': start['root']}
    queue = ['root']
    while queue:
        parent = queue.pop(0)
        for name in skeleton.children.get(parent, []):
            bone = skeleton.bones[name]
            angles = np.zeros((count, 3))
            if bone.dof:
                values = np.stack([f[name] for f in frames])
                if values.shape[1] != len(bone.dof):
                    raise ValueError(f'{name}: {values.shape[1]} values per frame for dof {bone.dof}')
                for k, dof in enumerate(bone.dof):
                    angles[:, 'xyz'.index(dof[1])] = values[:, k]
            c = euler_xyz(bone.axis_deg)
            rotation[name] = rotation[parent] @ c @ euler_xyz(angles) @ c.T
            start[name] = end[parent]
            end[name] = start[name] + bone.length * skeleton.metres_per_unit * (rotation[name] @ bone.direction)
            queue.append(name)
    return rotation, start, end


def rest_frames(skeleton: Skeleton) -> dict[str, np.ndarray]:
    """Each LAFAN1 joint's frame at the CMU rest pose (Z up): x on the CMU bone's rest direction (up for the pelvis),
    y on the joint's reference direction made orthogonal to x, z = x cross y."""
    frames = {}
    for joint, (_, bone, _, reference) in LAFAN1_JOINTS.items():
        x = UP if bone == 'root' else AXIS_MATRIX @ skeleton.bones[bone].direction
        y = reference - (reference @ x) * x
        if np.linalg.norm(y) < 0.3:
            raise ValueError(f'{joint}: CMU rest direction of {bone} is too close to its reference direction')
        y = y / np.linalg.norm(y)
        frames[joint] = np.stack([x, y, np.cross(x, y)], axis=1)
    return frames


def lafan1_joints(skeleton: Skeleton, frames: list[dict[str, np.ndarray]]):
    """LAFAN1-named joint positions (T, J, 3) in metres, Z up, and world rotation matrices (T, J, 3, 3)."""
    rotation, start, _ = forward_kinematics(skeleton, frames)
    rest = rest_frames(skeleton)
    names = list(LAFAN1_JOINTS)
    positions = np.stack([start[LAFAN1_JOINTS[n][0]] @ AXIS_MATRIX.T for n in names], axis=1)
    rotations = np.stack([AXIS_MATRIX @ rotation[LAFAN1_JOINTS[n][2]] @ AXIS_MATRIX.T @ rest[n] for n in names], axis=1)
    return names, positions, rotations


def skeleton_extent_m(skeleton: Skeleton) -> dict:
    """Rest-pose heights from the skeleton alone: top of the head bone and the lowest foot or toe end above the
    root. The head bone ends inside the head, so this is less than the subject's stature."""
    _, _, end = forward_kinematics(skeleton, [{'root': np.zeros(6), **{n: np.zeros(len(b.dof)) for n, b in
                                                                        skeleton.bones.items() if b.dof}}])
    root = float(end['root'][0, 1])
    head = float(end['head'][0, 1]) - root
    lowest = float(min(end[n][0, 1] for n in ('lfoot', 'rfoot', 'ltoes', 'rtoes'))) - root
    return {'head_end_above_root_m': head, 'lowest_foot_end_above_root_m': lowest,
            'head_end_to_lowest_foot_m': head - lowest}
