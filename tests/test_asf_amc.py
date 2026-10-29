"""CMU ASF/AMC reader (humanoid_motion.retarget.asf_amc) on a synthetic CMU-shaped skeleton."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from humanoid_motion.retarget import asf_amc as A
from humanoid_motion.retarget.sources import _asf_scipy_fk

HIERARCHY = {'root': ['lhipjoint', 'rhipjoint', 'lowerback'], 'lhipjoint': ['lfemur'], 'lfemur': ['ltibia'],
             'ltibia': ['lfoot'], 'lfoot': ['ltoes'], 'rhipjoint': ['rfemur'], 'rfemur': ['rtibia'], 'rtibia': ['rfoot'],
             'rfoot': ['rtoes'], 'lowerback': ['upperback'], 'upperback': ['thorax'],
             'thorax': ['lowerneck', 'lclavicle', 'rclavicle'], 'lowerneck': ['upperneck'], 'upperneck': ['head'],
             'lclavicle': ['lhumerus'], 'lhumerus': ['lradius'], 'lradius': ['lwrist'], 'lwrist': ['lhand'],
             'rclavicle': ['rhumerus'], 'rhumerus': ['rradius'], 'rradius': ['rwrist'], 'rwrist': ['rhand']}
# Acclaim frame: Y up, facing +Z, left on +X.
DIRECTIONS = {'lhipjoint': (.6, -.7, .35), 'lfemur': (.34, -.94, 0), 'ltibia': (.34, -.94, 0), 'lfoot': (.09, -.25, .96),
              'ltoes': (0, 0, 1), 'lowerback': (0, 1, 0), 'upperback': (0, 1, 0), 'thorax': (0, 1, 0),
              'lowerneck': (0, 1, 0), 'upperneck': (0, 1, 0), 'head': (0, 1, 0), 'lclavicle': (.96, .26, .09),
              'lhumerus': (1, 0, 0), 'lradius': (1, 0, 0), 'lwrist': (1, 0, 0), 'lhand': (1, 0, 0)}
LEFT_BONES = ('lhipjoint', 'lfemur', 'ltibia', 'lfoot', 'ltoes', 'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand')
DIRECTIONS.update({'r' + k[1:]: (-x, y, z) for k, (x, y, z) in DIRECTIONS.items() if k in LEFT_BONES})
AXES = {'lfemur': (0, 0, 20), 'lhumerus': (180, -30, -90), 'lwrist': (0, 90, 90), 'rfemur': (0, 0, -20)}
NO_DOF = ('lhipjoint', 'rhipjoint')


def write_asf(path):
    bones = []
    for k, (name, direction) in enumerate(DIRECTIONS.items()):
        dof = '' if name in NO_DOF else '\n     dof rx ry rz'
        axis = ' '.join(str(v) for v in AXES.get(name, (0, 0, 0)))
        bones.append(f'  begin\n     id {k + 1}\n     name {name}\n     direction {" ".join(map(str, direction))}\n'
                     f'     length 2.5\n     axis {axis}  XYZ{dof}\n  end')
    hierarchy = '\n'.join(f'    {parent} {" ".join(children)}' for parent, children in HIERARCHY.items())
    path.write_text(':version 1.10\n:name VICON\n:units\n  mass 1.0\n  length 0.45\n  angle deg\n:root\n'
                    '   order TX TY TZ RX RY RZ\n   axis XYZ\n   position 0 0 0\n   orientation 0 0 0\n'
                    ':bonedata\n' + '\n'.join(bones) + f'\n:hierarchy\n  begin\n{hierarchy}\n  end\n')


def frames(rng, count, scale):
    rows = []
    for _ in range(count):
        row = {'root': np.concatenate([rng.normal(size=3) * 5 + [0, 16, 0], rng.normal(size=3) * scale])}
        row.update({name: rng.normal(size=3) * scale for name in DIRECTIONS if name not in NO_DOF})
        rows.append(row)
    return rows


def test_euler_order_is_rotation_about_fixed_x_then_y_then_z():
    angles = np.random.default_rng(0).uniform(-180, 180, size=(50, 3))
    assert np.abs(A.euler_xyz(angles) - Rotation.from_euler('xyz', angles, degrees=True).as_matrix()).max() < 1e-12


def test_rest_pose_gives_lafan1_joint_frames(tmp_path):
    write_asf(tmp_path / 's.asf')
    skeleton = A.parse_asf(tmp_path / 's.asf')
    assert skeleton.metres_per_unit == pytest.approx(0.0254 / 0.45)
    names, positions, rotations = A.lafan1_joints(skeleton, frames(np.random.default_rng(1), 1, 0.0))
    left, forward, up = A.LEFT, A.FORWARD, A.UP
    frame = dict(zip(names, rotations[0]))
    assert np.allclose(frame['Hips'], np.stack([up, forward, left], 1))
    assert np.allclose(frame['Spine2'], np.stack([up, forward, left], 1))
    assert np.allclose(frame['LeftArm'], np.stack([left, -forward, up], 1))
    assert np.allclose(frame['RightArm'], np.stack([-left, -forward, -up], 1))
    thigh = A.AXIS_MATRIX @ np.array(DIRECTIONS['lfemur']) / np.linalg.norm(DIRECTIONS['lfemur'])
    assert np.allclose(frame['LeftUpLeg'][:, 0], thigh) and frame['LeftUpLeg'][:, 1] @ forward > 0.9
    assert frame['LeftUpLeg'][:, 2] @ left < -0.9                                   # z points right, as in LAFAN1
    assert frame['LeftToe'][:, 1] @ up > 0.99 and frame['LeftFoot'][:, 1] @ up > 0.2
    for name, (_, bone, _, _) in A.LAFAN1_JOINTS.items():                           # x along the bone, right-handed
        if bone != 'root':
            assert np.allclose(frame[name][:, 0], A.AXIS_MATRIX @ skeleton.bones[bone].direction)
        assert np.allclose(frame[name].T @ frame[name], np.eye(3)) and np.linalg.det(frame[name]) == pytest.approx(1)
    knee = positions[0, names.index('LeftLeg')] - positions[0, names.index('LeftUpLeg')]
    assert np.allclose(knee, 2.5 * skeleton.metres_per_unit * thigh)


def test_fk_matches_a_scipy_implementation_and_turns_rigidly(tmp_path):
    write_asf(tmp_path / 's.asf')
    skeleton = A.parse_asf(tmp_path / 's.asf')
    rows = frames(np.random.default_rng(2), 20, 40.0)
    names, positions, rotations = A.lafan1_joints(skeleton, rows)
    other = _asf_scipy_fk(skeleton, rows, [A.LAFAN1_JOINTS[n][0] for n in names]) @ A.AXIS_MATRIX.T
    assert np.abs(positions - other).max() < 1e-9
    # A root yaw (about Acclaim Y) of 90 degrees turns every joint frame by 90 degrees about Z.
    still = frames(np.random.default_rng(3), 1, 0.0)
    turned = [dict(still[0], root=np.array([0, 16, 0, 0, 90, 0.]))]
    _, _, rest = A.lafan1_joints(skeleton, still)
    _, _, yawed = A.lafan1_joints(skeleton, turned)
    assert np.allclose(yawed[0], Rotation.from_euler('z', 90, degrees=True).as_matrix() @ rest[0])


def test_amc_reader_keeps_frame_order(tmp_path):
    (tmp_path / 'm.amc').write_text('#!OML:ASF x\n:FULLY-SPECIFIED\n:DEGREES\n1\nroot 0 1 2 3 4 5\nltibia 7\n'
                                    '2\nroot 6 7 8 9 10 11\nltibia -7\n')
    rows = A.parse_amc(tmp_path / 'm.amc')
    assert len(rows) == 2 and rows[1]['root'][0] == 6 and rows[1]['ltibia'][0] == -7


def test_toe_frames_follow_the_foot(tmp_path):
    """The recorded toe angle is ignored: toe frames move with the foot bone, the toe joint held at rest."""
    write_asf(tmp_path / 's.asf')
    skeleton = A.parse_asf(tmp_path / 's.asf')
    rows = frames(np.random.default_rng(4), 5, 30.0)
    names, _, rotations = A.lafan1_joints(skeleton, rows)
    curled = [dict(row, ltoes=row['ltoes'] + 50.0, rtoes=row['rtoes'] - 50.0) for row in rows]
    _, _, again = A.lafan1_joints(skeleton, curled)
    assert np.allclose(rotations, again)
    rest = A.rest_frames(skeleton)
    for side in ('Left', 'Right'):
        foot, toe = rotations[:, names.index(side + 'Foot')], rotations[:, names.index(side + 'Toe')]
        assert np.allclose(toe, foot @ rest[side + 'Foot'].T @ rest[side + 'Toe'])


def test_two_angle_joint_back_on_its_branch(tmp_path):
    """(x + 180, z + 180) on a joint without the middle angle is a flipped segment; it goes back to (x, z)."""
    write_asf(tmp_path / 's.asf')
    skeleton = A.parse_asf(tmp_path / 's.asf')
    skeleton.bones['lfoot'].dof = ['rx', 'rz']
    row = frames(np.random.default_rng(5), 1, 0.0)[0]
    good = dict(row, lfoot=np.array([-14., 3.]))
    flipped = dict(row, lfoot=np.array([-194., 183.]))
    fixed, moved = A.two_dof_branch(skeleton, [good, flipped, dict(row, lfoot=np.array([346., 3.]))])
    assert moved == {'lfoot': 1}
    assert all(np.allclose(f['lfoot'], [-14., 3.]) for f in fixed)
    _, _, rotations = A.lafan1_joints(skeleton, fixed)
    assert np.allclose(rotations[0], rotations[1])
