from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from humanoid_motion.contracts import CSV_COLUMNS, angular_velocity_world, named_indices, resample_csv, validate_tracker
from humanoid_motion.urdf_kinematics import UrdfKinematics


def constant_motion() -> np.ndarray:
    t = np.arange(31) / 30
    csv = np.zeros((31, 36))
    csv[:, 0] = 2 * t
    csv[:, 3:7] = (Rotation.from_rotvec(np.outer(t, [0.3, -0.4, 0.5])) *
                   Rotation.from_euler('xyz', [0.4, 0.6, -0.3])).as_quat()
    csv[::2, 3:7] *= -1  # Same orientation with deliberately discontinuous signs.
    csv[:, 7:] = t[:, None] * np.arange(29)[None, :] / 10
    return csv


def test_resampling_world_velocity_and_endpoint():
    out = resample_csv(constant_motion(), CSV_COLUMNS, 30)
    assert len(out['timestamps']) == 50
    assert out['timestamps'][-1] == pytest.approx(.98)
    np.testing.assert_allclose(out['root_link_lin_vel_w'], np.tile([2, 0, 0], (50, 1)), atol=1e-12)
    np.testing.assert_allclose(out['root_ang_vel_w'], np.tile([.3, -.4, .5], (50, 1)), atol=1e-12)
    np.testing.assert_allclose(out['joint_vel'], np.tile(np.arange(29) / 10, (50, 1)), atol=1e-12)


@pytest.mark.parametrize('bad', ['columns', 'nan', 'quaternion', 'fps', 'short'])
def test_invalid_source_rejected(bad):
    csv, columns, fps = constant_motion(), CSV_COLUMNS.copy(), 30
    if bad == 'columns':
        columns[-1], columns[-2] = columns[-2], columns[-1]
    elif bad == 'nan':
        csv[1, 7] = np.nan
    elif bad == 'quaternion':
        csv[1, 3:7] *= 2
    elif bad == 'fps':
        fps = 0
    else:
        csv = csv[:2]
    with pytest.raises(ValueError):
        resample_csv(csv, columns, fps)


def test_joint_permutation_is_named_not_length():
    assert named_indices(['a', 'b'], ['b', 'a']) == [1, 0]
    for source, target in [(['a', 'a'], ['a', 'b']), (['a', 'b'], ['a', 'c'])]:
        with pytest.raises(ValueError):
            named_indices(source, target)


def test_urdf_twist_against_infinitesimal_fk(tmp_path):
    path = tmp_path / 'robot.urdf'
    path.write_text('''<robot name="test">
      <link name="pelvis"><inertial><origin xyz=".1 -.2 .3"/></inertial></link>
      <link name="arm"><inertial><origin xyz=".4 .1 0"/></inertial></link>
      <link name="tip"/>
      <joint name="j" type="revolute"><parent link="pelvis"/><child link="arm"/>
        <origin xyz=".3 -.5 .4" rpy=".1 .2 .3"/><axis xyz="0 1 0"/>
        <limit lower="-2" upper="2"/></joint>
      <joint name="fixed" type="fixed"><parent link="arm"/><child link="tip"/>
        <origin xyz=".7 0 .2"/></joint></robot>''')
    model = UrdfKinematics(path)
    p, v, w = np.array([1., 2., 3.]), np.array([.4, -.2, .3]), np.array([.5, .3, -.4])
    r = Rotation.from_euler('xyz', [.3, -.5, .7])
    state = model.evaluate(p, r.as_quat()[[3, 0, 1, 2]], {'j': .6}, v, w, {'j': .8})
    eps = 1e-6
    nearby = []
    for dt in [-eps, eps]:
        rd = Rotation.from_rotvec(w * dt) * r
        nearby.append(model.evaluate(p + v * dt, rd.as_quat()[[3, 0, 1, 2]],
                                    {'j': .6 + .8 * dt}, v, w, {'j': .8}))
    for name, body in state.items():
        np.testing.assert_allclose((nearby[1][name]['p'] - nearby[0][name]['p']) / (2 * eps), body['v'], atol=1e-9)
        np.testing.assert_allclose((nearby[1][name]['com_p'] - nearby[0][name]['com_p']) / (2 * eps), body['com_v'], atol=1e-9)
        np.testing.assert_allclose((nearby[1][name]['r'] * nearby[0][name]['r'].inv()).as_rotvec() / (2 * eps), body['w'], atol=1e-9)


def test_tracker_requires_actual_order_frequency_and_finite_arrays():
    sidecar = {'joint_names': ['a'], 'body_names': ['pelvis']}
    data = dict(fps=np.array([50]), joint_pos=np.zeros((3, 1)), joint_vel=np.zeros((3, 1)),
                body_pos_w=np.zeros((3, 1, 3)), body_lin_vel_w=np.zeros((3, 1, 3)),
                body_ang_vel_w=np.zeros((3, 1, 3)), body_quat_w=np.tile([1., 0, 0, 0], (3, 1, 1)))
    validate_tracker(data, sidecar, ['a'], ['pelvis'], .02)
    with pytest.raises(ValueError):
        validate_tracker(data, sidecar, ['wrong'], ['pelvis'], .02)
    with pytest.raises(ValueError):
        validate_tracker(data, sidecar, ['a'], ['pelvis'], .04)
    data['body_lin_vel_w'][1, 0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_tracker(data, sidecar, ['a'], ['pelvis'], .02)
