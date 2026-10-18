from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
from humanoid_motion.geometry import named_csv, source_fk, xml_forward_kinematics


def test_named_permutation_and_quaternion_contract():
    q = np.array([[1, 2, 3, .5, .1, .2, .3, 7, 8]], dtype=float)
    csv = named_csv(q, ['b', 'a'], ['a', 'b'])
    np.testing.assert_array_equal(csv, [[1, 2, 3, .1, .2, .3, .5, 8, 7]])
    with pytest.raises(ValueError):
        named_csv(q, ['a', 'b'], ['a', 'c'])
    with pytest.raises(ValueError):
        named_csv(q, ['a', 'a'], ['a', 'b'])
    with pytest.raises(ValueError):
        named_csv(q, ['a', 'b'], ['a', 'b', 'b'])
    with pytest.raises(ValueError):
        named_csv(np.zeros((2, 10)), ['a', 'b'], ['a', 'b'])


def test_known_quarter_turn_source_fk():
    pos = np.array([[[2,3,4], [1,0,0]]], dtype=float)
    quat = np.array([[[2**-.5,0,0,2**-.5], [1,0,0,0]]])
    actual = source_fk(pos, quat, np.array([-1,0]))
    np.testing.assert_allclose(actual, [[[2,3,4],[2,4,4]]], atol=1e-14)


def test_known_hinge_and_free_root_xml(tmp_path):
    xml = tmp_path / 'simple.xml'
    xml.write_text('''<mujoco><compiler angle="radian"/><worldbody>
    <body name="root"><freejoint/><body name="hinge" pos="1 0 0">
    <joint name="j" axis="0 0 1"/><body name="tip" pos="2 0 0"/>
    </body></body></worldbody></mujoco>''')
    fk = xml_forward_kinematics(xml, np.array([3,4,5,1,0,0,0,np.pi/2]), ['j'])
    np.testing.assert_allclose(fk['tip'][0], [4,6,5], atol=1e-14)


def test_real_run_preserves_frames_and_roundtrips():
    run = ROOT / 'outputs/walk1_s1_f300_480'
    if not run.exists():
        pytest.skip('Generate the licensed local dataset run first')
    import json
    diag = json.loads((run / 'diagnostics.json').read_text())
    ref = np.load(run / 'reference.npz')
    assert diag['frames_saved'] == diag['frames_requested'] == 180
    assert ref['qpos'].shape == (180, 36)
    assert np.isfinite(ref['qpos']).all()
    assert diag['csv_named_roundtrip_max_error'] < 1e-12
    assert diag['source_fk_independent_max_error_m'] < 1e-10
    assert diag['robot_fk_independent_position_error_m']['max'] < 1e-10
    assert diag['quaternion_norm_max_error'] < 1e-12
    assert diag['joint_limit_violations_gt_1e-6_rad'] == 0
    other = ROOT / 'outputs/walk1_s1_f300_480_repeat/reference.npz'
    if other.exists():
        np.testing.assert_allclose(ref['qpos'], np.load(other)['qpos'], rtol=0, atol=1e-12)
