import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_motion import ROOT
from humanoid_motion.tracking import sim2sim


def test_constants_match_the_isaac_observation_module():
    source = (ROOT / 'src/humanoid_motion/tracking/observations.py').read_text()
    for name in ('FUTURE_FRAMES', 'FUTURE_STRIDE', 'HISTORY'):
        assert int(re.search(rf'^{name} = (\d+)', source, re.M).group(1)) == getattr(sim2sim, name)
    assert "OBSERVATION_NAMES = {'reference': 'v2', 'robot': 'v2.1'}" in source


def test_quaternion_helpers_match_scipy():
    rng = np.random.default_rng(0)
    r = Rotation.random(50, random_state=1)
    q = r.as_quat()[:, [3, 0, 1, 2]]
    v = rng.normal(size=(50, 3))
    np.testing.assert_allclose(sim2sim.quat_apply(q, v), r.apply(v), atol=1e-12)
    np.testing.assert_allclose(sim2sim.quat_apply(sim2sim.quat_conj(q), v), r.inv().apply(v), atol=1e-12)
    other = Rotation.random(50, random_state=2)
    product = sim2sim.quat_mul(q, other.as_quat()[:, [3, 0, 1, 2]])
    np.testing.assert_allclose(np.abs(np.sum(product * (r * other).as_quat()[:, [3, 0, 1, 2]], 1)), 1, atol=1e-12)
    np.testing.assert_allclose(sim2sim.matrix_columns(q), r.as_matrix()[:, :, :2].reshape(50, 6), atol=1e-12)
    for k in range(5):
        yaw = sim2sim.yaw_quat(q[k])
        heading = Rotation.from_quat(yaw[[1, 2, 3, 0]]).apply([1., 0, 0])
        forward = r[k].apply([1., 0, 0])
        assert abs(heading[2]) < 1e-12
        np.testing.assert_allclose(heading[:2] / np.linalg.norm(heading[:2]),
                                   forward[:2] / np.linalg.norm(forward[:2]), atol=1e-9)
