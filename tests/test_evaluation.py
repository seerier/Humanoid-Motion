import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_motion.evaluation import completion_counts, pose_errors, validate_transfer_contract


def test_alignment_does_not_hide_world_drift_or_vertical_error():
    names = ['pelvis', 'torso_link', 'left_ankle_roll_link']
    reference = np.array([[[0., 0., .8], [.1, 0., 1.1], [-.2, 0., .1]]])
    reference_q = np.tile([1., 0, 0, 0], (1, 3, 1))
    yaw = Rotation.from_euler('z', .5)
    actual = yaw.apply(reference.reshape(-1, 3)).reshape(reference.shape) + [3, 4, 0]
    actual_q = np.tile(yaw.as_quat()[[3, 0, 1, 2]], (1, 3, 1))
    error = pose_errors(reference, reference_q, actual, actual_q, names)
    np.testing.assert_allclose(error['world_root_xy_m'], 5)
    np.testing.assert_allclose(error['aligned_body_position_m'], 0, atol=1e-14)
    np.testing.assert_allclose(error['world_body_orientation_rad'], .5, atol=1e-14)
    np.testing.assert_allclose(error['aligned_body_orientation_rad'], 0, atol=1e-14)
    actual[:, :, 2] += .2
    raised = pose_errors(reference, reference_q, actual, actual_q, names)
    np.testing.assert_allclose(raised['aligned_body_position_m'], .2, atol=1e-14)


def test_completion_denominator_keeps_failures_assistance_and_wrong_phase():
    success = dict(reference_frame_indices=[0, 1, 2], terminal_reason='clip_complete',
                   state_writes_after_initialization=0, assistance_count=0)
    attempts = [success, dict(success, state_writes_after_initialization=1),
                dict(success, reference_frame_indices=[1, 2]),
                dict(success, reference_frame_indices=[], terminal_reason='runtime_error'),
                dict(success, reference_frame_indices=[0, 1], terminal_reason='terminated')]
    result = completion_counts(attempts, 3)
    assert result['attempts'] == 5
    assert result['unassisted_completions'] == 1
    assert result['completion_rate'] == .2
    assert result['assisted_attempts'] == 1
    assert completion_counts([], 3)['completion_rate'] is None


def test_transfer_rejects_semantic_mismatch_despite_equal_array_sizes():
    import copy
    import pytest
    manifest = dict(audit_passed=True, schema='humanoid_motion.isaac_tracking.v1',
        tracking_commit='pinned', urdf_sha256='robot', joint_names=['left', 'right'],
        body_names=['pelvis', 'torso_link'], root_body='pelvis', anchor_body='torso_link',
        fps=50, coordinates=dict(quaternion='wxyz'), control=dict(action_offset='default_joint_pos'),
        runtime_com_offsets_b_m=[[0., 0., 0.], [0., 0., .1]], motion_sha256='training')
    changed = copy.deepcopy(manifest)
    changed['motion_sha256'] = 'new-clip'
    validate_transfer_contract(manifest, changed)
    for field, replacement in [
        ('joint_names', list(reversed(manifest['joint_names']))),
        ('body_names', list(reversed(manifest['body_names']))),
        ('fps', 30), ('coordinates', dict(quaternion='xyzw')),
        ('control', dict(action_offset='reference_joint_pos')),
        ('audit_passed', False), ('urdf_sha256', 'different-robot'),
    ]:
        wrong = dict(changed, **{field: replacement})
        with pytest.raises(ValueError):
            validate_transfer_contract(manifest, wrong)
