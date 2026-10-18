"""Independent rollout metrics; inputs must be time-matched world SI states."""
import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_motion.contracts import check_quaternions


def validate_transfer_contract(training: dict, test: dict) -> None:
    """Permit a different clip only with the same named robot/control interface."""
    if not training.get('audit_passed') or not test.get('audit_passed'):
        raise ValueError('Both motion interfaces must have passed their independent audits')
    for key in ('schema', 'tracking_commit', 'urdf_sha256', 'joint_names', 'body_names',
                'root_body', 'anchor_body', 'fps', 'coordinates', 'control',
                'runtime_com_offsets_b_m'):
        if key not in training or key not in test or training[key] != test[key]:
            raise ValueError(f'Transfer interface mismatch: {key}')


def pose_errors(reference_pos: np.ndarray, reference_quat: np.ndarray,
                actual_pos: np.ndarray, actual_quat: np.ndarray,
                body_names: list[str], root: str = 'pelvis', anchor: str = 'torso_link') -> dict[str, np.ndarray]:
    """Return errors per frame/body, preserving both raw world and tracker alignment.

    Quaternions are wxyz. Alignment uses yaw(actual_anchor * inverse(ref_anchor)),
    translates reference anchor XY to actual XY, and preserves reference Z.
    It therefore cannot be used as a measure of world-path accuracy.
    """
    if len(set(body_names)) != len(body_names) or root not in body_names or anchor not in body_names:
        raise ValueError('Missing/duplicate root, anchor or body names')
    if reference_pos.ndim != 3 or reference_pos.shape[1:] != (len(body_names), 3):
        raise ValueError('Expected frame × named body × world XYZ')
    n, bodies = reference_pos.shape[:2]
    if n < 1 or actual_pos.shape != reference_pos.shape:
        raise ValueError('Reference and actual frames must be explicitly time-matched')
    if reference_quat.shape != (n, bodies, 4) or actual_quat.shape != reference_quat.shape:
        raise ValueError('Quaternion shape mismatch')
    if not np.isfinite(reference_pos).all() or not np.isfinite(actual_pos).all():
        raise ValueError('Nonfinite position')
    check_quaternions(reference_quat)
    check_quaternions(actual_quat)
    r_ref = Rotation.from_quat(reference_quat.reshape(-1, 4)[:, [1, 2, 3, 0]])
    r_actual = Rotation.from_quat(actual_quat.reshape(-1, 4)[:, [1, 2, 3, 0]])
    anchor_i, root_i = body_names.index(anchor), body_names.index(root)
    anchor_ids = np.arange(n) * bodies + anchor_i
    difference = (r_actual[anchor_ids] * r_ref[anchor_ids].inv()).as_matrix()
    yaw = np.arctan2(difference[:, 1, 0], difference[:, 0, 0])
    alignment = Rotation.from_euler('z', np.repeat(yaw, bodies))
    shifted = actual_pos[:, anchor_i].copy()
    shifted[:, 2] = reference_pos[:, anchor_i, 2]
    relative = reference_pos - reference_pos[:, anchor_i:anchor_i+1]
    aligned_pos = shifted[:, None] + alignment.apply(relative.reshape(-1, 3)).reshape(n, bodies, 3)
    world_delta = actual_pos - reference_pos
    return dict(world_root_position_m=np.linalg.norm(world_delta[:, root_i], axis=-1),
                world_root_xy_m=np.linalg.norm(world_delta[:, root_i, :2], axis=-1),
                world_anchor_position_m=np.linalg.norm(world_delta[:, anchor_i], axis=-1),
                world_body_position_m=np.linalg.norm(world_delta, axis=-1),
                aligned_body_position_m=np.linalg.norm(actual_pos - aligned_pos, axis=-1),
                world_body_orientation_rad=(r_actual * r_ref.inv()).magnitude().reshape(n, bodies),
                aligned_body_orientation_rad=(r_actual * (alignment * r_ref).inv()).magnitude().reshape(n, bodies))


def completion_counts(attempts: list[dict], expected_frames: int) -> dict:
    """Retain every attempt, including runtime failures and assisted completions.

    Each record must include observed reference frame indices, terminal reason,
    and post-initialization state-write/assistance counts. A failed run can have
    an empty frame list. Initialization at frame 0 is not a hidden assistance.
    """
    if expected_frames < 2:
        raise ValueError('Need a complete clip with >=2 frames')
    reasons = {}
    complete = assisted = 0
    for attempt in attempts:
        reason = attempt['terminal_reason']
        if reason not in ('clip_complete', 'terminated', 'truncated', 'runtime_error', 'user_interrupt'):
            raise ValueError(f'Unknown termination reason: {reason}')
        reasons[reason] = reasons.get(reason, 0) + 1
        resets, assistance = attempt['state_writes_after_initialization'], attempt['assistance_count']
        if min(resets, assistance) < 0:
            raise ValueError('Negative reset/assistance count')
        aided = resets > 0 or assistance > 0
        assisted += int(aided)
        full = np.array_equal(np.asarray(attempt['reference_frame_indices']), np.arange(expected_frames))
        complete += int(full and reason == 'clip_complete' and not aided)
    return dict(attempts=len(attempts), unassisted_completions=complete, assisted_attempts=assisted,
                completion_rate=None if not attempts else complete / len(attempts), terminal_reasons=reasons)
