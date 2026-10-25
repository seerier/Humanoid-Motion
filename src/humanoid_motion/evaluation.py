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
    alignment = Rotation.from_euler('z', np.repeat(yaw, bodies)[:, None])  # (N, 1): one angle per rotation, any SciPy
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


EE_BODIES = ('left_ankle_roll_link', 'right_ankle_roll_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link')
# Failure criteria scored offline from recorded states. 'beyondmimic' is the pinned tracker's termination set that
# the online evaluator stops at; 'anchor' is the SuperMimic paper's (torso height and tilt only, no end effectors).
CRITERIA = {'beyondmimic': ('anchor_pos', 'anchor_ori', 'ee_body_pos'), 'anchor': ('anchor_pos', 'anchor_ori')}


def failure_terms(reference_pos: np.ndarray, reference_quat: np.ndarray, actual_pos: np.ndarray,
                  actual_quat: np.ndarray, body_names: list[str], anchor: str = 'torso_link',
                  height_m: float = .25, tilt: float = .8) -> dict[str, np.ndarray]:
    """Per-frame BeyondMimic failure terms (whole_body_tracking terminations) from time-matched world states.

    anchor_pos: anchor height differs by more than height_m. anchor_ori: the z components of gravity in the two
    anchor frames differ by more than tilt. ee_body_pos: an ankle or wrist height differs by more than height_m;
    the tracker compares against the heading-aligned reference, whose heights equal the raw reference heights.
    """
    a = body_names.index(anchor)
    ee = [body_names.index(name) for name in EE_BODIES]
    def up_z(q):  # R[2, 2] of wxyz quaternions = -(gravity direction in the body frame)_z
        return 1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2)
    return dict(anchor_pos=np.abs(reference_pos[:, a, 2] - actual_pos[:, a, 2]) > height_m,
                anchor_ori=np.abs(up_z(reference_quat[:, a]) - up_z(actual_quat[:, a])) > tilt,
                ee_body_pos=(np.abs(reference_pos[:, ee, 2] - actual_pos[:, ee, 2]) > height_m).any(axis=1))


def score_attempt(reference_pos: np.ndarray, reference_quat: np.ndarray, actual_pos: np.ndarray,
                  actual_quat: np.ndarray, body_names: list[str], criterion: str = 'beyondmimic') -> dict:
    """One nominal attempt scored as the online evaluator does: frame t of the recording against reference frame t,
    frame 0 (the initialization) not checked, stop at the first failing frame (kept in the scored prefix).

    `actual_*` may be shorter than the reference (a recording that already stopped) or run to the clip end.
    """
    n, expected = len(actual_pos), len(reference_pos)
    if n < 1 or n > expected:
        raise ValueError('Recording must be a nonempty prefix of the reference')
    terms = failure_terms(reference_pos[:n], reference_quat[:n], actual_pos, actual_quat, body_names)
    bad = np.zeros(n, dtype=bool)
    for key in CRITERIA[criterion]:
        bad |= terms[key]
    bad[0] = False
    hits = np.flatnonzero(bad)
    first = int(hits[0]) if len(hits) else None
    count = n if first is None else first + 1
    errors = pose_errors(reference_pos[:count], reference_quat[:count], actual_pos[:count], actual_quat[:count],
                         body_names)
    return dict(criterion=criterion, complete=first is None and n == expected, first_failure_frame=first,
                failure_terms=[] if first is None else [k for k in CRITERIA[criterion] if terms[k][first]],
                scored_frames=count, completion_fraction=(count - 1) / (expected - 1),
                metrics_on_observed_prefix={k: dict(mean=float(v.mean()), max=float(v.max())) for k, v in errors.items()})


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
