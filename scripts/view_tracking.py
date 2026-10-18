#!/usr/bin/env python3
"""Replay saved evaluation rollouts next to their reference motions in the browser (Viser); never steps physics.

    python scripts/view_tracking.py --library outputs/lafan1/runtime/test.json --evaluation ours=outputs/eval/test

Blue is the kinematic reference; orange is the recorded physical execution, drawn from the saved body poses.
Repeat --evaluation LABEL=DIR to compare several policies evaluated on the same library, clip by clip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import viser

from humanoid_motion import ROOT
from humanoid_motion.contracts import sha256
from humanoid_motion.evaluation import pose_errors

BLUE, ORANGE = (62, 155, 231), (243, 151, 57)
BODY_CHOICES = {'Right wrist': 'right_wrist_yaw_link', 'Left wrist': 'left_wrist_yaw_link',
                'Pelvis (root)': 'pelvis', 'Torso': 'torso_link',
                'Left ankle': 'left_ankle_roll_link', 'Right ankle': 'right_ankle_roll_link'}
FAILURE_LABELS = {'anchor_pos': 'torso height', 'anchor_ori': 'torso orientation', 'ee_body_pos': 'ankle or wrist height'}
ALL, FAILED, COMPLETED = 'All clips', 'Failed (this policy)', 'Completed (this policy)'


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def mesh_bindings(model: mujoco.MjModel, names: list[str]) -> list[tuple]:
    """Bind meshes to recorded body frames, allowing only fixed missing children."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bindings = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] != 1:
            continue
        body = int(model.geom_bodyid[geom])
        while mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) not in names:
            if body == 0 or model.body_jntnum[body] != 0:
                raise ValueError('Cannot attach visual mesh across an unrecorded moving joint')
            body = int(model.body_parentid[body])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        rotation = data.xmat[body].reshape(3, 3)
        offset = rotation.T @ (data.geom_xpos[geom] - data.xpos[body])
        local_rotation = rotation.T @ data.geom_xmat[geom].reshape(3, 3)
        bindings.append((geom, names.index(name), offset, local_rotation))
    return bindings


def model_signature(model: mujoco.MjModel) -> str:
    """Fingerprint compiled kinematics used by the mesh-frame audit."""
    digest = hashlib.sha256(bytes(model.names))
    for name in ('body_parentid', 'body_pos', 'body_quat', 'body_jntadr', 'body_jntnum',
                 'jnt_type', 'jnt_bodyid', 'jnt_pos', 'jnt_axis', 'jnt_qposadr', 'qpos0'):
        digest.update(np.asarray(getattr(model, name)).tobytes())
    return digest.hexdigest()


def validate_mesh_frames(model: mujoco.MjModel, reference: dict, manifest: dict) -> dict:
    """Check mesh carrier frames against the independently audited Isaac reference."""
    data = mujoco.MjData(model)
    names = manifest['body_names']
    root = names.index('pelvis')
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names]
    if min(ids) < 0:
        raise ValueError('Visual model lacks a runtime body')
    joints = [(int(model.jnt_qposadr[j]), manifest['joint_names'].index(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)))
        for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    max_position = max_angle = 0.
    for frame in range(len(reference['joint_pos'])):
        data.qpos[:3] = reference['body_pos_w'][frame, root]
        data.qpos[3:7] = reference['body_quat_w'][frame, root]
        for address, index in joints:
            data.qpos[address] = reference['joint_pos'][frame, index]
        mujoco.mj_forward(model, data)
        max_position = max(max_position, float(np.linalg.norm(
            data.xpos[ids] - reference['body_pos_w'][frame], axis=-1).max()))
        rotations = Rotation.from_matrix(data.xmat[ids].reshape(-1, 3, 3))
        recorded = Rotation.from_quat(reference['body_quat_w'][frame][:, [1, 2, 3, 0]])
        max_angle = max(max_angle, float((rotations * recorded.inv()).magnitude().max()))
    if max_position > 1e-4 or max_angle > 1e-4:
        raise ValueError(f'Visual body-frame mismatch: {max_position} m / {max_angle} rad')
    return dict(frames=len(reference['joint_pos']), max_position_m=max_position,
                max_orientation_rad=max_angle,
                rendering='visual meshes attached directly to saved Isaac body poses; no replay integration')


def parse_evaluations(entries: list[str]) -> dict[str, Path]:
    result = {}
    for entry in entries:
        label, separator, folder = entry.partition('=')
        if not separator:
            label, folder = Path(entry).name, entry
        if not label or label in result:
            raise ValueError(f'Empty or duplicate evaluation label: {label!r}')
        result[label] = Path(folder)
    return result


def load_cases(library: Path, evaluations: dict[str, Path]) -> tuple[dict, list[dict]]:
    """cases[(policy, clip_id)] for every clip that all policies evaluated; clips in category order."""
    rows = {row['id']: row for row in json.loads(library.read_text())['clips']}
    references, cases, evaluated = {}, {}, []
    for policy, folder in evaluations.items():
        results = json.loads((folder / 'results.json').read_text())
        if not results.get('perturbation_protocol', 'nominal').startswith('nominal'):
            raise ValueError(f'{policy}: the viewer replays nominal evaluations only')
        ids = set()
        for row in results['rows']:
            clip = row['clip_id']
            if clip not in rows:
                raise ValueError(f'{policy}: clip {clip} is not in {library}')
            motion_path = ROOT / rows[clip]['motion']
            if motion_path not in references:
                manifest = json.loads(motion_path.with_suffix('.manifest.json').read_text())
                if not manifest['audit_passed'] or sha256(motion_path) != manifest['motion_sha256']:
                    raise ValueError(f'Reference provenance mismatch: {motion_path}')
                references[motion_path] = (read_npz(motion_path), manifest)
            reference, manifest = references[motion_path]
            path = folder / clip
            report = json.loads((path / 'evaluation.json').read_text())
            completion = json.loads((path / 'completion.json').read_text())
            states = read_npz(path / 'rollout.npz')
            n = len(states['reference_frame_indices'])
            if report['motion_sha256'] != manifest['motion_sha256'] or report['body_names'] != manifest['body_names']:
                raise ValueError(f'Run/reference mismatch: {path}')
            if not np.array_equal(states['reference_frame_indices'], np.arange(n)) or n > len(reference['joint_pos']):
                raise ValueError('Rollout must be a contiguous recorded prefix')
            if report['last_reference_frame'] != n - 1 or completion['attempt']['reference_frame_indices'] != list(range(n)):
                raise ValueError('Recorded failure boundary mismatch')
            if not all(np.isfinite(a).all() for a in states.values()):
                raise ValueError('Nonfinite recorded state')
            errors = pose_errors(reference['body_pos_w'][:n], reference['body_quat_w'][:n],
                                 states['body_pos_w'], states['body_quat_w'], manifest['body_names'])
            cases[policy, clip] = dict(report=report, completion=completion, states=states, errors=errors, n=n,
                                       reference=reference, manifest=manifest)
            ids.add(clip)
        evaluated.append(ids)
    common = set.intersection(*evaluated)
    if not common:
        raise ValueError('No clip was evaluated by every policy')
    dropped = set.union(*evaluated) - common
    if dropped:
        print(f'Skipping {len(dropped)} clips not evaluated by every policy', flush=True)
    clips = sorted((rows[c] for c in common), key=lambda r: (r['category'], r['id']))
    return {key: case for key, case in cases.items() if key[1] in common}, clips


class RobotMeshes:
    def __init__(self, server: viser.ViserServer, path: str, model: mujoco.MjModel,
                 bindings: list[tuple], color: tuple[int, int, int], **mesh_options) -> None:
        self.group = server.scene.add_frame(path, show_axes=False)
        self.handles = []
        for geom, body, offset, rotation in bindings:
            mesh = model.geom_dataid[geom]
            v, nv = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            f, nf = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            handle = server.scene.add_mesh_simple(f'{path}/mesh_{geom}',
                model.mesh_vert[v:v+nv].copy(), model.mesh_face[f:f+nf].copy(),
                color=color, flat_shading=False, **mesh_options)
            self.handles.append((handle, body, offset, rotation))

    def update(self, positions: np.ndarray, quaternions: np.ndarray, origin: np.ndarray, opacity: float) -> None:
        rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
        for handle, body, offset, rotation in self.handles:
            handle.position = positions[body] - origin + rotations[body] @ offset
            handle.wxyz = Rotation.from_matrix(rotations[body] @ rotation).as_quat()[[3, 0, 1, 2]]
            handle.opacity = opacity


class TrackingViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        evaluations = parse_evaluations(args.evaluation)
        self.cases, clips = load_cases(args.library, evaluations)
        self.policies = list(evaluations)
        self.categories = {row['id']: row['category'] for row in clips}
        self.clip_ids = [row['id'] for row in clips]
        self.labels = {clip: self.clip_label(clip) for clip in self.clip_ids}
        self.by_label = {label: clip for clip, label in self.labels.items()}
        first = self.cases[self.policies[0], self.clip_ids[0]]['manifest']
        self.names = first['body_names']
        self.root = self.names.index('pelvis')
        self.anchor = self.names.index('torso_link')
        self.selection_lock = threading.RLock()
        self.updating_selection = False
        self.end_pause_until = None

        model = mujoco.MjModel.from_xml_path(str(args.model))
        cached = json.loads(args.audit_cache.read_text()) if args.audit_cache and args.audit_cache.exists() else None
        if cached is not None and cached['model_signature'] != model_signature(model):
            raise ValueError('Cached mesh audit uses different model kinematics')
        audits = dict(cached['motions']) if cached else {}
        for case in self.cases.values():
            digest = case['manifest']['motion_sha256']
            if case['manifest']['body_names'] != self.names or case['manifest']['joint_names'] != first['joint_names']:
                raise ValueError('All cases must use the same named robot interface')
            if digest not in audits:
                audits[digest] = validate_mesh_frames(model, case['reference'], case['manifest'])
            receipt = audits[digest]
            if (receipt['frames'] != len(case['reference']['joint_pos'])
                    or receipt['max_position_m'] > 1e-4 or receipt['max_orientation_rad'] > 1e-4):
                raise ValueError('Invalid mesh audit')
        if args.audit_cache:
            args.audit_cache.parent.mkdir(parents=True, exist_ok=True)
            args.audit_cache.write_text(json.dumps(dict(model_signature=model_signature(model), motions=audits), indent=2) + '\n')
        bindings = mesh_bindings(model, self.names)

        self.server = viser.ViserServer(host=args.host, port=args.port)
        gui, scene = self.server.gui, self.server.scene
        gui.configure_theme(control_layout='collapsible', control_width='large',
                            show_share_button=False, brand_color=(46, 135, 179))
        scene.set_up_direction('+z')
        scene.add_grid('/ground', width=8, height=8, plane='xy', cell_size=.25, section_size=1)
        gui.add_markdown('## Humanoid-Motion · tracking replay\n**Blue: kinematic reference · Orange: recorded physics**')
        self.policy = gui.add_dropdown('Policy', tuple(self.policies))
        categories = tuple(sorted(set(self.categories.values())))
        self.filter = gui.add_dropdown('Show', (ALL, FAILED, COMPLETED) + categories)
        self.policy_label, self.clip_id = self.policies[0], self.clip_ids[0]
        self.clip = gui.add_dropdown('Clip', tuple(self.labels[c] for c in self.clip_ids))
        self.clip_transport = gui.add_button_group('Change clip', ('Previous clip', 'Next clip'))
        self.playback_mode = gui.add_dropdown('Playback', ('Play once', 'Loop this clip', 'Play through the list'))
        self.list_status = gui.add_markdown('')
        self.select_reference()
        self.status = gui.add_markdown('')
        self.play = gui.add_checkbox('Play', initial_value=False)
        self.frame = gui.add_slider('Frame', min=0, max=self.current['n']-1, step=1, initial_value=0)
        self.speed = gui.add_slider('Speed', min=.1, max=1., step=.05, initial_value=.5)
        self.transport = gui.add_button_group('Go to', ('Start', 'Previous frame', 'Next frame', 'Last frame'))
        self.scope = gui.add_dropdown('Time range', ('Executed part', 'Full reference'))
        self.layout = gui.add_dropdown('View', ('Side by side', 'Overlay'))
        self.metrics = gui.add_markdown('')
        with gui.add_folder('Inspect', expand_by_default=True):
            self.body = gui.add_dropdown('Body of interest', tuple(BODY_CHOICES))
            self.opacity = gui.add_slider('Reference opacity', min=0., max=1., step=.05, initial_value=.65)
            self.traces = gui.add_checkbox('Root paths', initial_value=True)
            self.align = gui.add_checkbox('Align reference XY / yaw', initial_value=False,
                hint='Display only: hides global drift and heading error; world-error numbers keep their raw values.')
            self.camera = gui.add_button_group('Camera', ('Oblique', 'Front', 'Side', 'Feet'))
        with gui.add_folder('About this data', expand_by_default=False):
            gui.add_markdown('Replays saved Isaac Lab rollouts; nothing is re-simulated and no policy runs here.\n\n'
                'One nominal attempt per policy and clip, starting from the clip\'s first frame. An attempt stops at '
                'its first failure (torso height, torso orientation, or ankle/wrist height off by the BeyondMimic '
                'thresholds); past that point only the reference is shown.\n\n'
                'Side by side only adds a fixed Y offset; all numbers use the same raw world coordinates.')
        self.ref_mesh = RobotMeshes(self.server, '/reference', model, bindings, BLUE)
        self.actual_mesh = RobotMeshes(self.server, '/actual', model, bindings, ORANGE)
        self.ref_label = scene.add_label('/reference/label', 'Reference · kinematic')
        self.actual_label = scene.add_label('/actual/label', 'Executed · recorded physics')
        self.ref_mark = scene.add_point_cloud('/reference/focus', np.zeros((1, 3)), BLUE, point_size=.065)
        self.actual_mark = scene.add_point_cloud('/actual/focus', np.zeros((1, 3)), (255, 70, 80), point_size=.065)
        self.ref_trace = scene.add_line_segments('/reference/path', np.zeros((1, 2, 3)), BLUE,
                                                thickness=2., thickness_units='screen')
        self.actual_trace = scene.add_line_segments('/actual/path', np.zeros((1, 2, 3)), ORANGE,
                                                   thickness=2., thickness_units='screen')
        self.error_line = scene.add_line_segments('/error', np.zeros((1, 2, 3)), (236, 72, 96),
                                                 thickness=2., thickness_units='screen')
        self.camera_view = 'Oblique'
        self.camera.on_click(lambda _: self.set_camera(self.camera.value))
        self.transport.on_click(lambda _: self.navigate())
        self.policy.on_update(lambda _: self.select_list())
        self.filter.on_update(lambda _: self.select_list())
        self.clip.on_update(lambda _: self.select_clip())
        self.clip_transport.on_click(lambda _: self.step_clip(-1 if self.clip_transport.value == 'Previous clip' else 1))
        self.scope.on_update(lambda _: self.update_boundary())
        self.align.on_update(lambda _: self.alignment_changed())
        self.server.on_client_connect(lambda client: self.set_camera(self.camera_view, client))
        self.update_list_status()
        self.render(0)

    def clip_label(self, clip: str) -> str:
        marks = []
        for policy in self.policies:
            case = self.cases[policy, clip]
            mark = '✓' if case['report']['complete'] else f"✗ {(case['n'] - 1) / 50:.1f}s"
            marks.append(mark if len(self.policies) == 1 else f'{policy} {mark}')
        return f"{clip}  {' / '.join(marks)}"

    @property
    def current(self) -> dict:
        return self.cases[self.policy_label, self.clip_id]

    def visible_clips(self) -> list[str]:
        choice = self.filter.value
        if choice == ALL:
            return list(self.clip_ids)
        if choice in (FAILED, COMPLETED):
            want = choice == COMPLETED
            return [c for c in self.clip_ids if self.cases[self.policy.value, c]['report']['complete'] == want]
        return [c for c in self.clip_ids if self.categories[c] == choice]

    def update_list_status(self) -> None:
        shown = self.visible_clips()
        done = sum(self.cases[self.policy.value, c]['report']['complete'] for c in self.clip_ids)
        self.list_status.content = (f'**{self.policy.value}** completes **{done} / {len(self.clip_ids)}** clips. '
                                    f'Showing {len(shown)}.')

    def activate_clip(self, clip: str, *, playing: bool = False) -> None:
        with self.selection_lock:
            self.updating_selection = True
            try:
                self.play.value = False
                self.policy_label, self.clip_id = self.policy.value, clip
                self.clip.value = self.labels[clip]
                self.frame.value = 0
                self.update_boundary()
                self.set_camera(self.camera_view)
                self.end_pause_until = None
                self.play.value = playing
            finally:
                self.updating_selection = False

    def select_list(self) -> None:
        with self.selection_lock:
            if self.updating_selection:
                return
            self.updating_selection = True
            try:
                shown = self.visible_clips() or list(self.clip_ids)
                self.clip.options = tuple(self.labels[c] for c in shown)
                self.activate_clip(self.clip_id if self.clip_id in shown else shown[0])
                self.update_list_status()
            finally:
                self.updating_selection = False

    def select_clip(self) -> None:
        with self.selection_lock:
            if self.updating_selection:
                return
            clip = self.by_label[self.clip.value]
            if clip != self.clip_id:
                self.activate_clip(clip)

    def step_clip(self, direction: int, *, playing: bool = False) -> None:
        shown = self.visible_clips()
        if not shown:
            self.play.value = False
            return
        index = shown.index(self.clip_id) if self.clip_id in shown else (-1 if direction > 0 else 0)
        self.activate_clip(shown[(index + direction) % len(shown)], playing=playing)

    def select_reference(self) -> None:
        self.reference = self.current['reference']
        self.fps = float(self.reference['fps'].item())
        self.frames = len(self.reference['joint_pos'])
        self.origin = self.reference['body_pos_w'][0, self.root].copy()
        self.origin[2] = 0

    def update_boundary(self) -> None:
        self.play.value = False
        self.end_pause_until = None
        self.select_reference()
        limit = self.frames-1 if self.scope.value == 'Full reference' else self.current['n']-1
        self.frame.value = min(int(self.frame.value), limit)
        self.frame.max = limit

    def navigate(self) -> None:
        frame = {'Start': 0, 'Previous frame': max(0, int(self.frame.value)-1),
                 'Next frame': min(int(self.frame.max), int(self.frame.value)+1),
                 'Last frame': self.current['n']-1}[self.transport.value]
        self.play.value = False
        self.frame.value = frame

    def alignment_changed(self) -> None:
        if self.align.value:
            self.layout.value = 'Overlay'

    def set_camera(self, view: str, client: viser.ClientHandle | None = None) -> None:
        self.camera_view = view
        path = self.reference['body_pos_w'][:, self.root, :2]-self.origin[:2]
        center = np.r_[(path.min(axis=0)+path.max(axis=0))/2, .72]
        distance_scale = max(1., (float(np.ptp(path, axis=0).max())+1.6)/3.)
        offsets = {'Oblique': (2.9, -3.2, 1.9), 'Front': (0., -4.6, .55),
                   'Side': (4.6, 0., .55), 'Feet': (2.4, -2.8, .9)}
        if view == 'Feet':
            center[2] = .1
        clients = [client] if client else self.server.get_clients().values()
        for connection in clients:
            with connection.atomic():
                connection.camera.fov = np.deg2rad(45.)
                connection.camera.position = center + np.asarray(offsets[view])*distance_scale
                connection.camera.look_at = center

    def render(self, frame: int) -> None:
        case = self.current
        states, report = case['states'], case['report']
        actual_available = frame < case['n']
        position = self.reference['body_pos_w'][frame].copy()
        quaternion = self.reference['body_quat_w'][frame].copy()
        aligned = self.align.value and actual_available
        if aligned:
            actual_anchor = Rotation.from_quat(states['body_quat_w'][frame, self.anchor, [1, 2, 3, 0]])
            ref_rotations = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])
            difference = (actual_anchor * ref_rotations[self.anchor].inv()).as_matrix()
            yaw = Rotation.from_euler('z', np.arctan2(difference[1, 0], difference[0, 0]))
            shift = states['body_pos_w'][frame, self.anchor].copy()
            shift[2] = position[self.anchor, 2]
            position = shift + yaw.apply(position-position[self.anchor])
            quaternion = (yaw * ref_rotations).as_quat()[:, [3, 0, 1, 2]]
        body = self.names.index(BODY_CHOICES[self.body.value])
        with self.server.atomic():
            offset = .8 if self.layout.value == 'Side by side' else 0.
            self.ref_mesh.group.position = (0., offset, 0.)
            self.actual_mesh.group.position = (0., -offset, 0.)
            self.actual_mesh.group.visible = actual_available
            self.ref_mesh.update(position, quaternion, self.origin, self.opacity.value)
            self.ref_label.position = position[self.anchor] - self.origin + [0, 0, .63]
            self.ref_label.text = 'Reference · aligned XY/yaw' if aligned else 'Reference · kinematic'
            self.ref_mark.points = (position[body:body+1]-self.origin).astype(np.float32)
            ref_path = self.reference['body_pos_w'][:, self.root]-self.origin
            actual_path = states['body_pos_w'][:, self.root]-self.origin
            self.ref_trace.points = np.stack((ref_path[:-1], ref_path[1:]), axis=1)
            self.actual_trace.points = np.stack((actual_path[:-1], actual_path[1:]), axis=1)
            self.ref_trace.visible = self.actual_trace.visible = self.traces.value and not aligned
            self.error_line.visible = actual_available and self.layout.value == 'Overlay'
            if actual_available:
                actual_pos = states['body_pos_w'][frame]
                self.actual_mesh.update(actual_pos, states['body_quat_w'][frame], self.origin, 1.)
                self.actual_label.position = actual_pos[self.anchor]-self.origin+[0, 0, .63]
                self.actual_mark.points = (actual_pos[body:body+1]-self.origin).astype(np.float32)
                self.error_line.points = np.array([[position[body]-self.origin, actual_pos[body]-self.origin]])
                error = case['errors']
                z = float(actual_pos[body, 2]-self.reference['body_pos_w'][frame, body, 2])
                self.metrics.content = (f"**World root position error: {100*error['world_root_position_m'][frame]:.1f} cm**\n\n"
                    f"Aligned mean body error: {100*error['aligned_body_position_m'][frame].mean():.1f} cm\n\n"
                    f"{self.body.value} height, executed minus reference: **{100*z:+.1f} cm**\n\n"
                    f"Frame {frame} / {case['n']-1} · {frame/self.fps:.2f} s"
                    + ('\n\nThe reference is displayed aligned; world errors are still raw.' if aligned else ''))
            else:
                self.metrics.content = (f'**{frame/self.fps:.2f} s: reference only, no physics data.**\n\n'
                                        f'The execution stopped at {(case["n"]-1)/self.fps:.2f} s; the orange robot is hidden.')
            terminal = frame == case['n']-1
            flag = ' **← failure frame**' if terminal and report['terminal_reason'] == 'terminated' else ''
            ending = 'clip completed' if report['terminal_reason'] == 'clip_complete' else ' / '.join(
                FAILURE_LABELS.get(term, term) for term in report['failure_terms'])
            split = 'held-out clip' if report.get('evaluation_split', '').startswith('held_out') else 'training clip'
            self.status.content = (f"**{self.policy_label}** · {self.categories[self.clip_id]} · {split}\n\n"
                f"Executed **{(case['n']-1)/self.fps:.2f} / {(self.frames-1)/self.fps:.2f} s**{flag}\n\n"
                f"End: {ending} · resets after start {report['state_writes_after_initialization']}"
                f" · assistance {report['assistance_count']}")

    def run(self) -> None:
        last_state = None
        was_playing = False
        next_frame = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                with self.selection_lock:
                    if self.play.value:
                        if not was_playing:
                            if self.frame.value >= self.frame.max:
                                self.frame.value = 0
                            self.end_pause_until = None
                            next_frame = now + 1/self.fps/self.speed.value
                        if now >= next_frame:
                            if self.frame.value < self.frame.max:
                                self.end_pause_until = None
                                self.frame.value = int(self.frame.value)+1
                            elif self.playback_mode.value == 'Play once':
                                self.play.value = False
                            elif self.end_pause_until is None:
                                self.end_pause_until = now + .8
                            elif now >= self.end_pause_until:
                                if self.playback_mode.value == 'Loop this clip':
                                    self.frame.value = 0
                                    self.end_pause_until = None
                                else:
                                    self.step_clip(1, playing=True)
                            next_frame = now + 1/self.fps/self.speed.value
                    else:
                        self.end_pause_until = None
                    was_playing = self.play.value
                    state = (self.policy_label, self.clip_id, self.frame.value, self.layout.value, self.opacity.value,
                             self.traces.value, self.align.value, self.body.value, self.scope.value)
                    if state != last_state:
                        self.render(int(self.frame.value))
                        last_state = state
                # Idle: GUI changes are picked up within 50 ms. Playing: wake for the next frame only.
                time.sleep(min(max(next_frame - time.monotonic(), .002), .05) if self.play.value else .05)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--library', type=Path, required=True, help='Imported library the policies were evaluated on')
    parser.add_argument('--evaluation', action='append', required=True, metavar='LABEL=DIR',
                        help='evaluate.py output directory; repeat to compare policies')
    parser.add_argument('--model', type=Path, default=ROOT/'third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml')
    parser.add_argument('--audit-cache', type=Path, help='Read and update a mesh-frame audit cache (JSON)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8081)
    TrackingViewer(parser.parse_args()).run()


if __name__ == '__main__':
    main()
