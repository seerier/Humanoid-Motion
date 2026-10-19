#!/usr/bin/env python3
"""Render saved evaluation rollouts to an MP4 (plus a PNG still, and optionally a GIF or animated WebP) with MuJoCo's
offscreen renderer; nothing is re-simulated.

    # one policy on several clips, three per row, each panel labelled with its motion category
    python scripts/render_video.py grid --library outputs/lafan1/runtime/test.json --evaluation outputs/eval/test \
        --clips dance2_subject4_f2820_3000 sprint1_subject4_f6780_6960 --cols 3 --out outputs/videos/demo.mp4 \
        --webp outputs/videos/demo.webp
    # one row per clip: human source joints | G1 kinematic reference | recorded physical execution (also for failures)
    python scripts/render_video.py pipeline --library ... --evaluation ... --clips dance2_subject4_f2820_3000
    # analysis only, not for demos: the reference as a translucent grey copy over the execution
    python scripts/render_video.py grid ... --clips jumps1_subject1_f2820_3000 --ghost

Look: the G1 in its own two-tone materials on a light checker floor (the squares show travel and speed while the
camera follows the robot), a key light with shadows that follows the camera target, a soft fill light and 8x
multisampling. The default (cinematic) camera keeps a 3/4 front view that turns with the body's smoothed heading,
uses a 28 degree lens framed on the body's size from near eye level, leaves lead room in the direction of travel
and swings +-8 degrees over the clip; output is 50 fps, the rate the rollouts were recorded at. Every frame poses
the robot model from a recorded state. The execution turns red from its first failure; a failed attempt holds its
last recorded frame while the reference plays on.
"""
import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from humanoid_motion import ROOT

FPS = 50                                                                  # rollouts are recorded at 50 Hz
RED = (0.82, 0.24, 0.20, 1.0)
GREY = (0.62, 0.62, 0.62, 0.35)
HUMAN = (0.85, 0.45, 0.15, 1.0)
BG = np.array([247, 247, 245], np.uint8)
CHECKER = ([0.95, 0.95, 0.94], [0.86, 0.86, 0.85])
KEY_DIR = np.array([0.35, 0.45, -1.0])
FILL_DIR = np.array([-0.5, -0.2, -0.6])
GAP = 6
INK = (64, 64, 64)
ALERT = (196, 58, 48)
G1_XML = ROOT / 'third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml'
CATEGORIES = {'aiming': 'Aiming', 'dance': 'Dance', 'fallAndGetUp': 'Fall and get up', 'fight': 'Fight',
              'fightAndSports': 'Fight and sports', 'ground': 'Ground moves', 'jumps': 'Jump',
              'multipleActions': 'Multiple actions', 'run': 'Run', 'sprint': 'Sprint', 'walk': 'Walk'}
PIPELINE_LABELS = ['Human motion capture', 'Retargeted G1 reference', 'Physics-based tracking']
CINE_FOVY = 28.0                                                          # long lens: flatter than MuJoCo's default 45
CINE_ELEVATION = -4.0                                                     # near eye level


def lens(camera: str) -> float:
    return CINE_FOVY if camera == 'cinematic' else 45.


def build_model():
    """G1 scene with the floor and lights replaced: checker floor, directional key light with shadows, fill light."""
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    for geom in [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]:
        spec.delete(geom)
    for light in list(spec.lights):
        spec.delete(light)
    spec.add_texture(name='checker', type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=CHECKER[0], rgb2=CHECKER[1], width=512, height=512)
    floor = spec.add_material(name='checker_floor')
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'checker'
    floor.texuniform, floor.texrepeat, floor.reflectance = True, [1, 1], 0.0
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material='checker_floor', group=2,
                            contype=0, conaffinity=0)
    spec.worldbody.add_light(name='key', castshadow=True, diffuse=[0.55, 0.55, 0.55], specular=[0.12, 0.12, 0.12])
    spec.worldbody.add_light(name='fill', castshadow=False, diffuse=[0.18, 0.18, 0.2], specular=[0, 0, 0])
    model = spec.compile()
    model.light_type[:] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    model.light_dir[0], model.light_dir[1] = KEY_DIR / np.linalg.norm(KEY_DIR), FILL_DIR / np.linalg.norm(FILL_DIR)
    model.vis.quality.offsamples = 8
    model.vis.quality.shadowsize = 8192
    model.vis.map.shadowclip = 4.0                                        # shadow map covers 4 m around the key light
    model.vis.headlight.ambient[:] = 0.26
    model.vis.headlight.diffuse[:] = 0.08
    model.vis.headlight.specular[:] = 0.0
    model.vis.rgba.haze[:] = (*(BG / 255.), 1.0)
    for t in range(model.ntex):                                           # skybox -> plain slide background
        if model.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            a, n = model.tex_adr[t], model.tex_width[t] * model.tex_height[t] * model.tex_nchannel[t]
            model.tex_data[a:a + n] = np.tile(BG, n // 3)
    return model


class Stage:
    """One panel: the G1 model, a free camera set per frame from a View, lights that follow the camera target."""

    def __init__(self, w, h, fovy=45.):
        self.m = model = build_model()
        model.vis.global_.fovy = fovy
        self.floor = {g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE}
        self.r = mujoco.Renderer(model, h, w)
        self.opt = mujoco.MjvOption()                                     # visual meshes (group 1) and the floor (group 2)
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[1] = self.opt.geomgroup[2] = 1
        self.ghost_opt = mujoco.MjvOption()                               # the reference copy brings no second floor
        self.ghost_opt.geomgroup[:] = 0
        self.ghost_opt.geomgroup[1] = 1
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def paint(self, start, color):
        scn = self.r.scene
        for i in range(start, scn.ngeom):
            g = scn.geoms[i]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid not in self.floor:
                g.rgba[:] = color
                if color[3] < 1:
                    g.emission = 0.3

    def shoot(self, data, view, k, color=None, ghost=None, extra=None):
        """Render data from frame k of view; color=None keeps the robot's own materials."""
        look = view.look[k]
        self.cam.lookat[:] = look
        self.cam.azimuth, self.cam.distance, self.cam.elevation = view.azimuth[k], view.distance, view.elevation
        for i in range(self.m.nlight):
            self.m.light_pos[i] = look - 6.0 * self.m.light_dir[i]
        mujoco.mj_camlight(self.m, data)
        self.r.update_scene(data, camera=self.cam, scene_option=self.opt)
        if color is not None:
            self.paint(0, color)
        if ghost is not None:
            start = self.r.scene.ngeom
            mujoco.mjv_addGeoms(self.m, ghost, self.ghost_opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_ALL, self.r.scene)
            self.paint(start, GREY)
        if extra:
            extra(self.r.scene)
        return self.r.render().copy()


def joint_index(model, names, jmap=None):
    jmap = jmap or {}
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qadr = np.array([model.jnt_qposadr[j] for j in hinges])
    src = np.array([names.index(jmap.get(model.joint(j).name, model.joint(j).name)) for j in hinges])
    return qadr, src


def pose(model, data, qadr, src, root_pos, root_quat, joints):
    data.qpos[:3], data.qpos[3:7] = root_pos, root_quat
    data.qpos[qadr] = joints[src]
    mujoco.mj_kinematics(model, data)


def smooth(x: np.ndarray, fps: float, window_s: float) -> np.ndarray:
    """Centred moving average along axis 0, edges padded."""
    k = max(1, int(round(window_s * fps)) | 1)
    padded = np.pad(np.asarray(x, np.float64), [(k // 2, k // 2)] + [(0, 0)] * (np.ndim(x) - 1), mode='edge')
    kernel = np.ones(k) / k
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode='valid'), 0, padded)


def spring(x: np.ndarray, fps: float, hz: float = 0.8) -> np.ndarray:
    """Critically damped spring following x (T, D): it lags like a camera operator and never overshoots."""
    w, h = 2 * np.pi * hz, 1. / (4 * fps)
    pos, vel, out = np.array(x[0], np.float64), np.zeros(x.shape[1]), np.empty(x.shape)
    for i, target in enumerate(x):
        for _ in range(4):                                                # four substeps per frame
            vel += (w * w * (target - pos) - 2 * w * vel) * h
            pos += vel * h
        out[i] = pos
    return out


def follow(points: np.ndarray, fps: float, window_s: float = 0.6, top: float = 0.7, mode: str = 'average') -> np.ndarray:
    """Camera targets from per-frame body positions (T, B, 3): centred moving average (or a lagging spring), height
    kept in [0.3, top] m."""
    centre = spring(points.mean(axis=1), fps) if mode == 'spring' else smooth(points.mean(axis=1), fps, window_s)
    centre[:, 2] = np.clip(centre[:, 2] + 0.05, 0.3, top)                 # lower the target when the robot is on the floor
    return centre


class View:
    """Per-frame camera path: look-at points (T, 3) and azimuths (T,) in degrees; one distance and elevation."""

    def __init__(self, look, azimuth, distance, elevation):
        self.look, self.azimuth = look, np.broadcast_to(np.asarray(azimuth, np.float64), (len(look),))
        self.distance, self.elevation = float(distance), float(elevation)


def framed_distance(points: np.ndarray, fovy: float = CINE_FOVY, fill: float = 0.52) -> float:
    """Distance at which the body fills `fill` of the frame height: its height, or 0.6 of its floor span when lying,
    taken at the clip's 90th percentile so a standing phase is never cut off."""
    height = np.ptp(points[..., 2], axis=1)
    span = np.linalg.norm(np.ptp(points[..., :2], axis=1), axis=1)
    return float(np.percentile(np.maximum(height, 0.6 * span), 90) / fill / (2 * np.tan(np.radians(fovy / 2))))


def turn_with(root_quat: np.ndarray, fps: float, side: float = -35., max_turn: float = 70.) -> np.ndarray:
    """Azimuths for a 3/4 front view of the body: heading smoothed over 1 s, turning at most max_turn deg/s."""
    w, x, y, z = np.asarray(root_quat, np.float64).T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))       # forward (x) axis of the root body
    facing = smooth(np.stack([np.cos(yaw), np.sin(yaw)], axis=1), fps, 1.0)
    target = np.degrees(np.arctan2(facing[:, 1], facing[:, 0])) + 180. + side
    azimuth = np.empty_like(target)
    azimuth[0], step = target[0], max_turn / fps
    for i in range(1, len(target)):
        azimuth[i] = azimuth[i - 1] + np.clip((target[i] - azimuth[i - 1] + 180.) % 360. - 180., -step, step)
    return azimuth


def make_view(camera: str, points: np.ndarray, root_quat: np.ndarray, fps: float, follow_mode: str = 'average',
              orbit: float = 0.) -> View:
    """fixed: world-fixed 3/4 view, 45 degree lens, slightly from above. cinematic: turns with the body, long lens,
    framed on the body's size, near eye level, with lead room in the direction of travel; orbit adds a swing of
    +-orbit degrees (one sine period over the clip)."""
    look = follow(points, fps, mode=follow_mode)
    if camera == 'fixed':
        return View(look, 140., 2.6, -10.)
    velocity = smooth(np.gradient(look[:, :2], axis=0) * fps, fps, 1.0)
    lead = 0.3 * velocity                                                 # aim 0.3 s ahead, at most 0.35 m
    look[:, :2] += lead * np.minimum(1., 0.35 / np.maximum(np.linalg.norm(lead, axis=1, keepdims=True), 1e-9))
    swing = orbit * np.sin(2 * np.pi * np.arange(len(look)) / len(look))
    return View(look, turn_with(root_quat, fps) + swing, framed_distance(points), CINE_ELEVATION)


def caption(frame: np.ndarray, text: str, color=INK) -> np.ndarray:
    im = Image.fromarray(frame)
    h = im.height
    font = ImageFont.load_default(size=max(14, round(h * 0.065)))
    ImageDraw.Draw(im).text((round(h * 0.06), round(h * 0.05)), text, fill=color, font=font)
    return np.asarray(im)


def write(frames, out: Path, still_at: float, gif: Path | None = None, webp: Path | None = None):
    """MP4 (H.264), plus a GIF with a palette per frame region and/or a lossy animated WebP, in one ffmpeg pass."""
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    branches = ['m'] + (['g'] if gif else []) + (['w'] if webp else [])
    graph = f'[0:v]split={len(branches)}' + ''.join(f'[{b}]' for b in branches) if len(branches) > 1 else '[0:v]null[m]'
    if gif:
        graph += (';[g]split[g1][g2];[g1]palettegen=max_colors=256:stats_mode=diff[p];'
                  '[g2][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle[gout]')
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(FPS),
           '-i', '-', '-filter_complex', graph,
           '-map', '[m]', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', str(out)]
    if gif:
        gif.parent.mkdir(parents=True, exist_ok=True)
        cmd += ['-map', '[gout]', '-loop', '0', str(gif)]
    if webp:
        webp.parent.mkdir(parents=True, exist_ok=True)
        cmd += ['-map', '[w]', '-c:v', 'libwebp_anim', '-quality', '82', '-compression_level', '6', '-loop', '0', str(webp)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError('ffmpeg failed')
    still = frames[min(len(frames) - 1, int(still_at * FPS))]
    Image.fromarray(still).save(out.with_suffix('.png'))
    for path in [out, gif, webp]:
        if path:
            print('wrote', path, f'{w}x{h}', len(frames), 'frames, size', path.stat().st_size // 1024, 'KiB')


def grid(rows, gap=GAP):
    """Tile panels with white gutters (gap is even, so even panel sizes stay even for yuv420p)."""
    def join(parts, axis):
        out = [parts[0]]
        for p in parts[1:]:
            shape = list(p.shape)
            shape[axis] = gap
            out += [np.full(shape, 255, np.uint8), p]
        return np.concatenate(out, axis=axis)
    return join([join(list(r), 1) for r in rows], 0)


class TrackedRun:
    def __init__(self, stage, evaluation: Path, reference: dict, manifest: dict, camera: str = 'cinematic',
                 follow_mode: str = 'average', orbit: float = 0., blur: int = 1, shutter: float = 0.5):
        self.st = stage
        self.blur, self.shutter = blur, shutter                           # sub-frames per frame; open fraction of a frame
        z = np.load(evaluation / 'rollout.npz')
        self.d = {k: z[k] for k in z.files}
        ev = json.loads((evaluation / 'evaluation.json').read_text())
        done = json.loads((evaluation / 'completion.json').read_text())
        self.complete = done['unassisted_completions'] == done['attempts'] == 1
        self.fail = None if self.complete else ev.get('first_failure_frame', len(self.d['joint_pos']) - 1)
        self.ref = reference
        self.root = manifest['body_names'].index('pelvis')
        self.qadr, self.src = joint_index(stage.m, manifest['joint_names'])
        self.sim, self.ghost = mujoco.MjData(stage.m), mujoco.MjData(stage.m)
        self.ref_fps = float(reference['fps'].item())
        self.idx = self.d['reference_frame_indices']
        self.view = make_view(camera, self.d['body_pos_w'], self.d['body_quat_w'][:, self.root], self.ref_fps,
                              follow_mode, orbit)                         # a failed run stays put
        self.show_ghost = True
        self.label = None

    def pose_at(self, kf: float):
        """Execution at fractional rollout step kf: positions and joints linear, root rotation by normalised lerp."""
        d, k0 = self.d, int(np.floor(kf))
        k1, a = min(k0 + 1, len(self.idx) - 1), kf - k0
        q0, q1 = d['body_quat_w'][k0, self.root], d['body_quat_w'][k1, self.root]
        q = (1 - a) * q0 + a * (q1 if np.dot(q0, q1) >= 0 else -q1)
        pose(self.st.m, self.sim, self.qadr, self.src, (1 - a) * d['body_pos_w'][k0, self.root] + a * d['body_pos_w'][k1, self.root],
             q / np.linalg.norm(q), (1 - a) * d['joint_pos'][k0] + a * d['joint_pos'][k1])

    def frame(self, t):
        r = min(int(round(t * self.ref_fps)), len(self.ref['joint_pos']) - 1)
        k = min(int(np.searchsorted(self.idx, r)), len(self.idx) - 1)
        d = self.d
        pose(self.st.m, self.ghost, self.qadr, self.src, self.ref['body_pos_w'][r, self.root],
             self.ref['body_quat_w'][r, self.root], self.ref['joint_pos'][r])
        failed = not self.complete and self.fail is not None and k >= min(self.fail, len(self.idx) - 1)
        color, ghost = RED if failed else None, self.ghost if self.show_ghost else None
        if self.blur > 1 and not failed:                                 # average sub-frames over the open shutter
            steps = np.arange(len(self.idx))
            acc = None
            for j in range(self.blur):
                tj = t + ((j + 0.5) / self.blur - 0.5) * self.shutter / FPS
                self.pose_at(float(np.interp(tj * self.ref_fps, self.idx, steps)))
                sub = self.st.shoot(self.sim, self.view, k, color, ghost=ghost).astype(np.float32)
                acc = sub if acc is None else acc + sub
            img = np.round(acc / self.blur).astype(np.uint8)
        else:
            pose(self.st.m, self.sim, self.qadr, self.src, d['body_pos_w'][k, self.root], d['body_quat_w'][k, self.root], d['joint_pos'][k])
            img = self.st.shoot(self.sim, self.view, k, color, ghost=ghost)
        if failed:
            img = caption(img, f'{self.label}: failed' if self.label else 'Failed', ALERT)
        elif self.label:
            img = caption(img, self.label)
        return img


def human_bones(pos, parents, head_index, k):
    """Scene hook drawing the human source joints of frame k as capsules with joint spheres and a head sphere."""
    def add(scn):
        def geom(kind, size, at):
            if scn.ngeom >= scn.maxgeom:
                return None
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, kind, np.asarray(size, np.float64), np.asarray(at, np.float64), np.eye(3).ravel(),
                                np.array(HUMAN, np.float32))
            scn.ngeom += 1
            return g
        for j, par in enumerate(parents):
            if par < 0:
                continue
            g = geom(mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3))
            if g is not None:
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.026, pos[k, par].astype(np.float64), pos[k, j].astype(np.float64))
            geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.032, 0, 0], pos[k, j])
        geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.09, 0, 0], pos[k, head_index])
    return add


def load_clip(library: Path, evaluation: Path, clip: str) -> tuple[dict, dict, dict, Path]:
    rows = {row['id']: row for row in json.loads(library.read_text())['clips']}
    if clip not in rows:
        raise ValueError(f'{clip} is not in {library}')
    motion = ROOT / rows[clip]['motion']
    with np.load(motion) as z:
        reference = {k: z[k] for k in z.files}
    manifest = json.loads(motion.with_suffix('.manifest.json').read_text())
    return rows[clip], reference, manifest, evaluation / clip


def category(clip: str) -> str:
    return CATEGORIES.get(clip.split('_')[0].rstrip('0123456789'), clip)


def render_grid(args) -> None:
    labels = args.labels or [category(c) for c in args.clips]
    if len(labels) != len(args.clips):
        raise ValueError('--labels needs one label per clip')
    runs = []
    for clip, label in zip(args.clips, labels):
        _, reference, manifest, folder = load_clip(args.library, args.evaluation, clip)
        run = TrackedRun(Stage(args.width, args.height, lens(args.camera)), folder, reference, manifest, args.camera,
                         args.follow, args.orbit, args.blur)
        run.show_ghost, run.label = args.ghost, (label or None)
        print(clip, 'complete' if run.complete else f'fails at step {run.fail}')
        runs.append(run)
    rows = [runs[k:k + args.cols] for k in range(0, len(runs), args.cols)]
    while len(rows[-1]) < len(rows[0]):                                  # pad the last row with the background colour
        rows[-1].append(None)
    blank = np.broadcast_to(BG, (args.height, args.width, 3))
    frames = [grid([[run.frame(i / FPS) if run else blank for run in row] for row in rows])
              for i in range(int(args.seconds * FPS))]
    write(frames, args.out, args.still, args.gif, args.webp)


def render_pipeline(args) -> None:
    """Rows = clips; columns = human source joints, G1 kinematic reference, recorded physical execution."""
    rows = []
    for n, clip in enumerate(args.clips):
        row, reference, manifest, folder = load_clip(args.library, args.evaluation, clip)
        source = dict(np.load(ROOT / row['gmr'] / 'reference.npz'))
        hpos, parents = source['source_positions'], source['source_parents'].astype(int)
        hfps = float(1 / np.diff(source['timestamps']).mean())
        head = list(source['source_names']).index('Head')
        hstage = Stage(args.width, args.height, lens(args.camera))
        hdata = mujoco.MjData(hstage.m)
        hdata.qpos[2], hdata.qpos[3] = -10.0, 1.0                         # park the robot model under the floor
        mujoco.mj_kinematics(hstage.m, hdata)
        rstage = Stage(args.width, args.height, lens(args.camera))
        rdata = mujoco.MjData(rstage.m)
        root = manifest['body_names'].index('pelvis')
        qadr, qsrc = joint_index(rstage.m, manifest['joint_names'])
        rfps = float(reference['fps'].item())
        rview = make_view(args.camera, reference['body_pos_w'], reference['body_quat_w'][:, root], rfps, args.follow, args.orbit)
        at_ref = np.minimum(np.round(np.arange(len(hpos)) / hfps * rfps).astype(int), len(rview.look) - 1)
        if args.camera == 'fixed':                                        # the human is taller than the G1
            hview = View(follow(hpos, hfps, top=0.95), 140., 3.3, -10.)
        else:                                                             # same turn as the robot, framed on the human
            hview = View(follow(hpos, hfps, top=0.95, mode=args.follow), rview.azimuth[at_ref], framed_distance(hpos),
                         CINE_ELEVATION)
        run = TrackedRun(Stage(args.width, args.height, lens(args.camera)), folder, reference, manifest, args.camera,
                         args.follow, args.orbit, args.blur)
        run.show_ghost = False
        print(clip, 'complete' if run.complete else f'fails at step {run.fail}')
        top = n == 0                                                      # column titles on the first row only

        def human(t, hstage=hstage, hdata=hdata, hpos=hpos, parents=parents, head=head, hfps=hfps, hview=hview, top=top):
            kf = min(t * hfps, len(hpos) - 1.)                            # mocap is 30 Hz: interpolate between frames
            k0, a = int(kf), kf - int(kf)
            at = (1 - a) * hpos[k0] + a * hpos[min(k0 + 1, len(hpos) - 1)]
            img = hstage.shoot(hdata, hview, min(int(round(kf)), len(hpos) - 1), extra=human_bones(at[None], parents, head, 0))
            return caption(img, PIPELINE_LABELS[0]) if top else img

        def kinematic(t, rstage=rstage, rdata=rdata, ref=reference, root=root, qadr=qadr, qsrc=qsrc, rfps=rfps,
                      rview=rview, top=top):
            r = min(int(round(t * rfps)), len(ref['joint_pos']) - 1)
            pose(rstage.m, rdata, qadr, qsrc, ref['body_pos_w'][r, root], ref['body_quat_w'][r, root], ref['joint_pos'][r])
            img = rstage.shoot(rdata, rview, r)
            return caption(img, PIPELINE_LABELS[1]) if top else img
        run.label = PIPELINE_LABELS[2] if top else None
        rows.append([human, kinematic, run.frame])
    frames = [grid([[f(i / FPS) for f in row] for row in rows]) for i in range(int(args.seconds * FPS))]
    write(frames, args.out, args.still, args.gif, args.webp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('layout', choices=['grid', 'pipeline'])
    parser.add_argument('--library', type=Path, required=True, help='Imported library the policy was evaluated on')
    parser.add_argument('--evaluation', type=Path, required=True, help='evaluate.py output directory')
    parser.add_argument('--clips', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', help='grid layout: one panel label per clip (default: motion category; "" for none)')
    parser.add_argument('--ghost', action='store_true', help='grid layout: also draw the reference as a translucent grey copy (analysis only; leave off for demos)')
    parser.add_argument('--cols', type=int, default=3, help='grid layout: panels per row')
    parser.add_argument('--camera', choices=['cinematic', 'fixed'], default='cinematic',
                        help='cinematic: turns with the body, long lens, framed on its size; fixed: one world view')
    parser.add_argument('--follow', choices=['average', 'spring'], default='average',
                        help='camera target: centred moving average, or a critically damped spring that lags like an operator')
    parser.add_argument('--orbit', type=float, default=8., help='cinematic camera: swing of +-degrees over the clip (0 for none)')
    parser.add_argument('--fps', type=int, default=50, help='output frame rate (rollouts are recorded at 50 Hz)')
    parser.add_argument('--blur', type=int, default=1, help='motion blur: sub-frames averaged over a 180 degree shutter (execution only)')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/videos/demo.mp4')
    parser.add_argument('--gif', type=Path, help='also write a GIF here')
    parser.add_argument('--webp', type=Path, help='also write an animated WebP here')
    parser.add_argument('--width', type=int, default=400)
    parser.add_argument('--height', type=int, default=300)
    parser.add_argument('--seconds', type=float, default=6.0)
    parser.add_argument('--still', type=float, default=2.6, help='time of the PNG still, seconds')
    args = parser.parse_args()
    global FPS
    FPS = args.fps
    (render_grid if args.layout == 'grid' else render_pipeline)(args)


if __name__ == '__main__':
    main()
