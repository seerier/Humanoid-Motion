#!/usr/bin/env python3
"""Render saved evaluation rollouts to an MP4 (plus a PNG still) with MuJoCo's offscreen renderer; nothing is re-simulated.

    # one policy on several clips, two per row
    python scripts/render_video.py grid --library outputs/lafan1/runtime/test.json --evaluation outputs/eval/test \
        --clips sprint1_subject2_f3180_3360 jumps1_subject1_f2820_3000 --cols 2 --out outputs/videos/demo.mp4
    # one row per clip: human source joints | G1 kinematic reference | recorded physical execution
    python scripts/render_video.py pipeline --library ... --evaluation ... --clips dance2_subject4_f2820_3000

Every frame poses the robot model from a recorded state. The execution is blue while tracking and red from its
first failure; a failed attempt holds its last recorded frame while the reference plays on. In grid layout the
reference is a translucent grey copy.
"""
import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image

from humanoid_motion import ROOT

FPS = 25
BLUE = (0.22, 0.42, 0.80, 1.0)
RED = (0.82, 0.24, 0.20, 1.0)
GREY = (0.62, 0.62, 0.62, 0.35)
TEAL = (0.10, 0.50, 0.46, 1.0)
HUMAN = (0.72, 0.38, 0.12, 1.0)
FLOOR = (0.86, 0.86, 0.84, 1.0)
BG = np.array([248, 248, 246], np.uint8)
G1_XML = ROOT / 'third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml'


class Stage:
    """One panel: a model, a free camera following a point, a light floor and a plain background."""

    def __init__(self, model, w, h, distance=2.7, azimuth=135., elevation=-14.):
        self.m = model
        for t in range(model.ntex):                                       # skybox -> plain slide background
            if model.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                a, n = model.tex_adr[t], model.tex_width[t] * model.tex_height[t] * model.tex_nchannel[t]
                model.tex_data[a:a + n] = np.tile(BG, n // 3)
        model.vis.rgba.haze[:] = (*(BG / 255.), 1.0)
        model.vis.headlight.ambient[:] = 0.45
        model.vis.headlight.diffuse[:] = 0.55
        model.light_castshadow[:] = 0                                     # shadows darken the translucent reference
        model.light_type[:] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL     # the scene's spotlight leaves robots far from the origin dark
        for g in range(model.ngeom):
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:            # untextured, lit-from-within floor
                mat = model.geom_matid[g]
                if mat >= 0 and sum(model.geom_matid == mat) == 1:
                    model.mat_texid[mat] = -1
                    model.mat_rgba[mat] = FLOOR
                    model.mat_emission[mat] = 0.3
                    model.mat_reflectance[mat] = 0.0
                else:
                    model.geom_matid[g] = -1
                model.geom_rgba[g] = FLOOR
        self.floor = {g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE}
        meshes = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        visual = [g for g in meshes if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
        self.visual = set(visual or meshes)
        self.r = mujoco.Renderer(model, h, w)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 1
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation

    def paint(self, start, color, hide_floor=False):
        scn = self.r.scene
        for i in range(start, scn.ngeom):
            g = scn.geoms[i]
            if g.objtype != mujoco.mjtObj.mjOBJ_GEOM:
                continue
            if g.objid in self.floor:
                if hide_floor:
                    g.rgba[3] = 0.0
                continue
            if g.objid in self.visual:
                g.rgba[:] = color
                if color[3] < 1:
                    g.emission = 0.3
            else:
                g.rgba[3] = 0.0

    def shoot(self, data, color, look, ghost=None, ghost_color=GREY, extra=None, hide_robot=False):
        self.cam.lookat[:] = look
        self.r.update_scene(data, camera=self.cam, scene_option=self.opt)
        self.paint(0, (1, 1, 1, 0) if hide_robot else color)
        if ghost is not None:
            start = self.r.scene.ngeom
            mujoco.mjv_addGeoms(self.m, ghost, self.opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_ALL, self.r.scene)
            self.paint(start, ghost_color, hide_floor=True)
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


def write(frames, out: Path, still_at: float):
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(FPS),
           '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError('ffmpeg failed')
    still = frames[min(len(frames) - 1, int(still_at * FPS))]
    Image.fromarray(still).save(out.with_suffix('.png'))
    print('wrote', out, f'{w}x{h}', len(frames), 'frames, size', out.stat().st_size // 1024, 'KiB')


def grid(rows):
    return np.concatenate([np.concatenate(r, axis=1) for r in rows], axis=0)


class TrackedRun:
    def __init__(self, stage, evaluation: Path, reference: dict, manifest: dict):
        self.st = stage
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
        self.show_ghost = True

    def frame(self, t):
        r = min(int(round(t * self.ref_fps)), len(self.ref['joint_pos']) - 1)
        k = min(int(np.searchsorted(self.idx, r)), len(self.idx) - 1)
        d = self.d
        pose(self.st.m, self.sim, self.qadr, self.src, d['body_pos_w'][k, self.root], d['body_quat_w'][k, self.root], d['joint_pos'][k])
        pose(self.st.m, self.ghost, self.qadr, self.src, self.ref['body_pos_w'][r, self.root],
             self.ref['body_quat_w'][r, self.root], self.ref['joint_pos'][r])
        failed = not self.complete and self.fail is not None and k >= min(self.fail, len(self.idx) - 1)
        look = d['body_pos_w'][k].mean(axis=0).astype(np.float64)         # follow the body's centre; a failed run stays put
        look[2] = float(np.clip(look[2] + 0.1, 0.3, 0.7))                 # lower the camera target when the robot is on the floor
        return self.st.shoot(self.sim, RED if failed else BLUE, look, ghost=self.ghost if self.show_ghost else None)


def human_bones(pos, parents, head_index, k):
    """Scene hook drawing the human source joints of frame k as capsules plus a head sphere."""
    def add(scn):
        for j, par in enumerate(parents):
            if par < 0 or scn.ngeom >= scn.maxgeom:
                continue
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                np.array(HUMAN, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.028, pos[k, par].astype(np.float64), pos[k, j].astype(np.float64))
            scn.ngeom += 1
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.09, 0, 0]), pos[k, head_index].astype(np.float64),
                            np.eye(3).ravel(), np.array(HUMAN, np.float32))
        scn.ngeom += 1
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


def render_grid(args) -> None:
    runs = []
    for clip in args.clips:
        _, reference, manifest, folder = load_clip(args.library, args.evaluation, clip)
        run = TrackedRun(Stage(mujoco.MjModel.from_xml_path(str(G1_XML)), args.width, args.height), folder, reference, manifest)
        print(clip, 'complete' if run.complete else f'fails at step {run.fail}')
        runs.append(run)
    rows = [runs[k:k + args.cols] for k in range(0, len(runs), args.cols)]
    while len(rows[-1]) < len(rows[0]):                                  # pad the last row with the background colour
        rows[-1].append(None)
    blank = np.broadcast_to(BG, (args.height, args.width, 3))
    frames = [grid([[run.frame(i / FPS) if run else blank for run in row] for row in rows])
              for i in range(int(args.seconds * FPS))]
    write(frames, args.out, args.still)


def render_pipeline(args) -> None:
    """Rows = clips; columns = human source joints, G1 kinematic reference, recorded physical execution."""
    rows = []
    for clip in args.clips:
        row, reference, manifest, folder = load_clip(args.library, args.evaluation, clip)
        source = dict(np.load(ROOT / row['gmr'] / 'reference.npz'))
        hpos, parents = source['source_positions'], source['source_parents'].astype(int)
        hfps = float(1 / np.diff(source['timestamps']).mean())
        head = list(source['source_names']).index('Head')
        hstage = Stage(mujoco.MjModel.from_xml_path(str(G1_XML)), args.width, args.height)
        hdata = mujoco.MjData(hstage.m)
        hdata.qpos[2], hdata.qpos[3] = -10.0, 1.0                         # park the robot model under the floor
        mujoco.mj_kinematics(hstage.m, hdata)
        rstage = Stage(mujoco.MjModel.from_xml_path(str(G1_XML)), args.width, args.height)
        rdata = mujoco.MjData(rstage.m)
        root = manifest['body_names'].index('pelvis')
        qadr, qsrc = joint_index(rstage.m, manifest['joint_names'])
        rfps = float(reference['fps'].item())
        run = TrackedRun(Stage(mujoco.MjModel.from_xml_path(str(G1_XML)), args.width, args.height), folder, reference, manifest)
        run.show_ghost = False
        print(clip, 'complete' if run.complete else f'fails at step {run.fail}')

        def human(t, hstage=hstage, hdata=hdata, hpos=hpos, parents=parents, head=head, hfps=hfps):
            k = min(int(round(t * hfps)), len(hpos) - 1)
            look = hpos[k].mean(axis=0).astype(np.float64)
            look[2] = float(np.clip(look[2] + 0.1, 0.3, 0.7))
            return hstage.shoot(hdata, (1, 1, 1, 0), look, extra=human_bones(hpos, parents, head, k), hide_robot=True)

        def kinematic(t, rstage=rstage, rdata=rdata, ref=reference, root=root, qadr=qadr, qsrc=qsrc, rfps=rfps):
            r = min(int(round(t * rfps)), len(ref['joint_pos']) - 1)
            pose(rstage.m, rdata, qadr, qsrc, ref['body_pos_w'][r, root], ref['body_quat_w'][r, root], ref['joint_pos'][r])
            look = ref['body_pos_w'][r].mean(axis=0).astype(np.float64)
            look[2] = float(np.clip(look[2] + 0.1, 0.3, 0.7))
            return rstage.shoot(rdata, TEAL, look)
        rows.append([human, kinematic, run.frame])
    frames = [grid([[f(i / FPS) for f in row] for row in rows]) for i in range(int(args.seconds * FPS))]
    write(frames, args.out, args.still)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('layout', choices=['grid', 'pipeline'])
    parser.add_argument('--library', type=Path, required=True, help='Imported library the policy was evaluated on')
    parser.add_argument('--evaluation', type=Path, required=True, help='evaluate.py output directory')
    parser.add_argument('--clips', nargs='+', required=True)
    parser.add_argument('--cols', type=int, default=2, help='grid layout: panels per row')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/videos/demo.mp4')
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--seconds', type=float, default=6.0)
    parser.add_argument('--still', type=float, default=2.6, help='time of the PNG still, seconds')
    args = parser.parse_args()
    (render_grid if args.layout == 'grid' else render_pipeline)(args)


if __name__ == '__main__':
    main()
