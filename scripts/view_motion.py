#!/usr/bin/env python3
"""Inspect source motion and G1 kinematic reference in Viser (no physics steps)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import viser


from humanoid_motion import ROOT
SOURCE_COLOR = (54, 176, 245)
TARGET_COLOR = (240, 156, 47)
LEFT_COLOR = (67, 204, 143)
RIGHT_COLOR = (206, 107, 246)


def path_segments(points: np.ndarray) -> np.ndarray:
    return np.stack((points[:-1], points[1:]), axis=1).astype(np.float32)


def named_index(names: list[str], candidates: tuple[str, ...]) -> int | None:
    return next((names.index(name) for name in candidates if name in names), None)


class MotionViewer:
    def __init__(self, run: Path, host: str, port: int) -> None:
        self.run = run.resolve()
        self.metadata = json.loads((self.run / "metadata.json").read_text())
        with np.load(self.run / "reference.npz", allow_pickle=False) as archive:
            self.arrays = {key: archive[key] for key in archive.files}
        a = self.arrays
        self.qpos = a["qpos"]
        self.source = a["source_positions"]
        self.names = a["source_names"].tolist()
        self.parents = a["source_parents"].astype(int)
        self.timestamps = a["timestamps"]
        self.frames = len(self.qpos)
        if self.frames < 2 or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Viewer requires at least two strictly increasing timestamps")
        if self.source.shape != (self.frames, len(self.names), 3):
            raise ValueError("source_positions shape does not match source_names / qpos")
        if len(self.parents) != len(self.names) or np.any(self.parents >= len(self.names)):
            raise ValueError("Invalid source parent indices")
        xml = Path(self.metadata["model"]["xml_path"])
        self.model = mujoco.MjModel.from_xml_path(str(xml if xml.is_absolute() else ROOT / xml))
        self.data = mujoco.MjData(self.model)
        if self.qpos.shape != (self.frames, self.model.nq):
            raise ValueError("Reference qpos does not match the pinned model")
        self.edges = np.array([(parent, child) for child, parent in enumerate(self.parents)
                               if parent >= 0], dtype=int)
        self.root_index = named_index(self.names, ("Hips", "hips", "pelvis"))
        if self.root_index is None:
            self.root_index = int(np.flatnonzero(self.parents < 0)[0])
        # One constant translation, shared by every frame and both streams.
        self.origin = self.qpos[0, :3].copy()
        self.origin[2] = 0.0
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("+z")
        self.server.gui.configure_theme(control_layout="collapsible", control_width="medium",
                                        show_share_button=False, brand_color=(46, 135, 179))
        self.server.gui.add_markdown(
            "# Humanoid-Motion · retarget inspection\n"
            "**Blue:** source BVH skeleton, metres / Z up.\n\n"
            "**G1 mesh:** kinematic reference, **not physics**.\n\n"
            "**Orange points:** scaled and offset GMR targets. "
            "Green / purple foot markers indicate a source-derived stance proxy, "
            "not measured contact or forces."
        )
        self.play = self.server.gui.add_checkbox("Play", initial_value=False)
        self.frame = self.server.gui.add_slider("Frame", min=0, max=self.frames - 1,
                                                 step=1, initial_value=0)
        self.speed = self.server.gui.add_slider("Speed", min=0.1, max=2.0,
                                                 step=0.1, initial_value=1.0)
        self.layout = self.server.gui.add_dropdown("View", ("Side by side", "Overlay"))
        self.robot_opacity = self.server.gui.add_slider(
            "Robot opacity", min=0.0, max=1.0, step=0.05, initial_value=1.0,
            hint="1 = solid, 0 = hidden. Lower to see GMR targets inside the robot.")
        self.traces = self.server.gui.add_checkbox("Root / foot trajectories", initial_value=True)
        self.show_targets = self.server.gui.add_checkbox("GMR target points", initial_value=False)
        self.contact = self.server.gui.add_checkbox("Source stance proxy", initial_value=True)
        self.camera_center = (self.source[0, self.root_index] + self.qpos[0, :3]) / 2 - self.origin
        self.camera_center[2] = 0.8
        self.camera_view = "Oblique"
        for name in ("Oblique", "Front", "Side", "Feet"):
            button = self.server.gui.add_button(name)
            button.on_click(lambda _event, view=name: self.set_camera(view))
        self.status = self.server.gui.add_markdown("")
        self.server.gui.add_markdown(
            "World frame: metres, Z up, wxyz quaternions. Only a **constant shared XY "
            "translation** is applied. Side-by-side view adds ±1.1 m on Y; overlay removes "
            "only that display offset. No per-frame root centering.\n\n"
            "Camera presets use a fixed scene center: Front looks from world +X; "
            "Side from world +Y (not the moving robot heading). Feet looks toward "
            "the ground near the initial roots.\n\n"
            "Playback loops only replay the view; they are not simulation resets "
            "or evidence of successful tracking.\n\n"
            f"Run: `{self.run.name}`"
        )
        self.server.scene.add_grid("/ground", width=20, height=20, plane="xy",
                                   cell_size=0.25, section_size=1.0)
        self.source_group = self.server.scene.add_frame("/source", show_axes=False)
        self.robot_group = self.server.scene.add_frame("/robot", show_axes=False)
        self.source_lines = self.server.scene.add_line_segments(
            "/source/skeleton", self.source[0, self.edges] - self.origin,
            SOURCE_COLOR, thickness=3.0, thickness_units="screen")
        self.source_joints = self.server.scene.add_point_cloud(
            "/source/joints", self.source[0] - self.origin, SOURCE_COLOR,
            point_size=0.025, point_shape="circle", precision="float32")
        self.source_label = self.server.scene.add_label(
            "/source/label", "SOURCE · BVH", position=(0, 0, 2))
        self.robot_label = self.server.scene.add_label(
            "/robot/label", "G1 · KINEMATIC / NOT PHYSICS", position=(0, 0, 1.5))
        self.mesh_handles: list[tuple[int, viser.MeshHandle]] = []
        visual_ids = [g for g in range(self.model.ngeom)
                      if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                      and self.model.geom_group[g] == 1]
        if not visual_ids:
            raise ValueError("This viewer expects the audited G1 group-1 visual mesh geoms")
        for geom_id in visual_ids:
            mesh_id = self.model.geom_dataid[geom_id]
            v_start, v_count = self.model.mesh_vertadr[mesh_id], self.model.mesh_vertnum[mesh_id]
            f_start, f_count = self.model.mesh_faceadr[mesh_id], self.model.mesh_facenum[mesh_id]
            rgba = self.model.geom_rgba[geom_id]
            handle = self.server.scene.add_mesh_simple(
                f"/robot/mesh_{geom_id}",
                self.model.mesh_vert[v_start:v_start + v_count].copy(),
                self.model.mesh_face[f_start:f_start + f_count].copy(),
                color=tuple(int(channel * 255) for channel in rgba[:3]),
                opacity=float(rgba[3]), flat_shading=False)
            self.mesh_handles.append((geom_id, handle))
        self.targets = None
        if "target_positions" in a:
            self.targets = self.server.scene.add_point_cloud(
                "/robot/targets", a["target_positions"][0] - self.origin,
                TARGET_COLOR, point_size=0.045, point_shape="circle", visible=False)
        self.trajectory_handles = []
        self.add_trajectory("/source/root_path", self.source[:, self.root_index], SOURCE_COLOR)
        self.add_trajectory("/robot/root_path", self.qpos[:, :3], TARGET_COLOR)
        self.foot_indices: list[int | None] = []
        self.foot_markers = []
        body_names = a.get("body_names", np.array([], dtype=str)).tolist()
        for side, color in (("left", LEFT_COLOR), ("right", RIGHT_COLOR)):
            foot = named_index(self.names, (f"{side.title()}Foot", f"{side}_foot"))
            self.foot_indices.append(foot)
            if foot is not None:
                self.add_trajectory(f"/source/{side}_foot_path", self.source[:, foot], color)
                marker = self.server.scene.add_point_cloud(
                    f"/source/{side}_stance_proxy", self.source[0, foot:foot + 1] - self.origin,
                    color, point_size=0.10, point_shape="circle", visible=False)
            else:
                marker = None
            self.foot_markers.append(marker)
            body = named_index(body_names, (f"{side}_ankle_roll_link", f"{side}_foot"))
            if body is not None:
                self.add_trajectory(f"/robot/{side}_foot_path", a["body_positions"][:, body], color)

        @self.server.on_client_connect
        def initialize_camera(client: viser.ClientHandle) -> None:
            self.set_camera(self.camera_view, client)

        self.render(0)

    def add_trajectory(self, name: str, points: np.ndarray, color: tuple[int, int, int]) -> None:
        self.trajectory_handles.append(self.server.scene.add_line_segments(
            name, path_segments(points - self.origin), color,
            thickness=1.5, thickness_units="screen"))

    def set_camera(self, view: str, client: viser.ClientHandle | None = None) -> None:
        self.camera_view = view
        center = self.camera_center.copy()
        offsets = {
            "Oblique": (2.5, -3.0, 1.6),
            "Front": (4.2, 0.0, 0.35),
            "Side": (0.0, 4.2, 0.35),
            "Feet": (2.0, -2.4, 0.9),
        }
        if view == "Feet":
            center[2] = 0.15
        clients = [client] if client is not None else self.server.get_clients().values()
        for connected in clients:
            with connected.atomic():
                connected.camera.position = center + np.array(offsets[view])
                # Position also translates look_at; apply the final target last.
                connected.camera.look_at = center

    def render(self, frame: int) -> None:
        a = self.arrays
        self.data.qpos[:] = self.qpos[frame]
        mujoco.mj_forward(self.model, self.data)
        with self.server.atomic():
            offset = 1.1 if self.layout.value == "Side by side" else 0.0
            self.source_group.position = (0.0, offset, 0.0)
            self.robot_group.position = (0.0, -offset, 0.0)
            self.source_lines.points = (self.source[frame, self.edges] - self.origin).astype(np.float32)
            self.source_joints.points = (self.source[frame] - self.origin).astype(np.float32)
            for geom_id, handle in self.mesh_handles:
                handle.opacity = float(self.model.geom_rgba[geom_id, 3]) * self.robot_opacity.value
                handle.visible = self.robot_opacity.value > 0.0
                handle.position = self.data.geom_xpos[geom_id] - self.origin
                xyzw = Rotation.from_matrix(self.data.geom_xmat[geom_id].reshape(3, 3)).as_quat()
                handle.wxyz = xyzw[[3, 0, 1, 2]]
            self.source_label.position = self.source[frame, self.root_index] - self.origin + (0, 0, 1.0)
            self.robot_label.position = self.qpos[frame, :3] - self.origin + (0, 0, 0.85)
            if self.targets is not None:
                self.targets.points = (a["target_positions"][frame] - self.origin).astype(np.float32)
                self.targets.visible = self.show_targets.value
            for handle in self.trajectory_handles:
                handle.visible = self.traces.value
            stance = a.get("stance_mask")
            for side, (foot, marker) in enumerate(zip(self.foot_indices, self.foot_markers)):
                if marker is not None and foot is not None:
                    marker.points = (self.source[frame, foot:foot + 1] - self.origin).astype(np.float32)
                    marker.visible = bool(self.contact.value and stance is not None and stance[frame, side])
            source_frame = self.metadata["source"]["frame_start"] + frame
            contacts = "unavailable" if stance is None else ", ".join(
                f"{side}: {'stance' if active else 'swing'}"
                for side, active in zip(("L", "R"), stance[frame]))
            self.status.content = (
                f"**Time:** {self.timestamps[frame]:.3f} s · **source frame:** {source_frame}\n\n"
                f"**Reference frame:** {frame} / {self.frames - 1}\n\n"
                f"**Source stance proxy:** {contacts}\n\n"
                "No controller, contact forces, or simulated execution are shown."
            )

    def run_forever(self) -> None:
        last_state = None
        next_frame_time = time.monotonic()
        was_playing = False
        try:
            while True:
                now = time.monotonic()
                if self.play.value:
                    if not was_playing:
                        next_frame_time = now
                    if now >= next_frame_time:
                        frame = int(self.frame.value)
                        duration = float(self.timestamps[min(frame + 1, self.frames - 1)]
                                         - self.timestamps[max(0, frame if frame < self.frames - 1 else frame - 1)])
                        self.frame.value = (frame + 1) % self.frames
                        next_frame_time = now + duration / self.speed.value
                was_playing = self.play.value
                state = (self.frame.value, self.layout.value, self.traces.value,
                         self.show_targets.value, self.contact.value, self.robot_opacity.value)
                if state != last_state:
                    self.render(int(self.frame.value))
                    last_state = state
                time.sleep(0.004)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/walk1_s1_f300_480", help="Run directory with reference.npz and metadata.json (retarget_clip.py output)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    MotionViewer(args.run, args.host, args.port).run_forever()


if __name__ == "__main__":
    main()
