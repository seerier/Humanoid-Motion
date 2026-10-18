"""GMR retargeting backend: run unmodified GMR IK headlessly, keep every selected frame and audit FK.

Optional config key floor_limit (absent = unmodified GMR): an IK ground constraint, see add_floor_limit.
"""
import hashlib
import importlib.metadata
import json
import pickle
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

from humanoid_motion import ROOT
sys.path.insert(0, str(ROOT / 'third_party/GMR'))
import mink
import mujoco as mj
import numpy as np
from mink.exceptions import NoSolutionFound
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.params import IK_CONFIG_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
from humanoid_motion.geometry import named_csv, source_fk, xml_forward_kinematics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upstream_commit(name: str) -> str:
    """Git HEAD of a pinned checkout; a partial sync without .git must match configs/upstream_pins.json file by file."""
    repo = ROOT / 'third_party' / name
    if (repo / '.git').exists():
        return subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    pin = json.loads((ROOT / 'configs/upstream_pins.json').read_text())[name]
    for path, digest in pin.get('files', {}).items():
        if sha256(repo / path) != digest:
            raise ValueError(f'{name}/{path} differs from pinned commit {pin["commit"]}')
    return pin['commit']


def geom_bottom_z(model, data, g: int) -> float:
    """Lowest world z of one collision geom (sphere, capsule, box or mesh) at the current pose."""
    kind, size, z = model.geom_type[g], model.geom_size[g], data.geom_xpos[g, 2]
    axis_z = data.geom_xmat[g].reshape(3, 3)[2]      # world-z component of the geom's local axes
    if kind == mj.mjtGeom.mjGEOM_SPHERE:
        return z - size[0]
    if kind == mj.mjtGeom.mjGEOM_CAPSULE:
        return z - size[0] - abs(axis_z[2]) * size[1]
    if kind == mj.mjtGeom.mjGEOM_BOX:
        return z - float(np.abs(axis_z) @ size)
    if kind == mj.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[g]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        return z + float((vertices @ axis_z).min())
    raise ValueError(f'Unsupported foot collision geom type {kind}')


def geom_lowest_point(model, data, g: int) -> np.ndarray:
    """World position of the lowest point of one collision geom (sphere, capsule, box or mesh); z equals geom_bottom_z."""
    kind, size, centre = model.geom_type[g], model.geom_size[g], data.geom_xpos[g]
    rot = data.geom_xmat[g].reshape(3, 3)            # columns: the geom's local axes in world coordinates
    if kind == mj.mjtGeom.mjGEOM_SPHERE:
        return centre - [0., 0., size[0]]
    if kind == mj.mjtGeom.mjGEOM_CAPSULE:
        return centre - np.sign(rot[2, 2]) * size[1] * rot[:, 2] - [0., 0., size[0]]
    if kind == mj.mjtGeom.mjGEOM_BOX:
        return centre + rot @ (-np.sign(rot[2]) * size[:3])
    if kind == mj.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[g]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        return centre + rot @ vertices[int(np.argmin(vertices @ rot[2]))]
    raise ValueError(f'Unsupported foot collision geom type {kind}')


class FloorLimit(mink.Limit):
    """IK ground constraint: the lowest point of each foot collision geom stays at or above a horizontal floor.

    Velocity-damper form on the QP variable, a displacement dq (mink.solve_ik returns v = dq / dt): per IK step a geom's
    lowest point may move down by at most gain * gap, gap = height above floor_height + minimum_distance, so it
    approaches geometrically; a geom already below rises by at least gain * its depth (push-out). Rows exist only for
    geoms within detection_distance of that level, so away from the floor the QP is plain GMR's. The free root can
    always move up, so the rows never make the QP infeasible by themselves.
    """

    def __init__(self, model, geoms: list[int], floor_height: float, gain: float = .85, minimum_distance: float = 0.,
                 detection_distance: float = .1):
        self.model, self.geoms, self.floor_height = model, list(geoms), float(floor_height)
        self.gain, self.minimum_distance, self.detection_distance = float(gain), float(minimum_distance), float(detection_distance)
        self.active_rows = 0   # running count, for diagnostics

    def compute_qp_inequalities(self, configuration, dt):
        model, data = self.model, configuration.data
        rows, bounds, jac = [], [], np.empty((3, model.nv))
        for g in self.geoms:
            point = geom_lowest_point(model, data, g)
            gap = point[2] - self.floor_height - self.minimum_distance
            if gap > self.detection_distance:
                continue
            mj.mj_jac(model, data, jac, None, point, int(model.geom_bodyid[g]))
            rows.append(-jac[2].copy())   # -dz <= gain * gap
            bounds.append(self.gain * gap)
        self.active_rows += len(rows)
        return mink.Constraint(G=np.array(rows), h=np.array(bounds)) if rows else mink.Constraint()


class CountingCollisionAvoidanceLimit(mink.CollisionAvoidanceLimit):
    """mink's CollisionAvoidanceLimit, counting active rows (finite bounds) for diagnostics."""
    active_rows = 0

    def compute_qp_inequalities(self, configuration, dt):
        constraint = super().compute_qp_inequalities(configuration, dt)
        self.active_rows += int(np.isfinite(constraint.h).sum())
        return constraint


class MinkLimitsByKeyword:
    """Stand-in for the mink module inside GMR's motion_retarget, installed only with the floor limit.

    GMR calls mink.solve_ik(configuration, tasks, dt, solver, damping, self.ik_limits). In mink 0.0.13 the sixth
    positional parameter is safety_break, so GMR's ik_limits never reach the QP: limits=None gives mink's default
    [ConfigurationLimit(model)], the same as GMR's own first entry (upstream GMR issue #138, open PRs #177/#186).
    This forwards them as limits= and keeps safety_break=True, which the non-empty list made it.
    """
    def __getattr__(self, name):
        return getattr(mink, name)

    @staticmethod
    def solve_ik(configuration, tasks, dt, solver, damping, safety_break):
        return mink.solve_ik(configuration, tasks, dt, solver, damping, safety_break=True, limits=safety_break)


def add_floor_limit(r: GeneralMotionRetargeting, foot_geoms: list[list[int]], spec: dict):
    """Opt-in IK ground constraint on every foot collision geom, appended to r.ik_limits.

    spec: floor_height_m (required; the floor in the GMR frame, normally the recording's ground offset so the output
    stays in plain GMR's frame), kind, gain, minimum_distance_m, detection_distance_m, and for kind 'mink_collision'
    bound_relaxation and mink_units.
    - kind 'floor' (default): FloorLimit above.
    - kind 'mink_collision': mink.CollisionAvoidanceLimit pairing each foot geom with the model's 'floor' plane, moved
      to floor_height_m. mink 0.0.13 bounds a pair's approach by gain * (d - d_min) / dt although the QP variable is a
      displacement (fixed upstream in mink 1933012, 2026-09-29, after release 1.3.0: bound gain * (d - d_min)).
      mink_units 'per_step' (default) passes gain * dt so the bound is the upstream-fixed one; 'raw' passes gain as is
      (0.0.13 behaviour: about 425x the gap per step at dt = 0.002 s). Below d_min the bound is bound_relaxation, i.e.
      it forbids going deeper but never pushes out.
    GMR passes ik_limits positionally into safety_break (MinkLimitsByKeyword), so this also installs that proxy as the
    mink module of GMR's motion_retarget, process-wide.
    """
    if any(isinstance(limit, mink.VelocityLimit) for limit in r.ik_limits):
        raise ValueError('use_velocity_limit is inert in plain GMR (see MinkLimitsByKeyword); the floor limit would activate it')
    sys.modules[GeneralMotionRetargeting.__module__].mink = MinkLimitsByKeyword()
    model = r.model
    feet = [g for ids in foot_geoms for g in ids]
    kind = spec.get('kind', 'floor')
    gain, minimum = spec.get('gain', .85), spec.get('minimum_distance_m', 0.)
    detection = spec.get('detection_distance_m', .1)
    info = dict(kind=kind, floor_height_m=float(spec['floor_height_m']), gain=gain, minimum_distance_m=minimum,
                detection_distance_m=detection,
                foot_geoms={model.body(int(model.geom_bodyid[ids[0]])).name: [int(g) for g in ids] for ids in foot_geoms})
    if kind == 'floor':
        limit = FloorLimit(model, feet, spec['floor_height_m'], gain=gain, minimum_distance=minimum, detection_distance=detection)
    elif kind == 'mink_collision':
        floor = model.geom('floor').id
        if model.geom_type[floor] != mj.mjtGeom.mjGEOM_PLANE or model.geom_bodyid[floor] != 0 or model.geom_pos[floor, 2] != 0:
            raise ValueError("expected a world-body 'floor' plane at z = 0")
        model.geom_pos[floor, 2] = spec['floor_height_m']
        model.geom_sameframe[floor] = 0   # else mj_kinematics keeps the plane at its body frame (z = 0)
        r.configuration.update()
        units = spec.get('mink_units', 'per_step')
        mink_gain = gain * model.opt.timestep if units == 'per_step' else gain if units == 'raw' else None
        if mink_gain is None:
            raise ValueError(f'mink_units {units!r}')
        limit = CountingCollisionAvoidanceLimit(model, [(feet, [floor])], gain=mink_gain, minimum_distance_from_collisions=minimum,
                                                collision_detection_distance=detection,
                                                bound_relaxation=spec.get('bound_relaxation', 0.))
        if sorted(limit.geom_id_pairs) != sorted((min(g, floor), max(g, floor)) for g in feet):
            raise ValueError('some foot geoms failed mink collision-pair filtering against the floor')
        info.update(mink_units=units, mink_gain_argument=mink_gain, bound_relaxation=limit.bound_relaxation, floor_geom=int(floor))
    else:
        raise ValueError(f'floor_limit kind {kind!r}')
    r.ik_limits.append(limit)
    return limit, info


def summary(values: np.ndarray) -> dict:
    values = np.asarray(values)
    return {'mean': float(values.mean()), 'p95': float(np.percentile(values, 95)), 'max': float(values.max())} if values.size else {'mean': None, 'p95': None, 'max': None}


def retarget(cfg: dict, output: Path | None = None) -> dict:
    """Retarget one BVH window described by cfg; writes reference.npz, motion.pkl, motion.csv, metadata.json, diagnostics.json."""
    if cfg['robot'] != 'unitree_g1':
        raise ValueError('This release supports the Unitree G1 only')
    source = ROOT / cfg['source_file']
    out = Path(output) if output else ROOT / cfg['output_dir']
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite run: {out}; choose --output')
    out.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    frame_time = float(re.search(r'Frame Time:\s*([\d.eE+-]+)', text).group(1))
    fps = 1.0 / frame_time
    frames, upstream_height = load_bvh_file(str(source), format='lafan1')
    anim = read_bvh(str(source))
    start, stop = cfg['frame_start'], cfg['frame_end_exclusive']
    assert 0 <= start < stop <= len(frames) and stop - start >= 3
    frames = frames[start:stop]
    names = list(anim.bones)
    source_positions = np.array([[f[n][0] for n in names] for f in frames])
    source_quaternions = np.array([[f[n][1] for n in names] for f in frames])
    r = GeneralMotionRetargeting('bvh_lafan1', cfg['robot'], actual_human_height=cfg['human_height_m'], solver=cfg['solver'], damping=cfg['damping'], use_velocity_limit=cfg['use_velocity_limit'], verbose=False)
    model = r.model
    joint_ids = [i for i in range(model.njnt) if model.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE]
    joint_names = [model.joint(i).name for i in joint_ids]
    assert [int(model.jnt_qposadr[i]) for i in joint_ids] == list(range(7, model.nq))
    if cfg['robot'] == 'unitree_g1':
        assert model.nq == 36 and model.nv == 35 and len(joint_names) == 29
    body_names = [model.body(i).name for i in range(model.nbody)]
    # Robot body for each human role, read from the GMR IK table, so other humanoids need no hand-written names.
    role = {entry[0]: body for body, entry in r.ik_match_table2.items()}
    feet = [role['LeftFootMod'], role['RightFootMod']]
    target_names = list(r.human_scale_table)
    qposes, targets, positions, rotations, foot_bottoms, ik_errors, rotation_errors = [], [], [], [], [], [], []
    # Optional: let the first (cold-start) frame iterate to convergence; GMR's own cap applies afterwards.
    first_frame_max_iter, default_max_iter = cfg.get('first_frame_max_iter'), r.max_iter
    check_data = mj.MjData(model)
    foot_geoms = [[i for i in range(model.ngeom) if model.body(int(model.geom_bodyid[i])).name == foot
                   and (model.geom_contype[i] or model.geom_conaffinity[i])] for foot in feet]
    assert foot_geoms[0] and len(foot_geoms[0]) == len(foot_geoms[1])
    if cfg['robot'] == 'unitree_g1':   # the four sole spheres on each ankle_roll_link, as before
        assert feet == ['left_ankle_roll_link', 'right_ankle_roll_link'] and all(
            len(ids) == 4 and all(model.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE for g in ids) for ids in foot_geoms)
    # Optional ground constraint {floor_height_m, kind, gain, minimum_distance_m, ...} (add_floor_limit); absent = unmodified GMR.
    floor_limit, floor_info, floor_fallbacks = None, None, []
    if cfg.get('floor_limit'):
        floor_limit, floor_info = add_floor_limit(r, foot_geoms, cfg['floor_limit'])
    tick = time.perf_counter()
    for k, frame in enumerate(frames):
        r.max_iter = first_frame_max_iter if k == 0 and first_frame_max_iter else default_max_iter
        try:
            q = r.retarget(frame, offset_to_ground=cfg['offset_to_ground'])
        except NoSolutionFound:
            if floor_limit is None:
                raise
            # Infeasible QP with the floor: redo this frame without it from where the solve stopped; counted below.
            floor_fallbacks.append(k)
            r.ik_limits = [limit for limit in r.ik_limits if limit is not floor_limit]
            q = r.retarget(frame, offset_to_ground=cfg['offset_to_ground'])
            r.ik_limits.append(floor_limit)
        ik_errors.append(float(r.error2() if r.use_ik_match_table2 else r.error1()))
        rotation_errors.append([np.linalg.norm(task.compute_error(r.configuration)[3:]) for task in r.human_body_to_task2.values()])
        qposes.append(q)
        targets.append([r.scaled_human_data[n][0].copy() for n in target_names])
        check_data.qpos[:] = q
        mj.mj_forward(model, check_data)
        positions.append(check_data.xpos.copy())
        rotations.append(check_data.xquat.copy())
        foot_bottoms.append([[geom_bottom_z(model, check_data, g) for g in ids] for ids in foot_geoms])
    elapsed = time.perf_counter() - tick
    r.max_iter = default_max_iter
    qpos, targets, positions, rotations, foot_bottoms, ik_errors = map(np.array, (qposes, targets, positions, rotations, foot_bottoms, ik_errors))
    assert np.isfinite(qpos).all()
    foot_indices = [names.index(n) for n in ('LeftFoot', 'RightFoot')]
    source_feet = source_positions[:, foot_indices]
    source_speed = np.linalg.norm(np.gradient(source_feet, frame_time, axis=0), axis=-1)
    stance_floor = np.percentile(source_feet[:, :, 2], 5, axis=0)
    stance = (source_feet[:, :, 2] <= stance_floor + cfg['stance_height_above_p05_m']) & (source_speed < cfg['stance_speed_m_s'])
    np.savez_compressed(out / 'reference.npz', qpos=qpos, source_positions=source_positions, source_quaternions=source_quaternions, source_names=np.array(names), source_parents=anim.parents, target_positions=targets, target_names=np.array(target_names), body_positions=positions, body_quaternions=rotations, body_names=np.array(body_names), timestamps=np.arange(len(qpos))*frame_time, stance_mask=stance, foot_collision_bottom_z=foot_bottoms, ik_error=ik_errors)
    with (out / 'motion.pkl').open('wb') as f:
        pickle.dump({'fps': fps, 'root_pos': qpos[:, :3], 'root_rot': qpos[:, [4, 5, 6, 3]], 'dof_pos': qpos[:, 7:], 'local_body_pos': None, 'link_body_list': None}, f)
    # Explicit named contract from pinned csv_to_npz.py, independently of model size.
    if cfg['robot'] == 'unitree_g1':
        expected = [f'{side}_{j}_joint' for side in ('left', 'right') for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        expected += [f'waist_{j}_joint' for j in ('yaw','roll','pitch')]
        expected += [f'{side}_{j}_joint' for side in ('left','right') for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
    else:   # no BeyondMimic CSV contract for other robots yet: the model's own joint order
        expected = list(joint_names)
    csv = named_csv(qpos, joint_names, expected)
    np.savetxt(out / 'motion.csv', csv, delimiter=',', fmt='%.17g')
    roundtrip = np.loadtxt(out / 'motion.csv', delimiter=',')
    roundtrip[:, 3:7] = roundtrip[:, [6,3,4,5]]
    roundtrip[:, 7:] = roundtrip[:, 7:][:, [expected.index(n) for n in joint_names]]
    xml = Path(r.xml_file)
    independent_error = []
    for i, q in enumerate(qpos):
        fk = xml_forward_kinematics(xml, q, joint_names)
        independent_error.extend(np.linalg.norm(fk[n][0]-positions[i,j]) for j,n in enumerate(body_names) if n in fk)
    independent_source = source_fk(anim.pos[start:stop], anim.quats[start:stop], anim.parents)
    axis_matrix = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    independent_source = independent_source @ axis_matrix.T / 100
    residuals = {body: summary(np.linalg.norm(positions[:,body_names.index(body)]-targets[:,target_names.index(entry[0])],axis=1)) for body,entry in r.ik_match_table2.items()}
    robot_feet = positions[:, [body_names.index(foot) for foot in feet]]
    robot_speed_xy = np.linalg.norm(np.gradient(robot_feet[:,:,:2], frame_time, axis=0), axis=-1)
    limits = model.jnt_range[joint_ids]
    violation = np.maximum(limits[:,0]-qpos[:,7:],0)+np.maximum(qpos[:,7:]-limits[:,1],0)
    diagnostics = {
      'evidence_level': 'kinematic_reference_only_no_physics',
      'frames_requested': stop-start, 'frames_saved': len(qpos), 'finite': bool(np.isfinite(qpos).all()),
      'quaternion_norm_max_error': float(np.max(np.abs(np.linalg.norm(qpos[:,3:7],axis=1)-1))),
      'source_fk_independent_max_error_m': float(np.max(np.abs(independent_source-source_positions))),
      'robot_fk_independent_position_error_m': summary(np.array(independent_error)),
      'csv_named_roundtrip_max_error': float(np.max(np.abs(roundtrip-qpos))),
      'joint_limit_violations_gt_1e-6_rad': int((violation>1e-6).sum()), 'joint_limit_max_excess_rad': float(violation.max()),
      'joint_limit_hits_within_1e-3_rad': {name:int((np.minimum(np.abs(qpos[:,7+j]-limits[j,0]),np.abs(qpos[:,7+j]-limits[j,1]))<1e-3).sum()) for j,name in enumerate(joint_names)},
      'joint_frame_delta_rad': summary(np.abs(np.diff(qpos[:,7:],axis=0))),
      'joint_velocity_fd_rad_s': summary(np.abs(np.gradient(qpos[:,7:],frame_time,axis=0))),
      'joint_max_velocity_by_name_rad_s': dict(zip(joint_names, np.max(np.abs(np.gradient(qpos[:,7:],frame_time,axis=0)),axis=0).tolist())),
      'root_path_length_xy_m': float(np.linalg.norm(np.diff(qpos[:,:2],axis=0),axis=1).sum()),
      'root_z_range_m': [float(qpos[:,2].min()),float(qpos[:,2].max())],
      'world_target_body_position_residual_m': residuals,
      'ik_task_rotation_error_deg': {bone: summary(np.degrees(np.array(rotation_errors)[:, i])) for i, bone in enumerate(r.human_body_to_task2)},
      'source_stance_proxy_rule': {'height_baseline': 'per-foot clip p05 ankle height', 'height_m':cfg['stance_height_above_p05_m'],'speed_m_s':cfg['stance_speed_m_s'],'baseline_z_m':stance_floor.tolist(),'not_ground_truth':True},
      'feet': {side: {'stance_frames':int(stance[:,j].sum()), 'total_frames':len(qpos), 'robot_ankle_world_xy_speed_on_source_stance_m_s':summary(robot_speed_xy[stance[:,j],j]),'collision_sphere_min_z_m':float(foot_bottoms[:,j].min()),'frames_any_sphere_below_z0':int((foot_bottoms[:,j].min(axis=1)<0).sum())} for j,side in enumerate(('left','right'))},
      'execution': {'retarget_and_fk_seconds':elapsed,'frames_per_second':len(qpos)/elapsed,'device':'CPU','platform':platform.platform()},
      'not_measured': ['physical completion/failure','torque and actuator saturation','contact force','control feasibility'],
      'ik_task_error_norm': summary(ik_errors),
      'startup': ('cold IK state at first selected frame'
                  + (f'; first frame allowed {first_frame_max_iter} iterations' if first_frame_max_iter else '')
                  + '; no hidden warmup, smoothing or removed frames'),
      'derivatives': 'numpy.gradient at BVH frame_time, central interior / one-sided endpoints; diagnostics only, tracker must resample and recompute'
    }
    if floor_info:
        gap = foot_bottoms.min(axis=(1, 2)) - floor_info['floor_height_m']   # lowest foot geom above the IK floor, per frame
        diagnostics['floor_limit'] = {**floor_info, 'solver_fallback_frames': floor_fallbacks, 'active_rows': floor_limit.active_rows,
                                      'lowest_sole_minus_floor_m': {'min': float(gap.min()), 'p05': float(np.percentile(gap, 5))},
                                      'frames_below_floor_gt_1mm': int((gap < -1e-3).sum()),
                                      'frames_within_5mm_of_floor': int((gap < 5e-3).sum())}
    commits ={name: upstream_commit(name) for name in ('GMR','lafan1','whole_body_tracking')}
    meta = {'config':cfg,'upstream_commits':commits,'source':{'path':cfg['source_file'],'sha256':sha256(source),'frame_start':start,'frame_end_exclusive':stop,'total_frames':len(anim.pos),'fps':fps,'frame_time_s':frame_time,'positions':'BVH cm Y-up → metres Z-up via (x,-z,y); world positions, no trajectory recentering','quaternions':'wxyz','height_assumption_m':cfg['human_height_m'],'loader_hardcoded_height_m':upstream_height},'model':{'xml_path':str(xml.relative_to(ROOT)),'sha256':sha256(xml),'nq':model.nq,'nv':model.nv,'nu':model.nu,'root':f'{model.body(1).name} free joint, xyz metres + wxyz','joint_names':joint_names,'joint_units':'radians','hands':'fixed rubber hands; no finger actuators' if cfg['robot'] == 'unitree_g1' else 'as in the GMR model','feet':feet,'ik_config_path':str(r.xml_file).replace('assets/unitree_g1/g1_mocap_29dof.xml','general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json'),'ik_timestep_s':model.opt.timestep},'csv':{'columns':['root_x','root_y','root_z','qx','qy','qz','qw']+expected,'fps':fps,'resampled':False,'target':'pinned BeyondMimic CSV input, not its final training NPZ'},'outputs':{name:sha256(out/name) for name in ('reference.npz','motion.pkl','motion.csv')}}
    ik_path = Path(IK_CONFIG_DICT['bvh_lafan1'][cfg['robot']])
    meta['model']['ik_config_path'] = str(ik_path.relative_to(ROOT))
    meta['model']['ik_config_sha256'] = sha256(ik_path)
    meta['environment'] = {name: importlib.metadata.version(name) for name in ('numpy','scipy','mujoco','mink','daqp','qpsolvers','viser')}
    meta['environment']['python'] = platform.python_version()
    if floor_info:
        meta['model']['ik_floor'] = {'limit': f"humanoid_motion.retarget.gmr add_floor_limit kind={floor_info['kind']}",
                                     'floor_height_m': floor_info['floor_height_m']}
    meta['source']['selection_reason'] = cfg.get('selection_reason', 'Short ordinary walking / pause / restart segment; retained to inspect support transitions before dynamics. No performance-based frame exclusion.')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    return diagnostics
