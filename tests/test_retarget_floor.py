"""Opt-in IK floor constraint in humanoid_motion.retarget.gmr (Unitree G1)."""
import json
import subprocess
import sys

import mink
import mujoco as mj
import numpy as np
import pytest

from humanoid_motion import ROOT
from humanoid_motion.retarget import gmr as R
from general_motion_retargeting import motion_retarget
from general_motion_retargeting.utils.lafan1 import load_bvh_file

BVH = ROOT / 'data/walk1_subject1.bvh'
G1_XML = ROOT / 'third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml'
FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')


def foot_geoms(model):
    return [[g for g in range(model.ngeom) if model.body(int(model.geom_bodyid[g])).name == foot
             and (model.geom_contype[g] or model.geom_conaffinity[g])] for foot in FEET]


class Boom(mink.Limit):
    def compute_qp_inequalities(self, configuration, dt):
        raise RuntimeError('limit reached the QP')


def gmr(**kwargs):
    return R.GeneralMotionRetargeting('bvh_lafan1', 'unitree_g1', actual_human_height=1.75, solver='daqp', damping=.5,
                                      verbose=False, **kwargs)


@pytest.mark.skipif(not BVH.exists(), reason='LAFAN1 BVH not present')
def test_plain_gmr_never_passes_its_ik_limits_and_the_proxy_does(monkeypatch):
    monkeypatch.setattr(motion_retarget, 'mink', mink)   # restored after the test; add_floor_limit swaps it
    frame = load_bvh_file(str(BVH), format='lafan1')[0][300]
    r = gmr()
    r.ik_limits.append(Boom())
    r.retarget(frame)                                     # mink 0.0.13: ik_limits land in safety_break, never in the QP
    limit, _ = R.add_floor_limit(r, foot_geoms(r.model), {'floor_height_m': 0.})
    assert isinstance(motion_retarget.mink, R.MinkLimitsByKeyword) and r.ik_limits[-1] is limit
    with pytest.raises(RuntimeError, match='limit reached the QP'):
        r.retarget(frame)


def test_floor_limit_refuses_the_inert_velocity_limit(monkeypatch):
    monkeypatch.setattr(motion_retarget, 'mink', mink)
    r = gmr(use_velocity_limit=True)
    with pytest.raises(ValueError, match='inert'):
        R.add_floor_limit(r, foot_geoms(r.model), {'floor_height_m': 0.})


def test_lowest_point_matches_the_audit_sole_height():
    rng = np.random.default_rng(0)
    model = mj.MjModel.from_xml_path(str(G1_XML))
    data = mj.MjData(model)
    geoms = [g for ids in foot_geoms(model) for g in ids]
    assert len(geoms) == 8
    for _ in range(20):
        data.qpos[:] = model.qpos0
        data.qpos[3:7] = rng.normal(size=4)
        data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
        lo, hi = model.jnt_range[1:].T
        data.qpos[7:] = rng.uniform(lo, hi)
        mj.mj_forward(model, data)
        for g in geoms:
            assert R.geom_lowest_point(model, data, g)[2] == pytest.approx(R.geom_bottom_z(model, data, g), abs=1e-12)


def test_lowest_point_of_every_collision_geom():
    """floor_limit geoms 'collision': meshes, the sole spheres and the shoulder cylinders; cylinders against sampled rims."""
    rng = np.random.default_rng(1)
    model = mj.MjModel.from_xml_path(str(G1_XML))
    data = mj.MjData(model)
    groups = R.collision_geoms_by_body(model)
    geoms = [g for ids in groups for g in ids]
    assert len(geoms) == len(set(geoms)) == sum(1 for g in range(model.ngeom) if model.geom_bodyid[g]
                                               and (model.geom_contype[g] or model.geom_conaffinity[g]))
    assert any(model.geom_type[g] == mj.mjtGeom.mjGEOM_CYLINDER for g in geoms)
    theta = np.linspace(0, 2 * np.pi, 4000)
    for _ in range(10):
        data.qpos[:] = model.qpos0
        data.qpos[3:7] = rng.normal(size=4)
        data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
        lo, hi = model.jnt_range[1:].T
        data.qpos[7:] = rng.uniform(lo, hi)
        mj.mj_forward(model, data)
        for g in geoms:
            low = R.geom_bottom_z(model, data, g)
            assert R.geom_lowest_point(model, data, g)[2] == pytest.approx(low, abs=1e-12)
            if model.geom_type[g] == mj.mjtGeom.mjGEOM_CYLINDER:
                radius, half = model.geom_size[g, :2]
                rims = [np.stack([radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, s * half)], 1)
                        for s in (-1, 1)]
                world = data.geom_xpos[g] + np.concatenate(rims) @ data.geom_xmat[g].reshape(3, 3).T
                assert low == pytest.approx(world[:, 2].min(), abs=1e-6)


@pytest.mark.skipif(not BVH.exists(), reason='LAFAN1 BVH not present')
def test_floor_out_of_range_is_plain_gmr_and_floor_in_range_stops_penetration(tmp_path):
    """A floor far below changes nothing; a floor 1 cm above plain GMR's lowest sole is held by both kinds."""
    base = json.loads((ROOT / 'configs/walk1_s1.json').read_text())
    base.update(frame_start=300, frame_end_exclusive=325, first_frame_max_iter=200, output_dir='unused')

    def run(name, floor):
        cfg = dict(base, **({'floor_limit': floor} if floor else {}))
        (tmp_path / f'{name}.json').write_text(json.dumps(cfg))
        subprocess.run([sys.executable, str(ROOT / 'scripts/retarget_clip.py'), '--config', str(tmp_path / f'{name}.json'),
                        '--output', str(tmp_path / name)], check=True, capture_output=True)
        return json.loads((tmp_path / name / 'diagnostics.json').read_text())

    off = run('off', None)
    assert 'floor_limit' not in off
    floor = float(np.load(tmp_path / 'off/reference.npz')['foot_collision_bottom_z'].min()) + .01
    mink_kind = {'kind': 'mink_collision'}
    out = {name: run(name, spec) for name, spec in {
        'far': {'floor_height_m': -1.}, 'mink_far': {**mink_kind, 'floor_height_m': -1.},
        'on': {'floor_height_m': floor}, 'mink_on': {**mink_kind, 'floor_height_m': floor},
        'body_on': {'floor_height_m': floor, 'geoms': 'collision'}}.items()}
    for far in ('far', 'mink_far'):
        assert (tmp_path / far / 'motion.csv').read_bytes() == (tmp_path / 'off/motion.csv').read_bytes()
        assert out[far]['floor_limit']['active_rows'] == 0
    for name in ('on', 'mink_on', 'body_on'):
        on = out[name]['floor_limit']
        assert on['frames_below_floor_gt_1mm'] == 0 and on['solver_fallback_frames'] == [] and on['active_rows'] > 0
    assert 'geoms' not in out['on']['floor_limit'] and out['body_on']['floor_limit']['geoms'] == 'collision'
    assert out['body_on']['floor_limit']['lowest_body_minus_floor_m']['min'] >= -1e-3
