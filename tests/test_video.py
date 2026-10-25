"""Video front end: the GVHMR adapter (humanoid_motion.video.gvhmr_adapter) and the SMPL-X path into GMR.

Synthetic GVHMR-style parameters -> adapter -> npz -> sources.load_amass_smplx. Checks: the entry's own axis audit says
z up, frame count and rate survive, the parameter-level rotation equals the joint-level one to numerical precision, and
ground alignment puts the lowest foot vertices at z = 0. Tests that need the licensed SMPL-X neutral model
(data/body_models/smplx or HUMANOID_MOTION_SMPLX_BODY_MODELS) or smplx / mink are skipped where those are absent.
"""
import importlib.util
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from humanoid_motion import ROOT
from humanoid_motion.retarget.sources import load_amass_smplx, smplx_model_dir
from humanoid_motion.video import gvhmr_adapter as ad
from humanoid_motion.video import retarget_config

MODEL_DIR = smplx_model_dir(None, ROOT)
HAVE_MODEL = (MODEL_DIR / 'smplx' / 'SMPLX_NEUTRAL.npz').is_file() and importlib.util.find_spec('smplx') is not None
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason='SMPL-X neutral body model or smplx not available')
HAVE_GMR = (ROOT / 'third_party/GMR').is_dir() and importlib.util.find_spec('mink') is not None


def synthetic_params(frames: int = 12, seed: int = 0) -> dict:
    """A GVHMR-shaped clip in GVHMR's y-up frame: upright body (identity global orient = SMPL-X rest, y up), walking
    along +x with a little bob, small random joint motion, per-frame betas with jitter."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames) / ad.GVHMR_FPS
    transl = np.stack([0.8 * t, 0.1 * t, 0.02 * np.sin(2 * np.pi * t)], axis=1)   # y is up here (rises), z is depth (bobs)
    body_pose = 0.15 * rng.standard_normal((frames, 63)) * np.linspace(0, 1, frames)[:, None]
    global_orient = np.stack([0.05 * np.sin(t), 0.3 * t, 0.05 * np.cos(t)], axis=1)   # slow turn about the up axis
    betas = np.array([0.5, -0.3, 0.2, 0, 0, 0.1, 0, 0, 0, 0]) + 0.01 * rng.standard_normal((frames, 10))
    return {'body_pose': body_pose, 'betas': betas, 'global_orient': global_orient, 'transl': transl}


def test_rotation_matrix_is_gmrs():
    r = ad.Y_UP_TO_Z_UP
    assert np.allclose(r @ r.T, np.eye(3)) and np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(r @ [0, 1, 0], [0, 0, 1])          # y up -> z up
    assert np.allclose(r @ [0, 0, 1], [0, -1, 0])         # depth -> -y, right-handed


def test_rotate_params_translation_term():
    """Without the (R - I) J0 term the rotated body would sit in the wrong place whenever J0 != 0."""
    pelvis = np.array([0.0, -0.35, 0.01])
    orient = np.zeros((1, 3))
    trans = np.array([[1.0, 2.0, 3.0]])
    new_orient, new_trans = ad.rotate_params(orient, trans, pelvis, ad.Y_UP_TO_Z_UP)
    assert np.allclose(new_trans[0], ad.Y_UP_TO_Z_UP @ trans[0] + (ad.Y_UP_TO_Z_UP - np.eye(3)) @ pelvis)
    assert np.allclose(Rotation.from_rotvec(new_orient[0]).as_matrix(), ad.Y_UP_TO_Z_UP)


@needs_model
def test_adapter_npz_passes_the_entry_and_its_axis_audit(tmp_path):
    params = synthetic_params()
    arrays, report = ad.convert(params, MODEL_DIR, ground='feet_p05', betas_mode='mean')
    assert report['rotation_equivalence_max_error_m'] < 1e-5
    out = tmp_path / 'clip.npz'
    ad.write(arrays, report, out)
    with np.load(out, allow_pickle=False) as f:
        assert set(('gender', 'betas', 'root_orient', 'pose_body', 'trans', 'mocap_frame_rate')) <= set(f.files)
        assert f['betas'].shape == (16,) and np.all(f['betas'][10:] == 0)
        assert str(f['gender']) == 'neutral' and float(f['mocap_frame_rate']) == 30.0
    src = load_amass_smplx(out, 0, len(params['body_pose']), target_fps=30.0, height={'mode': 'mesh_stature'}, root=ROOT)
    assert src.fps == 30.0 and len(src.frames) == len(params['body_pose']) == src.total_frames
    assert src.audit['verdict'] == 'z_up_confirmed', src.audit
    assert src.audit['units_plausible_metres']
    assert 1.4 < src.human_height_m < 2.0
    # The entry's joints are the adapter's joints: same body model, same parameters.
    pelvis = src.positions[:, src.names.index('pelvis')]
    assert abs(float(pelvis[:, 2].min()) - report['pelvis_height_m']['min']) < 1e-4
    # Walking direction +x in GVHMR's frame stays +x; the rise along GVHMR's y becomes a rise along z; GVHMR's depth
    # z becomes -y (the bob is 0 at the first and last of these 12 frames, so y barely moves).
    assert pelvis[-1, 0] - pelvis[0, 0] > 0.2 and pelvis[-1, 2] - pelvis[0, 2] > 0.03 and abs(pelvis[-1, 1] - pelvis[0, 1]) < 0.02


@needs_model
def test_ground_alignment_puts_feet_on_the_floor():
    params = synthetic_params()
    params['transl'][:, 1] += 0.37                      # the whole clip floats 37 cm above GVHMR's ground
    arrays, report = ad.convert(params, MODEL_DIR, ground='feet_p05')
    lowest = report['ground']['lowest_foot_vertex_after_m']
    assert abs(lowest['p05']) < 1e-3 and lowest['min'] < 0.01 and lowest['max'] < 0.2
    assert report['ground']['frames_below_minus_2_5cm'] == 0
    arrays_none, report_none = ad.convert(params, MODEL_DIR, ground='none')
    assert np.allclose(arrays['trans'][:, :2], arrays_none['trans'][:, :2])
    assert np.allclose(arrays_none['trans'][:, 2] - arrays['trans'][:, 2], report['ground']['offset_removed_m'])


def test_static_feet_offset_planted_frames_and_interpolation():
    """Pure offset rule: planted frames take the planted foot's height (lower foot if both), gaps interpolate linearly,
    the ends hold, no planted frame at all gives None."""
    frames, fps = 30, 30.0
    left = 0.10 + 0.002 * np.arange(frames)            # left foot drifts up 2 mm per frame
    right = np.full(frames, 0.50)                        # right foot high in the air except where marked planted
    planted_left = np.ones(frames, bool)
    planted_left[10:20] = False
    planted_right = np.zeros(frames, bool)
    planted_right[25] = True
    right[25] = 0.05                                     # both planted at 25: lower wins
    offset, info = ad.static_feet_offset(left, right, planted_left, planted_right, fps, cutoff_hz=None)
    assert info['planted_frames'] == 20 and info['planted_left_frames'] == 20 and info['planted_right_frames'] == 1
    assert np.allclose(offset[:10], left[:10]) and np.allclose(offset[20:25], left[20:25]) and np.allclose(offset[26:], left[26:])
    assert np.isclose(offset[25], 0.05)
    assert np.allclose(offset[10:20], np.linspace(left[9], left[20], 12)[1:-1])
    none, info_none = ad.static_feet_offset(left, right, np.zeros(frames, bool), np.zeros(frames, bool), fps, cutoff_hz=None)
    assert none is None and info_none['planted_frames'] == 0
    # low-pass keeps a constant offset exactly and leaves a slow ramp nearly untouched
    smooth, info_s = ad.static_feet_offset(left, right, np.ones(frames, bool), np.zeros(frames, bool), fps)
    assert np.abs(smooth - left).max() < 0.01 and info_s['cutoff_hz'] == ad.STATIC_FEET_CUTOFF_HZ


@needs_model
def test_static_feet_ground_puts_planted_feet_on_the_floor_every_frame():
    params = synthetic_params()
    t = np.arange(params['transl'].shape[0]) / ad.GVHMR_FPS
    params['transl'][:, 1] += 0.37 + 0.15 * t          # floats 37 cm and keeps drifting up (time-varying, like a GVHMR squat)
    static = np.full((len(t), len(ad.STATIC_JOINTS)), -5.0)
    static[:, :4] = 5.0                                  # both feet reported planted in every frame, wrists not
    arrays, report = ad.convert(params, MODEL_DIR, ground='static_feet', static_conf=static)
    g = report['ground']
    assert g['static_feet_applied'] and g['planted_frames'] == len(t)
    lowest = g['lowest_foot_vertex_after_m']
    assert abs(lowest['min']) < 1e-3 and abs(lowest['max']) < 1e-3          # the lower foot is on the floor in every frame
    arrays_none, _ = ad.convert(params, MODEL_DIR, ground='none')
    assert np.allclose(arrays_none['trans'][:, 2] - arrays['trans'][:, 2], g['offset_per_frame_m'], atol=1e-4)
    assert g['offset_per_frame_m'][-1] - g['offset_per_frame_m'][0] > 0.04   # the drift was removed, not a constant
    with pytest.raises(ValueError):                      # static_feet without static_conf must be rejected
        ad.convert(params, MODEL_DIR, ground='static_feet')


@needs_model
def test_betas_modes():
    params = synthetic_params()
    _, first = ad.convert(params, MODEL_DIR, ground='none', betas_mode='first')
    _, mean = ad.convert(params, MODEL_DIR, ground='none', betas_mode='mean')
    assert np.allclose(first['betas10'], params['betas'][0], atol=1e-6)
    assert np.allclose(mean['betas10'], params['betas'].mean(axis=0), atol=1e-6)


def test_retarget_config_modes(tmp_path):
    npz = tmp_path / 'clip.npz'
    np.savez(npz, pose_body=np.zeros((7, 63)))
    production = retarget_config.build(npz, tmp_path, 'production', tmp_path / 'run', 'test')
    assert production['source_kind'] == 'smplx' and production['frame_end_exclusive'] == 7
    assert production['human_height'] == {'mode': 'mesh_stature'} and production['first_frame_max_iter'] == 200
    assert production['floor_limit'] == retarget_config.FLOOR_0MM and production['floor_limit'] is not retarget_config.FLOOR_0MM
    reconcile = retarget_config.build(npz, tmp_path, 'reconcile', tmp_path / 'run', 'test')
    assert reconcile['human_height'] == {'mode': 'gmr_beta0'} and 'floor_limit' not in reconcile
    with pytest.raises(ValueError):
        retarget_config.build(npz, tmp_path, 'other', tmp_path / 'run', 'test')


def test_smplx_model_dir_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv('HUMANOID_MOTION_SMPLX_BODY_MODELS', raising=False)
    assert smplx_model_dir(None, tmp_path) == tmp_path / 'data/body_models'
    assert smplx_model_dir('models', tmp_path) == tmp_path / 'models'
    monkeypatch.setenv('HUMANOID_MOTION_SMPLX_BODY_MODELS', str(tmp_path / 'elsewhere'))
    assert smplx_model_dir(None, tmp_path) == tmp_path / 'elsewhere'


def test_fk_check_reads_the_import_sidecar(tmp_path):
    from humanoid_motion.video.fk_check import runtime_names
    npz = tmp_path / 'motion.npz'
    (tmp_path / 'motion.manifest.json').write_text(json.dumps(dict(joint_names=['a'], body_names=['pelvis'], fps=50)))
    assert runtime_names(npz) == {'joint_names': ['a'], 'body_names': ['pelvis']}
    (tmp_path / 'motion.npz.names.json').write_text(json.dumps(dict(joint_names=['b'], body_names=['pelvis'])))
    assert runtime_names(npz)['joint_names'] == ['b']


@needs_model
@pytest.mark.skipif(not HAVE_GMR, reason='GMR checkout or mink not available')
def test_smplx_clip_retargets_and_gates(tmp_path):
    """The SMPL-X path through unmodified GMR with the production config: G1 run directory, gate, source audit."""
    from humanoid_motion.retarget.gmr import retarget
    from humanoid_motion.video import gate
    params = synthetic_params()
    arrays, report = ad.convert(params, MODEL_DIR, ground='feet_p05')
    npz = tmp_path / 'human' / 'smplx.npz'
    ad.write(arrays, report, npz)
    cfg = retarget_config.build(npz, tmp_path, 'production', tmp_path / 'run', 'test')
    cfg['source_file'] = str(npz)          # absolute: retarget() reads ROOT / source_file
    diagnostics = retarget(cfg, tmp_path / 'run')
    assert diagnostics['frames_saved'] == len(params['body_pose']) and diagnostics['finite']
    assert diagnostics['source_axis_audit']['verdict'] == 'z_up_confirmed'
    assert diagnostics['robot_fk_independent_position_error_m']['max'] < 1e-6
    meta = json.loads((tmp_path / 'run' / 'metadata.json').read_text())
    assert meta['source']['kind'] == 'smplx' and meta['model']['feet'] == ['left_ankle_roll_link', 'right_ankle_roll_link']
    assert meta['model']['ik_config_path'].endswith('smplx_to_g1.json')
    result = gate.evaluate(tmp_path / 'run', ROOT)
    assert set(result['checks']) == {'g1_joint_limits', 'g1_foot_penetration', 'human_pelvis_height', 'human_root_jerk'}
    assert abs(result['fps'] - 30.0) < 1e-6 and result['frames'] == len(params['body_pose'])
