#!/usr/bin/env python3
"""Hand-picked motion segments (e.g. acrobatics) -> G1 reference library, from LAFAN1 BVH or AMASS-style SMPL-X.

The LAFAN1 library (prepare_lafan1.py) cuts every recording into 6 s windows and rejects windows whose feet go more
than 5 cm into the floor; cartwheels and kip-ups fail that rule because plain GMR puts hands and feet through the
floor. Here each segment is named in a spec file and retargeted on its own:

1. Plain GMR over the segment (LAFAN1: configs/walk1_s1.json; SMPL-X: the video production settings of
   humanoid_motion.video.retarget_config without the floor, at --smplx-fps).
2. Ground offset: median lowest-sole height over human-stance frames (as prepare_lafan1.ground_offset).
3. GMR again with the IK floor constraint at that height on every collision geom of the robot (floor_limit geoms
   'collision'), so hands, knees and back stay on or above the floor too.
4. Root and bodies lowered by the offset, so the floor is z = 0; 50 Hz reference and independent URDF kinematics.
5. Audit: numbers only (lowest body point, lowest sole, joint speed, URDF limits, IK error); no segment is dropped.

Spec: a JSON list of {"id", "source_kind": "bvh_lafan1" | "smplx" | "asf_amc", "source_file", "start_s", "stop_s",
"category", "note"}. LAFAN1 sources may be given as "lafan1:<stem>" (extracted from data/lafan1.zip like
prepare_lafan1.py). CMU ASF/AMC segments also name "asf_file", "source_fps" (from the CMU subject index),
"human_height_m" and optionally "target_fps" (a divisor of source_fps; default source_fps); they use the LAFAN1 GMR
settings, since humanoid_motion.retarget.asf_amc hands GMR LAFAN1-convention frames.
Writes <out>/<id>/{plain, floor, gmr, prepared, segment.json}, then <out>/<name>.json and <out>/frozen.json for
scripts/import_library.py --manifests <name> --data-root <out>. <out> must lie inside the repository.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
from pathlib import Path
import zipfile

import numpy as np

from humanoid_motion import ROOT
from humanoid_motion.contracts import sha256

URDF = ROOT / 'third_party/whole_body_tracking/audited_asset/unitree_description/urdf/g1/main.urdf'   # as prepare_lafan1.py
SCHEMA = 'humanoid_motion.library.prepared.v1'
LAFAN1_FPS = 30
FLOOR = {'kind': 'floor', 'gain': 0.85, 'minimum_distance_m': 0.0, 'detection_distance_m': 0.1, 'geoms': 'collision'}


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        raise ValueError(f'{path} is outside the repository {ROOT}; choose --out inside it') from None


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def lafan1_source(stem: str) -> Path:
    """data/<stem>.bvh, extracted from data/lafan1.zip when missing (the archive is the reference copy)."""
    path = ROOT / 'data' / f'{stem}.bvh'
    with zipfile.ZipFile(ROOT / 'data/lafan1.zip') as archive:
        member = next((m for m in archive.namelist() if Path(m).name == path.name), None)
        if member is None:
            raise FileNotFoundError(f'{path.name} not in data/lafan1.zip')
        raw = archive.read(member)
    if not path.exists():
        path.write_bytes(raw)
    elif path.read_bytes() != raw:
        raise ValueError(f'{path} differs from data/lafan1.zip')
    return path


def base_config(seg: dict, smplx_fps: float) -> dict:
    kind = seg['source_kind']
    if kind == 'bvh_lafan1':
        source = seg['source_file']
        source = lafan1_source(source.split(':', 1)[1]) if source.startswith('lafan1:') else ROOT / source
        cfg = json.loads((ROOT / 'configs/walk1_s1.json').read_text())
        fps = LAFAN1_FPS
        cfg.update(source_file=str(source.resolve().relative_to(ROOT)), first_frame_max_iter=200)
    elif kind == 'asf_amc':
        fps = float(seg.get('target_fps', seg['source_fps']))
        cfg = json.loads((ROOT / 'configs/walk1_s1.json').read_text())
        cfg.update(source_kind='asf_amc', source_file=seg['source_file'], asf_file=seg['asf_file'],
                   source_fps=float(seg['source_fps']), target_fps=fps, human_height_m=float(seg['human_height_m']),
                   first_frame_max_iter=200)
    elif kind == 'smplx':
        fps = smplx_fps
        cfg = {'source_kind': 'smplx', 'source_file': seg['source_file'], 'target_fps': fps, 'robot': 'unitree_g1',
               'solver': 'daqp', 'damping': 0.5, 'use_velocity_limit': False, 'offset_to_ground': False,
               'stance_height_above_p05_m': 0.04, 'stance_speed_m_s': 0.2, 'human_height': {'mode': 'mesh_stature'},
               'first_frame_max_iter': 200}
    else:
        raise ValueError(f'source_kind {kind!r}')
    cfg.update(frame_start=int(round(seg['start_s'] * fps)), frame_end_exclusive=int(round(seg['stop_s'] * fps)),
               selection_reason=f"segment {seg['id']}: {seg.get('note', '')}".strip())
    cfg.pop('output_dir', None)
    return cfg


def ground_offset(reference: Path) -> dict:
    """Median lowest-sole height over human-stance frames (prepare_lafan1.ground_offset), one constant per segment."""
    with np.load(reference) as data:
        soles = data['foot_collision_bottom_z'].min(axis=2)
        stance = data['stance_mask'].astype(bool)
    values = soles[stance]
    floor = float(np.median(values)) if values.size else float(np.percentile(soles.min(axis=1), 5))
    return dict(offset_m=floor, stance_samples=int(values.size), fallback=not values.size)


def lower(run: Path, offset: float, out: Path) -> None:
    """Copy of a GMR run with the floor moved to z = 0 (prepare_lafan1.slice_window over the whole run)."""
    out.mkdir(parents=True, exist_ok=False)
    with np.load(run / 'reference.npz') as data:
        ref = dict(data)
    ref['qpos'][:, 2] -= offset
    ref['body_positions'][..., 2] -= offset
    ref['foot_collision_bottom_z'] -= offset
    if 'body_collision_bottom_z' in ref:
        ref['body_collision_bottom_z'] -= offset
    np.savez_compressed(out / 'reference.npz', **ref)
    csv = np.loadtxt(run / 'motion.csv', delimiter=',')
    csv[:, 2] -= offset   # root_z column
    np.savetxt(out / 'motion.csv', csv, delimiter=',', fmt='%.17g')
    meta = json.loads((run / 'metadata.json').read_text())
    meta['outputs'] = {name: sha256(out / name) for name in ('reference.npz', 'motion.csv')}
    meta['derived_from'] = dict(run=relative(run), ground_offset_m=offset, run_metadata_sha256=sha256(run / 'metadata.json'),
                                rule='root z and body z lowered by ground_offset_m, so the floor is z = 0')
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')


def audit(run: Path, prepared: Path) -> dict:
    from humanoid_motion.urdf_kinematics import UrdfKinematics
    with np.load(run / 'reference.npz') as data:
        ref = dict(data)
    with np.load(prepared / 'generalized_reference.npz') as data:
        gen = dict(data)
    robot = UrdfKinematics(URDF)
    names = gen['joint_names'].tolist()
    urdf_excess = max(float(np.maximum(j['limits'][0] - gen['joint_pos'][:, names.index(j['name'])],
                                       gen['joint_pos'][:, names.index(j['name'])] - j['limits'][1]).max())
                      for j in robot.joints if j['kind'] != 'fixed')
    pelvis = ref['body_positions'][:, list(ref['body_names']).index('pelvis'), 2]
    torso = ref['body_positions'][:, list(ref['body_names']).index('torso_link'), 2]
    return dict(
        lowest_body_m=float(ref['body_collision_bottom_z'].min()) if 'body_collision_bottom_z' in ref else None,
        lowest_sole_m=float(ref['foot_collision_bottom_z'].min()),
        joint_speed_max_rad_s=float(np.abs(gen['joint_vel']).max()),
        urdf_limit_excess_rad=urdf_excess, ik_error_median=float(np.median(ref['ik_error'])),
        ik_error_max=float(np.max(ref['ik_error'])), pelvis_height_range_m=[float(pelvis.min()), float(pelvis.max())],
        torso_below_pelvis_fraction=float((torso < pelvis).mean()), frames_50hz=int(len(gen['timestamps'])))


def build(seg: dict, out: Path, smplx_fps: float) -> dict:
    from humanoid_motion.retarget.gmr import retarget
    from humanoid_motion.retarget.reference import prepare
    d = out / seg['id']
    done = d / 'segment.json'
    if done.exists():
        return json.loads(done.read_text())
    cfg = base_config(seg, smplx_fps)
    for stage, extra in (('plain', None), ('floor', 'floor')):
        run = d / stage
        if (run / 'diagnostics.json').exists():
            continue
        if run.exists() and any(run.iterdir()):
            raise FileExistsError(f'Partial run {run} (no diagnostics.json): remove it first')
        config = dict(cfg)
        if extra:
            config['floor_limit'] = {**FLOOR, 'floor_height_m': ground_offset(d / 'plain/reference.npz')['offset_m']}
        write_json(d / f'{stage}.config.json', config)
        print('STEP retarget', seg['id'], stage, flush=True)
        retarget(config, run)
    ground = ground_offset(d / 'plain/reference.npz')
    if not (d / 'gmr/metadata.json').exists():
        lower(d / 'floor', ground['offset_m'], d / 'gmr')
    if not (d / 'prepared/manifest.json').exists():
        print('STEP prepare', seg['id'], flush=True)
        prepare(d / 'gmr', d / 'prepared', URDF)
    report = json.loads((d / 'prepared/manifest.json').read_text())
    diag = json.loads((d / 'floor/diagnostics.json').read_text())
    row = dict(id=seg['id'], source=seg['source_file'], source_sha256=sha256(ROOT / json.loads((d / 'floor.config.json').read_text())['source_file']),
               source_kind=seg['source_kind'], start_s=seg['start_s'], stop_s=seg['stop_s'], category=seg['category'],
               note=seg.get('note', ''), ground=ground, floor_solver_fallback_frames=len(diag['floor_limit']['solver_fallback_frames']),
               gmr=relative(d / 'gmr'), prepared=relative(d / 'prepared'),
               prepared_manifest_sha256=sha256(d / 'prepared/manifest.json'), frames=report['frames']['output'],
               fps=report['timing']['output_fps'], split='skill', metrics=audit(d / 'gmr', d / 'prepared'),
               rejection_reasons=[], usable=True)
    write_json(done, row)
    print('SEGMENT', seg['id'], json.dumps(row['metrics']), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', required=True, help='Library name: writes <out>/<name>.json and <out>/frozen.json')
    parser.add_argument('--smplx-fps', type=float, default=60., help='GMR target rate for SMPL-X sources')
    parser.add_argument('--only', nargs='*', help='Build only these segment ids (the manifest still lists every built one)')
    parser.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    args = parser.parse_args()
    out = args.out.resolve()
    relative(out)
    segments = json.loads(args.spec.read_text())
    if len({s['id'] for s in segments}) != len(segments):
        raise ValueError('Duplicate segment id')
    todo = [s for s in segments if not args.only or s['id'] in args.only]
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')   # one BLAS thread per worker
    with futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        jobs = {pool.submit(build, s, out, args.smplx_fps): s['id'] for s in todo}
        for job in futures.as_completed(jobs):
            job.result()
    rows = [json.loads((out / s['id'] / 'segment.json').read_text()) for s in segments if (out / s['id'] / 'segment.json').exists()]
    path = out / f'{args.manifest}.json'
    write_json(path, dict(schema=SCHEMA, group=args.manifest, clips=rows,
                          selection='hand-picked segments (spec ' + args.spec.name + '); GMR with the IK floor constraint on every '
                                    'collision geom; every segment listed, metrics only'))
    frozen_path = out / 'frozen.json'
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
    frozen[args.manifest] = sha256(path)
    write_json(frozen_path, frozen)
    print('MANIFEST', path, len(rows), 'segments', flush=True)


if __name__ == '__main__':
    main()
