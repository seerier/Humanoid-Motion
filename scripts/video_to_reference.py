#!/usr/bin/env python3
"""Monocular video -> GVHMR -> SMPL-X -> GMR -> Unitree G1 reference, in the same reference contract as LAFAN1.

Input: a video file (runs the official GVHMR demo, unmodified, from its own checkout and Python environment; see
scripts/setup_gvhmr.sh and docs/video.md) or an existing GVHMR result `hmr4d_results.pt`. Steps, each skipped when its
output already exists:
  1. video only: 30 fps constant-frame-rate re-encode (ffmpeg) unless the video already is 30 fps; GVHMR's demo labels
     every input 30 fps without resampling. 2. video only: GVHMR demo (static camera unless --moving-camera).
  3. humanoid_motion.video.gvhmr_adapter: SMPL-X npz, z up, grounded (default static_feet), mean betas.
  4. GMR (humanoid_motion.retarget.gmr) with the production config of humanoid_motion.video.retarget_config.
  5. Quality gate (routing only: a failed gate marks the clip as not usable, it never edits or drops it).
  6. humanoid_motion.retarget.reference.prepare: 50 Hz reference and independent URDF forward kinematics.
Layout: <out>/<clip_id>/{source.json, gvhmr/, human/smplx.npz, retarget/config.json, retarget/g1/, gate.json, prepared/,
clip.json}. --manifest NAME then writes <out>/NAME.json listing every clip under <out> that has a clip.json, plus
<out>/frozen.json, so `scripts/import_library.py --manifests NAME --data-root <out>` can import them. <out> must lie
inside the repository: the manifests store repository-relative paths.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from humanoid_motion import ROOT
from humanoid_motion.contracts import sha256

URDF = ROOT / 'third_party/whole_body_tracking/audited_asset/unitree_description/urdf/g1/main.urdf'   # as prepare_lafan1.py
SCHEMA = 'humanoid_motion.library.prepared.v1'
REENCODE = ['-an', '-vf', "fps=30,scale='min(1920,iw)':-2", '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p']


def relative(path: Path) -> str:
    """Repository-relative path, as the library manifests store them (scripts/import_library.py reads ROOT / path)."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        raise ValueError(f'{path} is outside the repository {ROOT}; choose --out inside it') from None


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def pinned_gvhmr_commit() -> str:
    return json.loads((ROOT / 'configs/upstream_pins.json').read_text())['GVHMR']['commit']


def find_ffmpeg(args) -> tuple[str, str]:
    ffmpeg = args.ffmpeg or os.environ.get('FFMPEG') or shutil.which('ffmpeg')
    if not ffmpeg and args.gvhmr_python and (Path(args.gvhmr_python).parent / 'ffmpeg').exists():
        ffmpeg = str(Path(args.gvhmr_python).parent / 'ffmpeg')   # the GVHMR conda environment ships one
    if not ffmpeg:
        raise FileNotFoundError('ffmpeg not found: pass --ffmpeg or set FFMPEG')
    ffprobe = Path(ffmpeg).with_name('ffprobe')
    return ffmpeg, str(ffprobe) if ffprobe.exists() else (shutil.which('ffprobe') or 'ffprobe')


def run_gvhmr(args, clip: Path, clip_id: str) -> tuple[Path, dict]:
    """Steps 1-2: optional 30 fps re-encode, then the official demo. Returns (hmr4d_results.pt, source record)."""
    video = args.video.resolve()
    gvhmr_root = args.gvhmr_root.resolve()
    if not args.gvhmr_python:
        raise ValueError('--video needs the GVHMR environment: pass --gvhmr-python or set GVHMR_PYTHON (docs/video.md)')
    commit = subprocess.check_output(['git', '-C', str(gvhmr_root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != pinned_gvhmr_commit():
        raise ValueError(f'{gvhmr_root} is at {commit}, expected the pinned {pinned_gvhmr_commit()} (scripts/setup_gvhmr.sh)')
    ffmpeg, ffprobe = find_ffmpeg(args)
    probe = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                                'stream=r_frame_rate,avg_frame_rate,width,height,nb_frames', '-of', 'json',
                                                str(video)], text=True))
    stream = probe['streams'][0]
    cut = args.start is not None or args.duration is not None
    is_30 = stream.get('r_frame_rate') == '30/1' and stream.get('avg_frame_rate') == '30/1'
    reencode = args.reencode == 'always' or cut or (args.reencode == 'auto' and not is_30)
    if args.reencode == 'never' and cut:
        raise ValueError('--start / --duration need a re-encode; drop --reencode never')
    if reencode:
        demo_input = clip / f'{clip_id}.mp4'
        if not demo_input.exists():
            trim = (['-ss', str(args.start)] if args.start is not None else []) + \
                   (['-t', str(args.duration)] if args.duration is not None else [])
            tmp = demo_input.with_name(demo_input.stem + '.partial.mp4')
            subprocess.run([ffmpeg, '-y', '-loglevel', 'error', *trim, '-i', str(video), *REENCODE, str(tmp)], check=True)
            tmp.replace(demo_input)
    else:
        demo_input = video
    results = clip / 'gvhmr' / demo_input.stem / 'hmr4d_results.pt'
    if not results.exists():
        cmd = [args.gvhmr_python, 'tools/demo/demo.py', '--video', str(demo_input.resolve()),
               '--output_root', str((clip / 'gvhmr').resolve())]
        cmd += [] if args.moving_camera else ['-s']
        cmd += ['--f_mm', str(args.f_mm)] if args.f_mm else []
        print('GVHMR:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=gvhmr_root, check=True)
    source = dict(input='video', video=str(video), video_name=video.name, video_sha256=sha256(video), probe=stream,
                  cut_start_s=args.start, cut_duration_s=args.duration,
                  reencoded=relative(demo_input) if reencode else None,
                  reencode='ffmpeg ' + ' '.join(REENCODE) if reencode else 'none: input is 30 fps constant frame rate',
                  gvhmr=dict(root=str(gvhmr_root), commit=commit, python=args.gvhmr_python,
                             static_camera=not args.moving_camera, f_mm=args.f_mm))
    return results, source


def run_clip(args) -> Path:
    from humanoid_motion.retarget.gmr import retarget
    from humanoid_motion.retarget.reference import prepare
    from humanoid_motion.video import gate, gvhmr_adapter, retarget_config
    clip_id = args.clip_id or (args.video.stem if args.video else args.gvhmr_results.resolve().parent.name)
    clip = args.out.resolve() / clip_id
    relative(clip)   # fail before any work when <out> is outside the repository
    clip.mkdir(parents=True, exist_ok=True)
    if args.video:
        results, source = run_gvhmr(args, clip, clip_id)
    else:
        results, source = args.gvhmr_results.resolve(), dict(input='gvhmr_results')
    source.update(gvhmr_results=str(results), gvhmr_results_sha256=sha256(results))
    if (clip / 'source.json').exists():
        if json.loads((clip / 'source.json').read_text())['gvhmr_results_sha256'] != source['gvhmr_results_sha256']:
            raise ValueError(f'{clip} was made from another GVHMR result; choose another --clip-id or --out')
    else:
        write_json(clip / 'source.json', source)

    npz = clip / 'human' / 'smplx.npz'
    if not npz.exists():
        print('STEP adapter', clip_id, args.ground, args.betas, flush=True)
        model = ['--smplx_body_model_dir', str(args.smplx_body_model_dir)] if args.smplx_body_model_dir else []
        gvhmr_adapter.main(['--gvhmr_results', str(results), '--out', str(npz), '--ground', args.ground,
                            '--betas', args.betas, *model])
    human = json.loads(npz.with_suffix('.json').read_text())
    if (human['ground']['mode'], human['betas_mode']) != (args.ground, args.betas):
        raise ValueError(f'{npz} was made with ground {human["ground"]["mode"]}, betas {human["betas_mode"]}')

    run_dir = clip / 'retarget' / 'g1'
    if not (run_dir / 'diagnostics.json').exists():   # written last; a run without it was interrupted
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f'Partial retarget run {run_dir} (no diagnostics.json): remove it first')
        print('STEP retarget', clip_id, flush=True)
        reason = f'video clip {clip_id} (production, {args.ground} ground)'
        cfg = retarget_config.build(npz, ROOT, 'production', run_dir, reason)
        write_json(clip / 'retarget' / 'config.json', cfg)
        retarget(cfg, run_dir)

    if not (clip / 'gate.json').exists():
        print('STEP gate', clip_id, flush=True)
        write_json(clip / 'gate.json', gate.evaluate(run_dir, ROOT))
    checks = json.loads((clip / 'gate.json').read_text())
    for name, check in checks['checks'].items():
        print(f"  {name:22s} {check['subject']:6s} {'PASS' if check['pass'] else 'FAIL'}  value={check['value']}", flush=True)

    meta = json.loads((run_dir / 'metadata.json').read_text())
    row = dict(id=clip_id, source=source.get('video_name') or str(results),
               source_sha256=source.get('video_sha256') or source['gvhmr_results_sha256'],
               gvhmr_results_sha256=source['gvhmr_results_sha256'], start=meta['source']['frame_start'],
               stop=meta['source']['frame_end_exclusive'], category=args.category, ground=args.ground, betas=args.betas,
               gmr=relative(run_dir), split=args.split,
               metrics={name: check['value'] for name, check in checks['checks'].items()},
               rejection_reasons=[name for name, check in checks['checks'].items() if not check['pass']],
               usable=bool(checks['passed']), gate=relative(clip / 'gate.json'))
    if not args.no_prepare:
        prepared = clip / 'prepared'
        if not (prepared / 'manifest.json').exists():
            print('STEP prepare', clip_id, flush=True)
            prepare(run_dir, prepared, URDF)
        report = json.loads((prepared / 'manifest.json').read_text())
        row.update(prepared=relative(prepared), prepared_manifest_sha256=sha256(prepared / 'manifest.json'),
                   frames=report['frames']['output'], fps=report['timing']['output_fps'])
    write_json(clip / 'clip.json', row)
    print('CLIP', clip_id, 'usable' if row['usable'] else 'not usable (gate)', clip, flush=True)
    return clip


def write_manifest(out: Path, name: str) -> Path:
    """<out>/<name>.json from every <out>/*/clip.json with a prepared reference, and <out>/frozen.json."""
    rows = []
    for path in sorted(out.glob('*/clip.json')):
        row = json.loads(path.read_text())
        if 'prepared_manifest_sha256' not in row:
            print('manifest: skipping', row['id'], '(not prepared)', flush=True)
            continue
        if sha256(ROOT / row['prepared'] / 'manifest.json') != row['prepared_manifest_sha256']:
            raise ValueError(f'Prepared manifest of {row["id"]} changed since its clip.json was written')
        rows.append(row)
    if not rows:
        raise ValueError(f'No prepared clips under {out}')
    path = out / f'{name}.json'
    write_json(path, dict(schema=SCHEMA, group=name, clips=rows,
                          selection='video clips: GVHMR -> SMPL-X adapter -> GMR production config -> 50 Hz reference; '
                                    'every clip listed, usable = quality gate passed (routing only)'))
    frozen_path = out / 'frozen.json'
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
    frozen[name] = sha256(path)
    write_json(frozen_path, frozen)
    print('MANIFEST', path, len(rows), 'clips,', sum(r['usable'] for r in rows), 'usable', flush=True)
    return path


def main() -> None:
    from humanoid_motion.video.gvhmr_adapter import BETAS_MODES, GROUND_MODES
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--video', type=Path, help='Video file; runs the GVHMR demo')
    source.add_argument('--gvhmr-results', type=Path, help='Existing GVHMR demo result (hmr4d_results.pt)')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/video', help='Library root; one directory per clip')
    parser.add_argument('--clip-id', help='Default: the video stem, or the GVHMR result directory name')
    parser.add_argument('--ground', choices=GROUND_MODES, default='static_feet')
    parser.add_argument('--betas', choices=BETAS_MODES, default='mean')
    parser.add_argument('--category', default='video')
    parser.add_argument('--split', default='video', help="Manifest split label (evaluate.py reports 'held_out_<split>')")
    parser.add_argument('--no-prepare', action='store_true', help='Stop after the gate (no 50 Hz reference)')
    parser.add_argument('--manifest', help='Then write <out>/<name>.json and <out>/frozen.json for import_library.py')
    parser.add_argument('--smplx-body-model-dir', type=Path,
                        help='Default: $HUMANOID_MOTION_SMPLX_BODY_MODELS or data/body_models')
    gvhmr = parser.add_argument_group('GVHMR (with --video)')
    gvhmr.add_argument('--gvhmr-root', type=Path, default=ROOT / 'third_party/GVHMR')
    gvhmr.add_argument('--gvhmr-python', default=os.environ.get('GVHMR_PYTHON'), help='Default: $GVHMR_PYTHON')
    gvhmr.add_argument('--moving-camera', action='store_true', help="Use GVHMR's visual odometry instead of -s")
    gvhmr.add_argument('--f-mm', type=float, help='Focal length in mm (full-frame equivalent), if known')
    gvhmr.add_argument('--reencode', choices=('auto', 'always', 'never'), default='auto',
                       help='30 fps re-encode before GVHMR; auto = unless the video is 30 fps constant frame rate')
    gvhmr.add_argument('--start', type=float, help='Cut start (s); implies a re-encode')
    gvhmr.add_argument('--duration', type=float, help='Cut duration (s); implies a re-encode')
    gvhmr.add_argument('--ffmpeg', help='Default: $FFMPEG, ffmpeg on PATH, or the one next to --gvhmr-python')
    args = parser.parse_args()
    if not (args.video or args.gvhmr_results or args.manifest):
        parser.error('give --video or --gvhmr-results (and/or --manifest)')
    if args.video or args.gvhmr_results:
        run_clip(args)
    if args.manifest:
        write_manifest(args.out.resolve(), args.manifest)


if __name__ == '__main__':
    main()
