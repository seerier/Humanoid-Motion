#!/usr/bin/env python3
"""Import prepared reference windows into Isaac (BeyondMimic motion format) through one application, content-addressed.

Needs an NVIDIA GPU with Isaac Sim 4.5 / Isaac Lab 2.1. Writes <runtime-root>/<split>.json libraries for training
and evaluation; each clip passes the kinematic audit in humanoid_motion.tracking.isaac_export.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from humanoid_motion import ROOT
from humanoid_motion.library import atomic_json
from humanoid_motion.contracts import sha256, TRACKING_COMMIT, URDF_SHA256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifests', nargs='+', required=True, help='Split names to import, e.g. train validation test')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'outputs/lafan1',
                        help='Holds <split>.json and frozen.json from prepare_lafan1.py')
    parser.add_argument('--runtime-root', type=Path, help='Default: <data-root>/runtime')
    parser.add_argument('--deadline', type=float, default=float('inf'), help='Epoch seconds; stop before this')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.urdf = ROOT / 'third_party/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf'
    args.tracking_root = ROOT / 'third_party/whole_body_tracking'
    if sha256(args.urdf) != URDF_SHA256 or subprocess.check_output(['git','-C',str(args.tracking_root),'rev-parse','HEAD'],text=True).strip() != TRACKING_COMMIT:
        raise ValueError('Unpinned asset or source')
    if subprocess.check_output(['git','-C',str(args.tracking_root),'diff','HEAD','--','source'],text=True):
        raise ValueError('Modified upstream source')
    data_root = args.data_root.resolve()
    runtime_root = (args.runtime_root or data_root / 'runtime').resolve()
    groups = args.manifests
    tag = '-'.join(groups)
    cache = runtime_root / 'cache-v2'
    cache.mkdir(parents=True, exist_ok=True)
    args.usd_dir = runtime_root / f'assets-v2-{tag}'
    launcher = None
    runtime = None
    start = time.monotonic()
    try:
        for group in groups:
            manifest_path = data_root / f'{group}.json'
            frozen = json.loads((data_root / 'frozen.json').read_text())
            if sha256(manifest_path) != frozen[group]:
                raise ValueError('Frozen data manifest mismatch')
            source = json.loads(manifest_path.read_text())
            exported = []
            for row in source['clips']:
                if time.time() >= args.deadline:
                    raise TimeoutError('Import deadline exceeded')
                args.prepared = ROOT / row['prepared']
                if sha256(args.prepared / 'manifest.json') != row['prepared_manifest_sha256']:
                    raise ValueError('Prepared manifest changed')
                prepared = json.loads((args.prepared / 'manifest.json').read_text())
                for name, digest in prepared['files'].items():
                    if sha256(args.prepared / name) != digest:
                        raise ValueError('Prepared file changed')
                key = row['prepared_manifest_sha256']
                destination = cache / key
                with (cache / f'{key}.lock').open('a') as lock:
                    while True:
                        try:
                            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.time() >= args.deadline:
                                raise TimeoutError('Cache lock deadline')
                            time.sleep(.5)
                    if not (destination / 'motion.manifest.json').exists():
                        if launcher is None:
                            launcher = AppLauncher(args)
                        from humanoid_motion.tracking.isaac_export import export
                        args.output = cache / f'{key}.building-{os.getpid()}'
                        args.output.mkdir(parents=True, exist_ok=False)
                        with np.load(args.prepared / 'generalized_reference.npz') as z:
                            generalized = dict(z)
                        with np.load(args.prepared / 'independent_urdf.npz') as z:
                            independent = dict(z)
                        resources = {}
                        runtime = export(args, generalized, independent, start, resources, runtime=runtime)
                        atomic_json(args.output / 'resources.json', resources)
                        os.replace(args.output, destination)
                    sidecar = json.loads((destination / 'motion.manifest.json').read_text())
                    if not sidecar['audit_passed'] or sha256(destination / 'motion.npz') != sidecar['motion_sha256']:
                        raise ValueError('Cache hash or audit failure')
                    exported.append({**row, 'motion': str((destination/'motion.npz').relative_to(ROOT)),
                        'motion_sha256': sidecar['motion_sha256'],
                        'runtime_manifest_sha256': sha256(destination/'motion.manifest.json')})
                atomic_json(runtime_root / f'import-{tag}-status.json', dict(group=group, done=len(exported), total=len(source['clips']), elapsed_s=time.monotonic()-start))
            result = {**source, 'schema': 'humanoid_motion.library.isaac.v1', 'clips': exported,
                      'source_manifest_sha256': frozen[group]}
            with (runtime_root/f'{group}.lock').open('a') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                dest = runtime_root / f'{group}.json'
                if dest.exists() and json.loads(dest.read_text()) != result:
                    raise ValueError('Concurrent library publication mismatch')
                atomic_json(dest, result)
        atomic_json(runtime_root / f'import-{tag}-complete.json', dict(complete=True,source_hashes=frozen))
    except BaseException as error:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        atomic_json(runtime_root / ('error-' + Path(__file__).stem + '-' + str(os.getpid()) + '.json'), dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        if launcher is not None:
            from isaaclab.sim import SimulationContext
            context = SimulationContext.instance()
            if context is not None:
                context._disable_app_control_on_stop_handle = True
            launcher.app.close(wait_for_replicator=False)
    print('BATCH_IMPORT_COMPLETE', tag, flush=True)

if __name__ == '__main__':
    main()
