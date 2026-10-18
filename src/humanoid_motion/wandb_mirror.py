"""Optional WandB mirror of a training run's TensorBoard scalars.

Uploads only those scalars, the given config and system stats: no motions,
checkpoints, code, git diff or console. Motions never travel through WandB.
Never fatal: an online failure falls back to offline, then to no mirror.
HUMANOID_MOTION_WANDB (online | offline | disabled, default disabled) controls our runs;
WANDB_MODE keeps third-party code disabled.
"""
import json
import os
from pathlib import Path

RECORD = 'wandb_run.json'
MODES = ('online', 'offline', 'disabled')
SCOPE = 'TensorBoard scalars, config and system stats only; no motions, checkpoints, code, git diff or console'


def _write(path: Path, record: dict) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    os.replace(tmp, path)


def _teardown(wandb) -> None:
    for step in (wandb.finish, wandb.tensorboard.unpatch):
        try:
            step()
        except Exception:
            pass


def start(output: Path, job_type: str, config: dict):
    """Call before the runner creates its SummaryWriter. Returns the run, or None."""
    output = Path(output)
    path = output / RECORD
    requested = os.environ.get('HUMANOID_MOTION_WANDB', 'disabled')
    if requested not in MODES:
        raise ValueError(f'HUMANOID_MOTION_WANDB={requested!r}; expected one of {MODES}')
    previous = json.loads(path.read_text()) if path.exists() else {}
    record = dict(requested_mode=requested, scope=SCOPE, errors=[])
    if requested == 'disabled':
        _write(path, dict(record, mode='disabled'))
        return None
    try:
        import wandb
    except ImportError as error:
        _write(path, dict(record, mode='unavailable', errors=[repr(error)]))
        return None
    # A requeued online run resumes its stored id. WandB ignores resume offline, so offline
    # segments get fresh ids (same name/group) instead of colliding when synced later.
    fresh = wandb.util.generate_id()
    earlier = previous.get('earlier_ids', []) + ([previous['id']] if previous.get('id') else [])
    # Outside the TensorBoard logdir: otherwise the watcher re-ingests WandB's own copies of the event files.
    wandb_dir = Path(os.environ.get('WANDB_DIR') or output.parent).resolve()
    if wandb_dir == output.resolve() or output.resolve() in wandb_dir.parents:
        raise ValueError(f'WANDB_DIR {wandb_dir} must lie outside the TensorBoard logdir {output}')
    wandb_dir.mkdir(parents=True, exist_ok=True)
    for mode in [requested] if requested == 'offline' else [requested, 'offline']:
        run_id = previous['id'] if mode == 'online' and previous.get('mode') == 'online' else fresh
        saved = os.environ.get('WANDB_MODE')
        os.environ['WANDB_MODE'] = mode  # agree with mode= regardless of WandB's precedence rules
        try:
            run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'Humanoid-Motion'), id=run_id, resume='allow',
                             name=output.name, group=output.parent.name, job_type=job_type, dir=str(wandb_dir),
                             config=config, mode=mode, sync_tensorboard=True, save_code=False,
                             settings=wandb.Settings(disable_git=True, console='off', init_timeout=60))
        except Exception as error:
            record['errors'].append(f'{mode}: {error!r}')
            _teardown(wandb)
            continue
        finally:
            if saved is None:
                os.environ.pop('WANDB_MODE', None)
            else:
                os.environ['WANDB_MODE'] = saved
        record.update(mode=mode, id=run_id, earlier_ids=[i for i in earlier if i != run_id], project=getattr(run, 'project', None),
                      url=getattr(run, 'url', None) if mode == 'online' else None)
        if mode == 'offline':
            record['sync'] = f'wandb sync {wandb_dir}/wandb/offline-run-*-{run_id}  # from a node with network'
        _write(path, record)
        print(f'WandB mirror: {mode} run {run_id}', flush=True)
        return run
    _write(path, dict(record, mode='failed'))
    print('WandB mirror unavailable; training continues with local TensorBoard only', flush=True)
    return None


def finish(run, failed: bool = False) -> None:
    if run is None:
        return
    try:
        run.finish(exit_code=1 if failed else 0)
    except Exception as error:
        print(f'WandB finish failed; training outputs unaffected: {error!r}', flush=True)
