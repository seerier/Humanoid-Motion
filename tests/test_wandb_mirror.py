"""WandB mirror contract with a fake client; no network, account or Isaac needed."""
import json
import os
import sys
import types
import pytest
from humanoid_motion import wandb_mirror as wandb_run


def fake_wandb(fail_modes=()):
    calls = []
    def init(**kw):
        calls.append(dict(kw, env_mode=os.environ.get('WANDB_MODE')))
        if kw['mode'] in fail_modes:
            raise RuntimeError(f'{kw["mode"]} refused')
        return types.SimpleNamespace(project=kw['project'], url='https://wandb.example/run', finish=lambda exit_code: calls.append(('finish', exit_code)))
    module = types.SimpleNamespace(init=init, finish=lambda: None, Settings=lambda **kw: kw,
                                   tensorboard=types.SimpleNamespace(unpatch=lambda: None),
                                   util=types.SimpleNamespace(generate_id=lambda: 'fresh01'))
    return module, calls


def record(tmp_path):
    return json.loads((tmp_path/wandb_run.RECORD).read_text())


def test_disabled_never_imports_client(tmp_path, monkeypatch):
    monkeypatch.setenv('HUMANOID_MOTION_WANDB', 'disabled')
    monkeypatch.setitem(sys.modules, 'wandb', None)
    assert wandb_run.start(tmp_path, 'single_clip', {}) is None
    assert record(tmp_path)['mode'] == 'disabled'


def test_missing_client_is_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv('HUMANOID_MOTION_WANDB', 'online')
    monkeypatch.setitem(sys.modules, 'wandb', None)
    assert wandb_run.start(tmp_path, 'single_clip', {}) is None
    assert record(tmp_path)['mode'] == 'unavailable'


def test_online_failure_falls_back_offline_and_restores_env(tmp_path, monkeypatch):
    module, calls = fake_wandb(fail_modes=('online',))
    monkeypatch.setitem(sys.modules, 'wandb', module)
    monkeypatch.setenv('HUMANOID_MOTION_WANDB', 'online')
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    run = wandb_run.start(tmp_path, 'single_clip', dict(seed=0))
    assert run is not None and [c['mode'] for c in calls] == ['online', 'offline']
    assert [c['env_mode'] for c in calls] == ['online', 'offline']
    assert os.environ['WANDB_MODE'] == 'disabled'
    saved = record(tmp_path)
    assert saved['mode'] == 'offline' and saved['url'] is None and 'wandb sync' in saved['sync']
    assert len(saved['errors']) == 1
    # Upload scope: scalars via TensorBoard, never code, git diff or console.
    assert calls[-1]['sync_tensorboard'] and not calls[-1]['save_code']
    assert calls[-1]['settings'] == dict(disable_git=True, console='off', init_timeout=60)


def test_requeue_reuses_run_id(tmp_path, monkeypatch):
    module, calls = fake_wandb()
    monkeypatch.setitem(sys.modules, 'wandb', module)
    monkeypatch.setenv('HUMANOID_MOTION_WANDB', 'online')
    (tmp_path/wandb_run.RECORD).write_text(json.dumps(dict(id='kept01', mode='online')))
    wandb_run.start(tmp_path, 'overnight_multimotion', {})
    assert calls[0]['id'] == 'kept01' and calls[0]['resume'] == 'allow'
    assert record(tmp_path)['url'] == 'https://wandb.example/run'


def test_offline_requeue_gets_fresh_id_and_keeps_earlier(tmp_path, monkeypatch):
    module, calls = fake_wandb(fail_modes=('online',))
    monkeypatch.setitem(sys.modules, 'wandb', module)
    monkeypatch.setenv('HUMANOID_MOTION_WANDB', 'online')
    (tmp_path/wandb_run.RECORD).write_text(json.dumps(dict(id='seg01', mode='offline', earlier_ids=['seg00'])))
    wandb_run.start(tmp_path, 'overnight_multimotion', {})
    assert [c['id'] for c in calls] == ['fresh01', 'fresh01']
    assert record(tmp_path)['earlier_ids'] == ['seg00', 'seg01']


def test_total_failure_returns_none(tmp_path, monkeypatch):
    module, _ = fake_wandb(fail_modes=('online', 'offline'))
    monkeypatch.setitem(sys.modules, 'wandb', module)
    monkeypatch.setenv('HUMANOID_MOTION_WANDB', 'online')
    assert wandb_run.start(tmp_path, 'single_clip', {}) is None
    assert record(tmp_path)['mode'] == 'failed'


def test_typo_fails_early(tmp_path, monkeypatch):
    monkeypatch.setenv('HUMANOID_MOTION_WANDB', 'onlin')
    with pytest.raises(ValueError):
        wandb_run.start(tmp_path, 'single_clip', {})


def test_finish_errors_do_not_propagate():
    def broken(exit_code):
        raise OSError('network gone')
    wandb_run.finish(types.SimpleNamespace(finish=broken), failed=True)
    wandb_run.finish(None)


def test_wandb_dir_inside_tensorboard_logdir_is_rejected(tmp_path, monkeypatch):
    module, _ = fake_wandb()
    monkeypatch.setitem(sys.modules, 'wandb', module)
    monkeypatch.setenv('HUMANOID_MOTION_WANDB', 'online')
    monkeypatch.setenv('WANDB_DIR', str(tmp_path/'nested'))
    with pytest.raises(ValueError):
        wandb_run.start(tmp_path, 'single_clip', {})
