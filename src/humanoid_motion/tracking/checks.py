"""Runtime integration checks using the actual pinned adapter and RSL storage."""
from types import SimpleNamespace
import torch
from humanoid_motion.tracking.command import MultiMotionCommand


def check_runtime_contracts(device):
    from rsl_rl.storage import RolloutStorage
    storage=RolloutStorage('rl',2,3,[1],[1],[1],device=device)
    storage.rewards[:,0,0]=torch.tensor([1.,100.,100.],device=device)
    storage.rewards[:,1,0]=torch.tensor([1.,100.,100.],device=device)
    storage.dones[0,0,0]=1
    storage.compute_returns(torch.full((2,1),100.,device=device),gamma=1.,lam=1.,normalize_advantage=False)
    assert float(storage.returns[0,0,0])==1., 'PPO return crossed episode boundary'
    assert float(storage.returns[0,1,0])==301., 'Nonterminal return chain incorrect'
    command=object.__new__(MultiMotionCommand)
    command._env=SimpleNamespace(common_step_counter=1)
    command.lengths=torch.tensor([3,7,4],device=device)
    command.offsets=torch.tensor([0,3,10],device=device)
    command.clip_ids=torch.tensor([0,1,2,1],device=device)
    command.time_steps=torch.tensor([1,3,12,8],device=device)
    command.last_advanced_step=-1
    command._update_command=lambda:None
    command.advance()
    assert command.phases.tolist()==[2,1,3,6], 'Independent clip phase mismatch'
    command._env.common_step_counter=2
    command.advance()
    assert command.phases.tolist()==[2,2,3,6], 'Last frame advanced into next clip'
    try:
        command.advance()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Double phase advancement accepted')
    command._reset_allowed=False
    try:
        command._resample_command([0])
    except RuntimeError:
        pass
    else:
        raise AssertionError('Hidden state write accepted')
    return dict(ppo_terminal_return_cut=True,variable_lengths=True,independent_environments=True,
                boundary_no_implicit_switch=True,double_advance_rejected=True,state_writes_require_reset=True)
