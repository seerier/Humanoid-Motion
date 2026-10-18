import numpy as np
import pytest
from humanoid_motion.library import bounded_probabilities, flat_indices, atomic_json


def test_variable_length_independent_indices():
    lengths=np.array([3,7,4])
    clips=np.array([1,0,2,1]); phases=np.array([6,2,0,1])
    np.testing.assert_equal(flat_indices(lengths,clips,phases),[9,2,10,4])
    with pytest.raises(ValueError):
        flat_indices(lengths,clips,np.array([7,2,0,1]))
    with pytest.raises(ValueError):
        flat_indices(lengths,np.array([3]),np.array([0]))


@pytest.mark.parametrize('rates',[np.zeros(128),np.ones(128),np.r_[1,np.zeros(127)],np.geomspace(1e-10,1,128)])
def test_probability_bounds_coverage(rates):
    p=bounded_probabilities(rates,True)
    assert abs(p.sum()-1)<1e-12
    assert p.min()>=.5/len(p)-1e-12
    assert p.max()<=2/len(p)+1e-12
    np.testing.assert_allclose(bounded_probabilities(rates,False),1/128)
    draws=np.random.default_rng(0).choice(128,100000,p=p)
    assert len(np.unique(draws))==128


def test_atomic_replacement_and_finite(tmp_path):
    import json
    path=tmp_path/'status.json'
    atomic_json(path,{'n':1});atomic_json(path,{'n':2})
    assert json.loads(path.read_text())=={'n':2}
    with pytest.raises(ValueError):
        atomic_json(path,{'n':float('nan')})
    assert json.loads(path.read_text())=={'n':2}


def test_reject_invalid_probability_inputs():
    for rates in ([],[-1,1],[float('nan'),1],[[1,2]]):
        with pytest.raises(ValueError):
            bounded_probabilities(np.asarray(rates),True)

