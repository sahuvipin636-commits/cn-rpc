import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rpc.states import StateModel, O, STATES  # noqa: E402


def seq(codes, times):
    return {"link": "x", "source": "kyc_origination", "t0": 0.0, "t": np.array(times, float),
            "code": np.array(codes), "w": np.ones(len(codes)), "gold": None}


def test_filter_ignores_later_calls():
    m = StateModel()
    a = m.filter_at(seq([O["rnr"], O["rejected"]], [1, 2]), [5.0])[0]
    b = m.filter_at(seq([O["rnr"], O["rejected"], O["does_not_exist"]], [1, 2, 6]), [5.0])[0]
    assert np.allclose(a, b) and np.isclose(a.sum(), 1)


def test_evidence_moves_states_the_right_way():
    m = StateModel()
    rej = m.filter_at(seq([O["rejected"]] * 4, [1, 2, 3, 4]), [5.0])[0]
    dead = m.filter_at(seq([O["switched_off_long"]] * 4, [1, 2, 3, 4]), [5.0])[0]
    assert rej[STATES.index("avoiding")] > rej[STATES.index("reachable")]
    assert dead[STATES.index("dead")] + dead[STATES.index("temp_off")] > 0.8


def test_confidence_decays_without_evidence():
    m = StateModel()
    s = seq([O["rpc"]] * 3, [1, 2, 3])
    soon, later = m.filter_at(s, [4.0, 120.0])
    assert soon[STATES.index("reachable")] > later[STATES.index("reachable")]
