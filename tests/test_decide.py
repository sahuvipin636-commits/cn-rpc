import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rpc.safety import apply_safety, DEFAULTS  # noqa: E402
from rpc.decide import assign_hours, account_aggregates, OWNED_LIVE_BLOCKS_TRACE  # noqa: E402


def num(**kw):
    base = dict(p_reachable=.8, p_avoiding=.1, p_temp_off=0, p_dead=0, p_recycled=0, p_third_party=.05,
                p_invalid=0, wrong_after_rpc=0, days_since_wrong=np.nan, days_since_rpc=3.0,
                new_owner_after_rpc=0, n_does_not_exist=0, dead_streak=0)
    base.update(kw)
    return pd.DataFrame([base])


def test_safety_tiers():
    assert apply_safety(num()).status[0] == "dial"
    assert apply_safety(num(p_recycled=.3, p_reachable=.5)).status[0] == "verify_first"
    assert apply_safety(num(p_recycled=.9, p_reachable=0)).status[0] == "suppress_recycled"
    assert apply_safety(num(new_owner_after_rpc=1)).status[0] == "suppress_recycled"
    assert apply_safety(num(n_does_not_exist=1)).status[0] == "drop_invalid"
    # one stranger after the borrower: verify, never plain dial, never silent suppression
    assert apply_safety(num(wrong_after_rpc=1, days_since_wrong=1.0)).status[0] == "verify_first"


def test_avoider_blocks_trace():
    n = num(p_reachable=.2, p_avoiding=.7).assign(account_id="A", status="dial", acc_reject_rate=.6,
                                                     traces_n=0, acc_att=10)
    assert account_aggregates(n).best_owned_live.iloc[0] >= OWNED_LIVE_BLOCKS_TRACE


def test_hours_respect_capacity():
    plan = pd.DataFrame({"account_id": list("abc"), "action": "call", "ev": [3, 2, 1]})
    by_hour = pd.DataFrame({8: [.3, .3, .3], 18: [.2, .2, .2]}, index=list("abc"))
    out = assign_hours(plan, by_hour, {8: 1, 18: 1})
    assert out.set_index("account_id").scheduled_hour.to_dict() == {"a": 8, "b": 18, "c": None} or \
        out.scheduled_hour.isna().sum() == 1 and (out.scheduled_hour == 8).sum() == 1
