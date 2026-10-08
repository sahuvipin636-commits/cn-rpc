import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rpc.remarks import tag_dial_remarks, tag_field_remarks  # noqa: E402


def dial(text):
    return tag_dial_remarks(pd.DataFrame({"attempt_id": ["a"], "remark": [text]})).iloc[0]


def field(text):
    return tag_field_remarks(pd.DataFrame({"visit_id": ["v"], "remark": [text]})).iloc[0]


def test_typos_snap_to_the_right_phrase():
    assert dial("hun gup mid call").tag_hung_up
    assert dial("2 ring ek baad disconnect").tag_rejected
    assert dial("notr eachable").tag_not_reachable


def test_three_languages():
    assert dial("call cut maadidru").lang == "kn" and dial("call cut maadidru").tag_rejected
    assert dial("hosa number antharu, customer gottilla").tag_new_owner
    assert dial("avara hendathi pick maadidru").tag_tp_spouse


def test_extraction():
    r = dial("number band hai 2 mahine se")
    assert r.tag_long_off and r.months == 2
    r = dial("PTP 30/04, amount 12,500")
    assert (r.ptp_day, r.ptp_month, r.amount) == (30, 4, 12500)
    assert dial("bot: shaam ko call back").tag_evening and dial("bot: RNR").is_bot


def test_field_multipart_and_lakh_amounts():
    r = field("1,77,600 cash collect; nija mane Milk Dairy hinde ide, 2 cross munde")
    assert r.tag_cash and r.tag_address_correction and r.amount == 177600
    assert field("avara hendathi sikkidru, kelsakke hogidaare antharu").tag_at_work
