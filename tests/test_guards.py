"""Guards for split discipline and gold-label access."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rpc import io  # noqa: E402
from rpc.config import RAW  # noqa: E402

SRC = ROOT / "src" / "rpc"
needs_data = pytest.mark.skipif(not (RAW / "dial_attempts.csv").exists(), reason="data/raw not present")


def test_gold_not_loadable_through_generic_loader():
    with pytest.raises(PermissionError):
        io.load("verified_contact_points")


def test_only_io_reads_gold_file_directly():
    offenders = [p.name for p in SRC.glob("*.py")
                 if p.name != "io.py" and "verified_contact_points" in p.read_text(encoding="utf-8")]
    assert not offenders, f"read gold via io.load_gold instead: {offenders}"


def test_only_evaluate_requests_test_split_gold():
    pat = re.compile(r"load_gold\([^)]*['\"]test['\"]")
    offenders = [p.name for p in list(SRC.glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
                 if p.name != "evaluate.py" and pat.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"only evaluate.py may touch test gold: {offenders}"


@needs_data
def test_data_validates_clean():
    assert io.validate(io.load_all()) == []


@needs_data
def test_gold_respects_splits():
    sp = io.load("splits").set_index("account_id").split
    g = io.load_gold(splits=("train",))
    assert set(sp.loc[g.account_id]) == {"train"}
