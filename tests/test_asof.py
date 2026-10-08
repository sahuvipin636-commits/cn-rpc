"""A snapshot at D must be identical whether or not the future exists in the data."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rpc.config import PROCESSED, RAW  # noqa: E402
from rpc.io import load_all, add_flags  # noqa: E402
from rpc.features import enrich_attempts, build_snapshot  # noqa: E402

ready = (RAW / "dial_attempts.csv").exists() and (PROCESSED / "remarks_dial.parquet").exists()


@pytest.mark.skipif(not ready, reason="needs data/raw and Stage 2 outputs")
@pytest.mark.parametrize("cutoff", ["2026-04-22", "2026-06-03"])
def test_snapshot_ignores_the_future(cutoff):
    D = pd.Timestamp(cutoff)
    t = load_all()
    E = enrich_attempts(add_flags(t["dial_attempts"]), pd.read_parquet(PROCESSED / "remarks_dial.parquet"),
                        pd.read_parquet(PROCESSED / "attempt_trust.parquet"))
    rf = pd.read_parquet(PROCESSED / "remarks_field.parquet")
    full = build_snapshot(D, E, t, rf)

    past = dict(t)
    past["phones"] = t["phones"][t["phones"].added_date <= D]
    past["payments"] = t["payments"][t["payments"].payment_ts < D]
    past["skip_traces"] = t["skip_traces"][t["skip_traces"].trace_date < D]
    past["field_visits"] = t["field_visits"][t["field_visits"].checkin_ts < D]
    trunc = build_snapshot(D, E[E.attempt_ts < D], past, rf)

    key = ["account_id", "phone_id"]
    a = full.sort_values(key).reset_index(drop=True)
    b = trunc.sort_values(key).reset_index(drop=True)[a.columns]
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
