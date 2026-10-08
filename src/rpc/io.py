"""Stage 1: load CN's CSVs with correct types, validate them, and guard the gold labels."""
from __future__ import annotations

import pandas as pd

from .config import RAW, SPLITS, RPC_DISPOSITIONS, THIRD_PARTY_DISPOSITIONS, DEAD_RESPONSES

# file -> (expected columns, timestamp columns)
TABLES = {
    "accounts": (["account_id", "lender_id", "portfolio", "income_type", "town_id", "preferred_language",
                  "bucket_start", "dpd_start", "emi_amount", "overdue_start", "outstanding",
                  "salary_credit_day", "bureau_score_band", "other_active_loans", "paid_other_lenders_30d",
                  "last_bounce_reason", "ability_to_pay_estimate", "prev_ptp_count", "prev_ptp_broken",
                  "dialling_arm"], []),
    "splits": (["account_id", "split"], []),
    "lenders": (["lender_id", "lender_name", "lender_type", "kyc_address_format"], []),
    "agents": (["agent_id", "channel", "language_team", "town_id", "tenure_months", "shift"], []),
    "phones": (["phone_id", "account_id", "source", "relation_recorded", "added_date", "phone_masked",
                "priority_slot"], ["added_date"]),
    "dial_attempts": (["attempt_id", "account_id", "phone_id", "attempt_ts", "channel", "agent_id",
                       "dialling_arm", "selection_propensity", "network_response", "ring_duration_s",
                       "talk_duration_s", "hangup_by", "disposition", "remark", "ptp_id", "has_transcript"],
                      ["attempt_ts"]),
    "payments": (["payment_id", "account_id", "payment_ts", "amount", "channel"], ["payment_ts"]),
    "skip_traces": (["trace_id", "account_id", "trace_date", "trigger_rule", "result",
                     "new_contact_point_id", "cost_inr"], ["trace_date"]),
    "addresses": (["address_id", "account_id", "address_type", "source", "added_date", "town_id",
                   "address_text"], ["added_date"]),
    "field_visits": (["visit_id", "account_id", "address_id", "agent_id", "visit_date", "start_ts",
                      "checkin_ts", "checkin_x", "checkin_y", "gps_accuracy_m", "dwell_s", "outcome",
                      "ptp_id", "remark", "photo_hash"], ["visit_date", "start_ts", "checkin_ts"]),
}
_GOLD_FILE = "verified_contact_points"


def load(name: str) -> pd.DataFrame:
    """Load one table with parsed timestamps. Gold labels are not loadable here."""
    if name == _GOLD_FILE:
        raise PermissionError("Use load_gold(splits=...) for verified labels.")
    cols, ts_cols = TABLES[name]
    df = pd.read_csv(RAW / f"{name}.csv", parse_dates=ts_cols, keep_default_na=True)
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name}.csv is missing columns: {sorted(missing)}")
    for c in df.columns:  # plain object strings keep downstream code simple
        if pd.api.types.is_string_dtype(df[c]) and c not in ts_cols:
            df[c] = df[c].astype(object)
    if name == "dial_attempts":
        df["remark"] = df["remark"].fillna("")
        df = df.sort_values("attempt_ts", kind="stable").reset_index(drop=True)
    return df


def load_all() -> dict[str, pd.DataFrame]:
    return {n: load(n) for n in TABLES}


def load_gold(splits: tuple[str, ...] = ("train",)) -> pd.DataFrame:
    """Verified contact points, restricted to the requested account splits.
    Only evaluate.py may ask for 'test'."""
    bad = set(splits) - set(SPLITS)
    if bad:
        raise ValueError(f"unknown splits {bad}")
    gold = pd.read_csv(RAW / f"{_GOLD_FILE}.csv", parse_dates=["verified_date"])
    sp = load("splits")
    keep = sp.loc[sp.split.isin(splits), "account_id"]
    return gold[gold.account_id.isin(keep)].reset_index(drop=True)


def add_flags(att: pd.DataFrame) -> pd.DataFrame:
    """Derived outcome flags used everywhere. Logged flags are as typed by agents (noisy)."""
    att = att.copy()
    att["answered"] = att.network_response.eq("answered")
    att["rejected"] = att.network_response.eq("busy_rejected")
    att["dead_resp"] = att.network_response.isin(DEAD_RESPONSES)
    att["logged_rpc"] = att.disposition.isin(RPC_DISPOSITIONS)
    att["logged_tp"] = att.disposition.isin(THIRD_PARTY_DISPOSITIONS)
    att["logged_wrong"] = att.disposition.eq("wrong_number")
    att["impossible_rpc"] = att.logged_rpc & ~att.answered   # RPC typed on a call nobody answered
    att["hour"] = att.attempt_ts.dt.hour
    att["date"] = att.attempt_ts.dt.normalize()
    return att


def validate(t: dict[str, pd.DataFrame]) -> list[str]:
    """Referential and sanity checks. Returns a list of problems (empty = clean)."""
    issues = []
    acc = set(t["accounts"].account_id)

    def check(cond: bool, msg: str):
        if not cond:
            issues.append(msg)

    check(t["accounts"].account_id.is_unique, "accounts.account_id not unique")
    check(set(t["splits"].account_id) == acc, "splits do not cover exactly the accounts")
    check(not t["phones"].duplicated(["account_id", "phone_id"]).any(), "duplicate (account, phone)")
    for name in ["phones", "dial_attempts", "payments", "skip_traces", "addresses", "field_visits"]:
        check(t[name].account_id.isin(acc).all(), f"{name} has unknown account_id")
    links = set(map(tuple, t["phones"][["account_id", "phone_id"]].values))
    att_links = set(map(tuple, t["dial_attempts"][["account_id", "phone_id"]].values))
    check(att_links <= links, f"{len(att_links - links)} dialled (account, phone) not in phones.csv")
    check(t["dial_attempts"].agent_id.isin(t["agents"].agent_id).all(), "dial_attempts has unknown agent")
    check(t["field_visits"].address_id.isin(t["addresses"].address_id).all(), "visit to unknown address")
    h = t["dial_attempts"].attempt_ts.dt.hour
    check(h.between(8, 18).all(), f"{(~h.between(8, 18)).sum()} calls outside 08:00 to 19:00")
    p = t["dial_attempts"].selection_propensity
    check(p.between(0, 1).all() and (p > 0).all(), "selection_propensity outside (0, 1]")
    return issues
