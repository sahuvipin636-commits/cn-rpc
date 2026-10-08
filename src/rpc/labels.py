"""Stage 3b: weekly snapshots, labels, and the attempt-level modelling table.

Two training views:
  link_week : one row per (cutoff, account, phone): did it reach the borrower in the next 7 days?
              (used for the per-contact-point AUC / calibration metric CN asks for)
  attempts  : every call in the 7 days after a cutoff, joined to that cutoff's features, with the
              outcome split into the three factors:
                y_alive    line works (not switched off / unreachable / non-existent)
                y_picked   a human picked up or rejected, given alive
                y_rpc_soft it was the borrower, given answered (trust-weighted)
              plus the hour, channel and propensity of the call (for the best-hour model and IPS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import WINDOW_START, WINDOW_END
from .features import build_snapshot, fill_counts

HORIZON_DAYS = 7


def weekly_cutoffs() -> list[pd.Timestamp]:
    return list(pd.date_range(WINDOW_START, WINDOW_END - pd.Timedelta(days=1), freq=f"{HORIZON_DAYS}D"))


SCORING_CUTOFF = WINDOW_END   # features as of 30 June: used to predict the 2 July verification


def build_snapshots(cutoffs, E, t, field_remarks) -> pd.DataFrame:
    snaps = [build_snapshot(D, E, t, field_remarks) for D in cutoffs]
    return fill_counts(pd.concat(snaps, ignore_index=True))


def assign_cutoff(ts: pd.Series, cutoffs) -> pd.Series:
    cut = pd.DatetimeIndex(sorted(cutoffs))
    idx = cut.searchsorted(ts, side="right") - 1
    out = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    ok = idx >= 0
    out[ok] = cut[idx[ok]]
    return out


def link_week_labels(E: pd.DataFrame, cutoffs) -> pd.DataFrame:
    e = E.assign(cutoff=assign_cutoff(E.attempt_ts, cutoffs))
    e["miss"] = 1 - e.rpc_soft                       # P(this call was NOT a borrower contact)
    g = e.groupby(["cutoff", "link"])
    lab = pd.DataFrame({"next_att": g.size(), "next_dead_share": g.dead_resp.mean(),
                        "next_rpc_soft": 1 - g.miss.prod(), "next_rpc_any": g.rpc_trusted.any()})
    return lab.reset_index()


def attempt_table(E: pd.DataFrame, snaps: pd.DataFrame, agents: pd.DataFrame, cutoffs) -> pd.DataFrame:
    e = E.assign(cutoff=assign_cutoff(E.attempt_ts, cutoffs))
    lang = agents.set_index("agent_id").language_team
    a = pd.DataFrame({
        "attempt_id": e.attempt_id, "cutoff": e.cutoff, "link": e.link, "attempt_ts": e.attempt_ts,
        "hour": e.hour, "dow": e.attempt_ts.dt.dayofweek, "is_bot": e.channel.eq("voice_bot"),
        "agent_id": e.agent_id, "agent_lang": e.agent_id.map(lang),
        "dialling_arm": e.dialling_arm, "propensity": e.selection_propensity,
        "y_alive": e.alive_resp, "y_picked": e.answered | e.rejected, "y_answered": e.answered,
        "y_rejected": e.rejected, "y_rpc_soft": e.rpc_soft, "y_rpc_logged": e.logged_rpc,
        "y_tp": e.logged_tp, "y_wrong": e.logged_wrong,
    })
    # never carry next-week labels into the attempt table: they are outcomes, not features
    feats = snaps.drop(columns=[c for c in snaps.columns if c == "dialling_arm" or c.startswith("next_")])
    out = a.merge(feats, on=["cutoff", "link"], how="inner").copy()
    out["lang_match"] = out.agent_lang.eq("all") | out.agent_lang.eq(out.preferred_language)
    out.attrs["dropped_no_snapshot"] = len(a) - len(out)
    return out
