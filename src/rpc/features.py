"""Stage 3a: features for every (account, phone) as of a cutoff date D.

Strict as-of rule: only events with timestamp < D are used. The same function works for
any date, so production can run it daily; training uses weekly cutoffs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import WINDOW_START

DEAD = "dead_resp"
TAG_COUNTS = ["new_owner", "not_known", "long_off", "no_network", "evening", "hardship", "refused",
              "hung_up", "ptp", "borrower_away", "language_barrier", "claims_paid", "dispute",
              "tp_office", "tp_spouse", "tp_parent", "tp_sibling"]
HOUR_BANDS = {"morning": (8, 10), "midday": (10, 16), "evening": (16, 19)}
TRUSTED = 0.5          # a logged RPC with trust >= this counts as a confirmed borrower contact


def enrich_attempts(att: pd.DataFrame, remarks: pd.DataFrame, trust: pd.DataFrame) -> pd.DataFrame:
    """Join flags, remark tags and RPC trust onto attempts (att must have add_flags columns)."""
    tags = remarks[["attempt_id"] + [f"tag_{t}" for t in TAG_COUNTS]]
    e = att.merge(tags, on="attempt_id", how="left").merge(trust, on="attempt_id", how="left")
    e["rpc_soft"] = e.rpc_trust.fillna(0.0)
    e["rpc_trusted"] = e.rpc_soft >= TRUSTED
    e["alive_resp"] = ~e[DEAD]
    e["link"] = e.account_id + "|" + e.phone_id
    for band, (lo, hi) in HOUR_BANDS.items():
        e[f"in_{band}"] = e.hour.between(lo, hi - 1)
    return e


def _link_history(h: pd.DataFrame, D: pd.Timestamp) -> pd.DataFrame:
    g = h.groupby("link")
    f = pd.DataFrame({
        "n_att": g.size(),
        "n_answered": g.answered.sum(), "n_rejected": g.rejected.sum(), "n_dead": g[DEAD].sum(),
        "n_switched_off": g.network_response.apply(lambda s: (s == "switched_off").sum()),
        "n_not_reachable": g.network_response.apply(lambda s: (s == "not_reachable").sum()),
        "n_does_not_exist": g.network_response.apply(lambda s: (s == "number_does_not_exist").sum()),
        "rpc_soft": g.rpc_soft.sum(), "n_logged_rpc": g.logged_rpc.sum(),
        "n_tp": g.logged_tp.sum(), "n_wrong": g.logged_wrong.sum(),
        "n_bot": g.channel.apply(lambda s: (s == "voice_bot").sum()),
        **{f"tag_{t}": g[f"tag_{t}"].sum() for t in TAG_COUNTS},
    })
    for c in ["answered", "rejected", "dead"]:
        f[f"{c}_rate"] = f[f"n_{c}"] / f.n_att
    f["rpc_rate"] = f.rpc_soft / f.n_att
    f["tp_share_answered"] = f.n_tp / f.n_answered.replace(0, np.nan)
    f["wrong_share_answered"] = f.n_wrong / f.n_answered.replace(0, np.nan)
    f["rpc_share_answered"] = f.rpc_soft / f.n_answered.replace(0, np.nan)

    days = lambda ts: (D - ts).dt.total_seconds() / 86400
    f["days_since_last_att"] = days(g.attempt_ts.max())
    f["days_since_first_att"] = days(g.attempt_ts.min())
    last = lambda mask: h[mask].groupby("link").attempt_ts.max()
    first = lambda mask: h[mask].groupby("link").attempt_ts.min()
    f["days_since_rpc"] = days(last(h.rpc_trusted))
    f["days_since_answered"] = days(last(h.answered))
    f["days_since_alive"] = days(last(h.alive_resp))
    f["days_since_tp"] = days(last(h.logged_tp))
    f["days_since_wrong"] = days(last(h.logged_wrong))

    # streaks at the end of history
    lt_alive = last(h.alive_resp).reindex(f.index)
    lt_rpc = last(h.rpc_trusted).reindex(f.index)
    hh = h.join(lt_alive.rename("lt_alive"), on="link").join(lt_rpc.rename("lt_rpc"), on="link")
    f["dead_streak"] = (hh[DEAD] & (hh.lt_alive.isna() | (hh.attempt_ts > hh.lt_alive))).groupby(hh.link).sum()
    f["no_rpc_streak"] = (hh.lt_rpc.isna() | (hh.attempt_ts > hh.lt_rpc)).groupby(hh.link).sum()
    # recycled change-point: borrower confirmed earlier, then strangers or a long dead stretch
    fr = first(h.rpc_trusted).reindex(f.index)
    hh = hh.join(fr.rename("first_rpc"), on="link")
    after = hh.first_rpc.notna() & (hh.attempt_ts > hh.first_rpc)
    f["wrong_after_rpc"] = (after & hh.logged_wrong).groupby(hh.link).sum()
    f["new_owner_after_rpc"] = (after & hh.tag_new_owner.fillna(False)).groupby(hh.link).sum()
    f["dead_streak_after_rpc"] = np.where(fr.notna(), f.dead_streak, 0)
    f["last_response"] = h.groupby("link").network_response.last()

    # recent window
    r = h[h.attempt_ts >= D - pd.Timedelta(days=14)].groupby("link")
    f["att_14d"] = r.size()
    f["dead_14d"] = r[DEAD].sum()
    f["rejected_14d"] = r.rejected.sum()
    f["rpc_14d"] = r.rpc_soft.sum()
    return f


def _account_history(h: pd.DataFrame, D: pd.Timestamp) -> pd.DataFrame:
    g = h.groupby("account_id")
    f = pd.DataFrame({"acc_att": g.size(), "acc_answered_rate": g.answered.mean(),
                      "acc_reject_rate": g.rejected.mean(), "acc_rpc_soft": g.rpc_soft.sum(),
                      "acc_evening_callback": g.tag_evening.sum(), "acc_hardship": g.tag_hardship.sum(),
                      "acc_refused": g.tag_refused.sum(), "acc_ptp": g.tag_ptp.sum()})
    f["acc_days_since_rpc"] = (D - h[h.rpc_trusted].groupby("account_id").attempt_ts.max()).dt.days
    # when does this borrower pick up (shared across all their numbers)
    for band in HOUR_BANDS:
        m = h[h[f"in_{band}"]]
        gb = m.groupby("account_id")
        f[f"acc_att_{band}"] = gb.size()
        f[f"acc_rpc_rate_{band}"] = gb.rpc_soft.sum() / gb.size()
    return f


def _field_history(v: pd.DataFrame, vr: pd.DataFrame, D: pd.Timestamp) -> pd.DataFrame:
    x = v[v.checkin_ts < D].merge(vr, on="visit_id")
    if x.empty:
        return pd.DataFrame()
    g = x.groupby("account_id")
    f = pd.DataFrame({"fv_n": g.size(),
                      **{f"fv_{o}": g.outcome.apply(lambda s, o=o: (s == o).sum())
                         for o in ["met_borrower", "met_family", "locked_premises", "address_not_traceable",
                                   "neighbour_says_shifted", "no_such_person", "cash_collected"]},
                      "fv_at_work": g.tag_at_work.sum(), "fv_at_shop": g.tag_at_shop.sum()})
    f["fv_days_since_met"] = (D - x[x.outcome.isin(["met_borrower", "cash_collected"])]
                              .groupby("account_id").checkin_ts.max()).dt.days
    return f


def build_snapshot(D: pd.Timestamp, E: pd.DataFrame, t: dict, field_remarks: pd.DataFrame) -> pd.DataFrame:
    """Features for every (account, phone) known before D. E = enrich_attempts(...)."""
    D = pd.Timestamp(D)
    ph = t["phones"][t["phones"].added_date <= D].copy()   # a number added on day D is known on day D
    ph["link"] = ph.account_id + "|" + ph.phone_id
    known = t["phones"][t["phones"].added_date <= D]
    deg = known.groupby("phone_id").account_id.nunique().rename("shared_accounts")
    lend = (known.merge(t["accounts"][["account_id", "lender_id"]])
            .groupby("phone_id").lender_id.nunique().rename("shared_lenders"))
    base = ph.join(deg, on="phone_id").join(lend, on="phone_id")
    base["days_on_file"] = (D - base.added_date).dt.days
    base["n_numbers_on_account"] = base.groupby("account_id").phone_id.transform("size")
    base = base.drop(columns=["added_date", "phone_masked"])

    acc = t["accounts"].copy()
    acc["dpd_now"] = acc.dpd_start + (D - WINDOW_START).days
    acc["salary_day_known"] = acc.salary_credit_day.notna()
    base = base.merge(acc, on="account_id", how="left").merge(t["splits"], on="account_id", how="left")

    h = E[E.attempt_ts < D]
    if len(h):
        base = base.join(_link_history(h, D), on="link")
        base = base.join(_account_history(h, D), on="account_id")
        base["rpc_on_other_numbers"] = base.acc_rpc_soft.fillna(0) - base.rpc_soft.fillna(0)
    pay = t["payments"][t["payments"].payment_ts < D].groupby("account_id")
    base = base.join(pd.DataFrame({"pay_n": pay.size(), "pay_amount": pay.amount.sum(),
                                   "pay_days_since": (D - pay.payment_ts.max()).dt.days}), on="account_id")
    tr = t["skip_traces"][t["skip_traces"].trace_date < D].groupby("account_id")
    base = base.join(pd.DataFrame({"traces_n": tr.size(),
                                   "traces_found": tr.result.apply(lambda s: (s != "no_new_info").sum())}),
                     on="account_id")
    fv = _field_history(t["field_visits"], field_remarks, D)
    if len(fv):
        base = base.join(fv, on="account_id")
    base.insert(0, "cutoff", D)
    return fill_counts(base)


COUNT_PREFIXES = ("n_att", "n_answered", "n_rejected", "n_dead", "n_switched", "n_not_", "n_does", "n_logged",
                  "n_tp", "n_wrong", "n_bot", "tag_", "acc_att", "fv_")
COUNT_COLS = ["rpc_soft", "dead_streak", "no_rpc_streak", "wrong_after_rpc", "new_owner_after_rpc",
              "dead_streak_after_rpc", "att_14d", "dead_14d", "rejected_14d", "rpc_14d", "acc_rpc_soft",
              "acc_evening_callback", "acc_hardship", "acc_refused", "acc_ptp", "rpc_on_other_numbers",
              "pay_n", "pay_amount", "traces_n", "traces_found"]


def fill_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Counts become 0 when there is no history; rates and 'days since' stay NaN (unknown, not zero)."""
    cols = [c for c in df.columns if (c.startswith(COUNT_PREFIXES) and c != "fv_days_since_met") or c in COUNT_COLS]
    df[cols] = df[cols].fillna(0)
    return df

