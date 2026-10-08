"""Stage 6b: per-account decision with a plain-language reason.

For each account, compare expected value in rupees (net of cost):
  call   best allowed number at its best hour:  P(RPC) x value_of_RPC - call cost
         (voice bot or tele-agent, whichever nets more)
  visit  best address:  P(meet) x P(pay | met) x expected amount - visit cost
  trace  P(trace finds a number) x payment uplift from a new number x expected amount - trace cost
         only if no live, borrower-owned number exists (avoiders are not lost; tracing them wastes money)
  wait   0
All value parameters are estimated on train accounts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import COSTS

CALLABLE = ("dial", "verify_first")
OWNED_LIVE_BLOCKS_TRACE = 0.30     # if any dialable number has P(alive & owned) above this, do not trace
VISIT_SKIP_OUTCOMES = {"neighbour_says_shifted", "no_such_person"}


# ---------------------------------------------------------------- value parameters
def fit_value_params(t: dict, trust: pd.DataFrame, train_ids: set) -> dict:
    a = t["dial_attempts"].merge(trust, on="attempt_id", how="left")
    a = a[a.account_id.isin(train_ids) & (a.rpc_trust >= .5)][["attempt_id", "account_id", "attempt_ts"]]
    pay = t["payments"]
    m = a.merge(pay[["account_id", "payment_ts", "amount"]], on="account_id", how="left")
    m["hit"] = (m.payment_ts > m.attempt_ts) & (m.payment_ts <= m.attempt_ts + pd.Timedelta(days=7))
    per = m.groupby("attempt_id").agg(account_id=("account_id", "first"), hit=("hit", "any"))
    per = per.merge(t["accounts"][["account_id", "bucket_start", "overdue_start"]], on="account_id")
    amt = m[m.hit].groupby("account_id").amount.median()
    ratio = (amt / t["accounts"].set_index("account_id").overdue_start).dropna().median()

    v = t["field_visits"][t["field_visits"].account_id.isin(train_ids)]
    vm = v.merge(pay[["account_id", "payment_ts"]], on="account_id", how="left")
    vm["hit"] = (vm.payment_ts >= vm.checkin_ts) & (vm.payment_ts <= vm.checkin_ts + pd.Timedelta(days=7))
    vh = vm.groupby("visit_id").agg(outcome=("outcome", "first"), hit=("hit", "any"))
    met = vh.outcome.isin(["met_borrower", "cash_collected"])

    s = t["skip_traces"][t["skip_traces"].account_id.isin(train_ids)]
    sm = s.merge(pay[["account_id", "payment_ts"]], on="account_id", how="left")
    sm["hit"] = (sm.payment_ts > sm.trace_date) & (sm.payment_ts <= sm.trace_date + pd.Timedelta(days=30))
    sh = sm.groupby("trace_id").agg(found=("result", lambda x: x.iloc[0] == "new_phone_found"), hit=("hit", "any"))
    return {
        "p_pay_given_rpc": per.groupby("bucket_start").hit.mean().to_dict(),
        "amount_ratio": float(ratio),
        "p_meet_base": float(met.mean()),
        # cash collected on the visit is itself a payment
        "p_pay_given_meet": float((vh[vh.outcome == "met_borrower"].hit.sum() + (vh.outcome == "cash_collected").sum())
                                  / max(met.sum(), 1)),
        "trace_uplift": float(sh[sh.found].hit.mean() - sh[~sh.found].hit.mean()),
    }


def expected_amount(acc: pd.DataFrame, params: dict) -> pd.Series:
    return np.maximum(acc.emi_amount, params["amount_ratio"] * acc.overdue_start)


def value_of_rpc(acc: pd.DataFrame, params: dict) -> pd.Series:
    p = acc.bucket_start.map(params["p_pay_given_rpc"]).fillna(np.mean(list(params["p_pay_given_rpc"].values())))
    return p * expected_amount(acc, params)


# ---------------------------------------------------------------- account aggregates
def account_aggregates(n: pd.DataFrame) -> pd.DataFrame:
    """n: numbers with state probabilities and safety status."""
    owned_live = (n.p_reachable + n.p_avoiding)
    dialable = n.status.isin(CALLABLE)
    g = n.assign(owned_live=owned_live, owned_live_dialable=np.where(dialable, owned_live, 0),
                 deadish=n.p_dead + n.p_temp_off,
                 avoid_w=n.p_avoiding).groupby("account_id")
    agg = pd.DataFrame({
        "best_owned_live": g.owned_live_dialable.max(),
        "any_owned_live": g.owned_live.max(),
        "avoid_share": g.avoid_w.sum() / g.owned_live.sum().clip(lower=1e-9),
        "mean_dead": g.deadish.mean(), "mean_third_party": g.p_third_party.mean(),
        "mean_recycled": g.p_recycled.mean(), "n_numbers": g.size(),
        "n_dialable": g.status.apply(lambda s: s.isin(CALLABLE).sum()),
        "share_recycled_or_invalid": g.status.apply(lambda s: s.isin(["suppress_recycled", "drop_invalid"]).mean()),
    })
    first = n.groupby("account_id").first()
    for c in ["acc_reject_rate", "traces_n", "acc_att"]:
        agg[c] = first[c].fillna(0) if c in first else 0
    return agg


TRACE_FEATURES = ["best_owned_live", "avoid_share", "mean_dead", "mean_third_party", "mean_recycled",
                  "n_numbers", "acc_reject_rate", "traces_n"]


class TraceModel:
    """P(skip-trace finds a new phone), from account state at the time of the trace."""

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.lr = LogisticRegression(C=1.0, max_iter=1000)
        self.mu, self.sd = X[TRACE_FEATURES].mean(), X[TRACE_FEATURES].std().replace(0, 1)
        self.lr.fit(((X[TRACE_FEATURES] - self.mu) / self.sd).fillna(0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(((X[TRACE_FEATURES] - self.mu) / self.sd).fillna(0))[:, 1]

    def coefficients(self) -> pd.Series:
        return pd.Series(self.lr.coef_[0], index=TRACE_FEATURES).sort_values()


# ---------------------------------------------------------------- field option
def field_options(D: pd.Timestamp, t: dict, field_remarks: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Best address per account, P(meet) there, and when to go."""
    adr = t["addresses"][(t["addresses"].added_date <= D) & (t["addresses"].town_id != "OUT")]
    v = t["field_visits"][t["field_visits"].checkin_ts < D].merge(field_remarks, on="visit_id")
    k, base = 4.0, params["p_meet_base"]
    g = v.assign(met=v.outcome.isin(["met_borrower", "cash_collected"])).sort_values("checkin_ts") \
         .groupby("address_id")
    hist = pd.DataFrame({"visits": g.size(), "met": g.met.sum(), "last_outcome": g.outcome.last()})
    a = adr.join(hist, on="address_id")
    a[["visits", "met"]] = a[["visits", "met"]].fillna(0)
    a["p_meet"] = (a.met + base * k) / (a.visits + k)
    a.loc[a.last_outcome.isin(VISIT_SKIP_OUTCOMES), "p_meet"] = 0.0
    a.loc[a.address_type.eq("office"), "p_meet"] *= 0.8
    best = a.sort_values("p_meet", ascending=False).groupby("account_id").first()
    acc_tags = v.groupby("account_id").agg(at_work=("tag_at_work", "sum"), at_shop=("tag_at_shop", "sum"))
    best = best.join(acc_tags)
    best["visit_when"] = np.select(
        [best.at_work.fillna(0) > 0, best.at_shop.fillna(0) > 0],
        ["before 10:00 or after 17:00 (family says borrower is at work in the day)",
         "during shop hours (borrower has been met at their shop)"], "10:00 to 13:00 (usual round)")
    return best[["address_id", "address_type", "p_meet", "visits", "last_outcome", "visit_when"]]


# ---------------------------------------------------------------- per-account decision
def _pct(x) -> str:
    return f"{100 * x:.0f}%"


def decide(numbers: pd.DataFrame, accounts: pd.DataFrame, params: dict, trace_model: TraceModel,
           fields: pd.DataFrame) -> pd.DataFrame:
    """numbers: one row per number with status, p_rpc_agent, p_rpc_bot, best_hour, factor probs."""
    acc = accounts.set_index("account_id")
    V = value_of_rpc(acc, params)
    amount = expected_amount(acc, params)
    n = numbers.copy()
    n["V"] = n.account_id.map(V)
    n["ev_agent"] = n.p_rpc_agent * n.V - COSTS["tele_agent"]
    n["ev_bot"] = n.p_rpc_bot * n.V - COSTS["voice_bot"]
    n["ev_call"] = np.where(n.status.isin(CALLABLE), np.maximum(n.ev_agent, n.ev_bot), -np.inf)
    n["call_channel"] = np.where(n.ev_agent >= n.ev_bot, "tele_agent", "voice_bot")
    best = n.sort_values("ev_call", ascending=False).groupby("account_id").first()

    agg = account_aggregates(n)
    agg["p_find"] = trace_model.predict(agg)
    out = agg.join(best[["phone_id", "source", "relation_recorded", "status", "ev_call", "call_channel",
                         "safety_reason",
                         "best_hour", "p_rpc_agent", "p_rpc_bot", "f_alive", "f_answer", "f_owned"]])
    out = out.join(fields, how="left")
    out["amount"] = amount.reindex(out.index)
    out["ev_visit"] = out.p_meet.fillna(0) * params["p_pay_given_meet"] * out.amount - COSTS["field_visit"]
    out["ev_trace"] = out.p_find * params["trace_uplift"] * out.amount - COSTS["skip_trace"]
    out.loc[out.best_owned_live >= OWNED_LIVE_BLOCKS_TRACE, "ev_trace"] = -np.inf
    out["ev_call"] = out.ev_call.fillna(-np.inf)

    options = out[["ev_call", "ev_visit", "ev_trace"]].assign(ev_wait=0.0)
    out["action"] = options.idxmax(axis=1).str.replace("ev_", "", regex=False)
    out["ev"] = options.max(axis=1)
    out["reason"] = [_reason(r) for r in out.itertuples()]
    return out.reset_index(), n


def _reason(r) -> str:
    if r.action == "call":
        who = "voice bot" if r.call_channel == "voice_bot" else "agent"
        weakest = min([("line may be off", r.f_alive), ("may not pick up", r.f_answer),
                       ("may not be the borrower", r.f_owned)], key=lambda x: x[1])
        verify = f" VERIFY FIRST: {r.safety_reason}." if r.status == "verify_first" else ""
        return (f"Call {r.phone_id} ({r.source}) by {who} at {int(r.best_hour):02d}:00: "
                f"{_pct(r.p_rpc_agent if who == 'agent' else r.p_rpc_bot)} chance of reaching the borrower, "
                f"worth Rs {r.ev_call:,.0f} net. Main risk: {weakest[0]} ({_pct(weakest[1])}).{verify}")
    if r.action == "visit":
        why = ("borrower holds a live number but avoids calls" if r.avoid_share >= .5 and r.any_owned_live >= .3
               else "no number is likely to reach the borrower")
        return (f"Field visit to {r.address_id} ({r.address_type}), {r.visit_when}: {why}; "
                f"{_pct(r.p_meet)} chance of meeting, worth Rs {r.ev_visit:,.0f} net.")
    if r.action == "trace":
        return (f"Skip-trace: no live borrower number (best {_pct(r.best_owned_live)}); of {int(r.n_numbers)} numbers, "
                f"{_pct(r.share_recycled_or_invalid)} recycled or invalid, {_pct(r.mean_dead)} look dead, "
                f"{_pct(r.mean_third_party)} reach third parties. "
                f"{_pct(r.p_find)} chance of finding a number, worth Rs {r.ev_trace:,.0f} net of Rs 104.")
    blocked = " Trace blocked: borrower has a live number and is avoiding." \
        if r.best_owned_live >= OWNED_LIVE_BLOCKS_TRACE and r.avoid_share >= .5 else ""
    call = ("no number can be called (all suppressed, paused or third party)" if not np.isfinite(r.ev_call)
            else f"best call nets Rs {r.ev_call:,.0f}")
    return f"Wait: nothing is worth its cost now; {call}, visit nets Rs {r.ev_visit:,.0f}.{blocked}"


# ---------------------------------------------------------------- scoring numbers at a date
def score_numbers(D, rows: pd.DataFrame, SF: pd.DataFrame, model) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every number at its best hour (agent), plus bot probability and the three factors at that hour.
    rows: snapshot rows at D. Returns (numbers, P(RPC) by hour)."""
    D = pd.Timestamp(D)
    n = rows.merge(SF[SF.cutoff == D].drop(columns="cutoff"), on="link", how="left")
    n = n.assign(dow=D.dayofweek, is_bot=False, lang_match=True).reset_index(drop=True)
    grid = model.by_hour(n, is_bot=False)
    n["best_hour"] = grid.idxmax(axis=1)
    n["p_rpc_agent"] = grid.max(axis=1)
    at_best = n.assign(hour=n.best_hour)
    n["p_rpc_bot"] = model.predict(at_best.assign(is_bot=True)).p_rpc.to_numpy()
    raw = model.raw(at_best)
    n["f_alive"], n["f_answer"], n["f_owned"] = raw.alive.to_numpy(), raw.answer.to_numpy(), raw.owned.to_numpy()
    return n, grid


# ---------------------------------------------------------------- dialler capacity
def assign_hours(plan: pd.DataFrame, by_hour: pd.DataFrame, capacity: dict) -> pd.DataFrame:
    """Greedy: highest-value calls first, each to its best hour that still has capacity."""
    left = dict(capacity)
    calls = plan[plan.action.eq("call")].sort_values("ev", ascending=False)
    slot = {}
    for acc_id in calls.account_id:
        ranked = by_hour.loc[acc_id].sort_values(ascending=False).index
        for h in ranked:
            if left.get(h, 0) > 0:
                left[h] -= 1
                slot[acc_id] = h
                break
    plan = plan.copy()
    plan["scheduled_hour"] = plan.account_id.map(slot)
    return plan


def schedule_visits(plan: pd.DataFrame, per_day: int) -> pd.DataFrame:
    """Field capacity: highest-value visits go today, the rest queue in value order."""
    plan = plan.copy()
    v = plan[plan.action.eq("visit")].sort_values("ev", ascending=False)
    plan["visit_queue_day"] = np.nan
    plan.loc[v.index, "visit_queue_day"] = np.arange(len(v)) // max(per_day, 1)
    return plan
