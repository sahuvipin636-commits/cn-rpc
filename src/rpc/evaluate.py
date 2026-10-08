"""Stage 7: one-time evaluation on the untouched TEST accounts.

This is the only module allowed to read test-split gold labels.
  eval_states   hidden-state model vs verified status (test gold), with baselines
  eval_safety   are risky numbers protected and borrower numbers kept? (test gold)
  eval_rpc      per-call and per-contact-point prediction quality (test calls)
  ope           RPC per 1,000 dials: new policy vs rule engine, estimated on the randomised
                arm with its logged propensities (self-normalised IPS, account bootstrap)
  trace_replay  re-decide every historical skip-trace with the new engine
  waste         dials the rule engine spent on numbers the new system would have stopped
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .config import GOLD_STATUSES
from .io import load_gold
from .states import StateModel, group_probs, OBS, VERIFY_T
from .safety import apply_safety
from .decide import CALLABLE, account_aggregates, decide, field_options, score_numbers

STOP = {"suppress_recycled", "drop_invalid", "pause_dead"}
NO_DEBT_TALK = STOP | {"locate_only", "verify_first"}


def test_gold() -> pd.DataFrame:
    return load_gold(("test",)).assign(link=lambda d: d.account_id + "|" + d.phone_id)


def _gold_metrics(df: pd.DataFrame) -> tuple[float, float]:
    P = np.clip(df[GOLD_STATUSES].to_numpy(float), 1e-6, 1)
    P = P / P.sum(axis=1, keepdims=True)
    y = df.verified_status.map({c: i for i, c in enumerate(GOLD_STATUSES)}).to_numpy()
    return log_loss(y, P, labels=list(range(len(GOLD_STATUSES)))), float((P.argmax(1) == y).mean())


# ------------------------------------------------------------------ states and safety
def eval_states(state_model: StateModel, obs: pd.DataFrame, links: pd.DataFrame) -> dict:
    gold = test_gold()
    sub = links[links.link.isin(gold.link)]
    seqs = StateModel.sequences(obs, sub)
    p = np.vstack([state_model.filter_at(s, [VERIFY_T])[0] for s in seqs])
    model = gold[["link", "verified_status"]].merge(group_probs(p).assign(link=[s["link"] for s in seqs]), on="link")

    tr = load_gold(("train",)).assign(link=lambda d: d.account_id + "|" + d.phone_id) \
        .merge(links[["link", "source"]], on="link")
    prior = tr.groupby("source").verified_status.value_counts(normalize=True).unstack(fill_value=0) \
        .reindex(columns=GOLD_STATUSES, fill_value=0) + 0.02
    prior = prior.div(prior.sum(axis=1), axis=0)
    base = gold.merge(links[["link", "source"]], on="link").join(prior, on="source")

    last = obs.groupby("link").code.last()
    rule = {OBS.index("does_not_exist"): "invalid_number", OBS.index("third_party"): "third_party_number",
            OBS.index("wrong_new_owner"): "not_borrower_number", OBS.index("wrong_other"): "not_borrower_number",
            OBS.index("switched_off"): "switched_off", OBS.index("switched_off_long"): "switched_off",
            OBS.index("not_reachable"): "switched_off"}
    rule_pred = gold.link.map(last).map(rule).fillna("borrower_number")
    pred = model[GOLD_STATUSES].idxmax(axis=1)
    return {
        "n": len(model),
        "table": pd.DataFrame({
            "state model": dict(zip(["log_loss", "accuracy"], _gold_metrics(model))),
            "source prior (train gold)": dict(zip(["log_loss", "accuracy"], _gold_metrics(base))),
            "last-call rule": {"log_loss": np.nan, "accuracy": float((rule_pred == gold.verified_status).mean())},
        }).T,
        "confusion": pd.crosstab(model.verified_status, pred).reindex(index=GOLD_STATUSES, columns=GOLD_STATUSES,
                                                                      fill_value=0),
    }


def eval_safety(numbers_plan: pd.DataFrame) -> dict:
    g = test_gold().merge(numbers_plan[["account_id", "phone_id", "status"]], on=["account_id", "phone_id"])
    risky = g.verified_status.isin(["not_borrower_number", "third_party_number"])
    borrower = g.verified_status.eq("borrower_number")
    return {"table": pd.crosstab(g.verified_status, g.status),
            "risky_protected": f"{int((risky & g.status.isin(NO_DEBT_TALK)).sum())}/{int(risky.sum())}",
            "borrower_blocked": f"{int((borrower & g.status.isin(STOP | {'locate_only'})).sum())}/{int(borrower.sum())}",
            "borrower_verify_first": int((borrower & g.status.eq('verify_first')).sum())}


# ------------------------------------------------------------------ prediction quality
def _xent(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def eval_rpc(model, A_train: pd.DataFrame, A_test: pd.DataFrame, snaps_test: pd.DataFrame,
             SF: pd.DataFrame) -> dict:
    y = A_test.y_rpc_soft.to_numpy()
    pr = model.predict(A_test)
    band = lambda h: pd.cut(h, [7, 9, 15, 19], labels=["morning", "midday", "evening"])
    rate = A_train.groupby([A_train.source, band(A_train.hour)], observed=True).y_rpc_soft.mean()
    base = pd.Series(list(zip(A_test.source, band(A_test.hour)))).map(rate).fillna(A_train.y_rpc_soft.mean()).to_numpy()
    calls = pd.DataFrame({
        "new model": {"log_loss": _xent(pr.p_rpc.to_numpy(), y), "auc": roc_auc_score(y >= .5, pr.p_rpc),
                      "mean_pred": pr.p_rpc.mean(), "mean_obs": y.mean()},
        "source x hour rate": {"log_loss": _xent(base, y), "auc": roc_auc_score(y >= .5, base),
                               "mean_pred": base.mean(), "mean_obs": y.mean()}}).T
    by_pf = pd.DataFrame({pf: {"calls": len(g), "observed": g.y_rpc_soft.mean(),
                               "predicted": pr.p_rpc.loc[g.index].mean()}
                          for pf, g in A_test.groupby("portfolio")}).T
    heads = {"alive": roc_auc_score(A_test.y_alive, pr.alive),
             "answer": roc_auc_score(A_test.y_answered[A_test.y_alive], pr.answer[A_test.y_alive]),
             "owned": roc_auc_score(A_test.y_rpc_soft[A_test.y_answered] >= .5, pr.owned[A_test.y_answered])}
    sv = snaps_test[snaps_test.next_att > 0].merge(SF, on=["cutoff", "link"], how="left") \
        .assign(hour=10, dow=1, is_bot=False, lang_match=True)
    sv["p"] = model.predict(sv).p_rpc.to_numpy()
    cp_auc = roc_auc_score(sv.next_rpc_any.astype(bool), sv.p)
    return {"calls": calls, "by_portfolio": by_pf, "heads": pd.Series(heads), "contact_point_auc": cp_auc,
            "contact_point_rows": len(sv)}


# ------------------------------------------------------------------ status over time
def status_history(snaps: pd.DataFrame, SF: pd.DataFrame, th: dict) -> pd.DataFrame:
    s = snaps.merge(SF, on=["cutoff", "link"], how="left")
    st = apply_safety(s, th)
    return pd.concat([s[["cutoff", "link", "account_id", "phone_id", "priority_slot", "no_rpc_streak"]], st], axis=1)


# ------------------------------------------------------------------ off-policy evaluation
def ope(model, A_eval: pd.DataFrame, snaps: pd.DataFrame, SF: pd.DataFrame, status: pd.DataFrame,
        n_boot: int = 1000, seed: int = 7) -> dict:
    """Randomised-arm calls only. For each logged call, which number would each policy have dialled
    at that moment? Weight = 1[policy choice == logged number] / logged propensity."""
    ra = A_eval[A_eval.dialling_arm.eq("random_contact_point")].reset_index(drop=True)
    cand = ra[["attempt_id", "cutoff", "account_id", "hour", "is_bot", "lang_match", "dow"]] \
        .merge(snaps.drop(columns=["dialling_arm"], errors="ignore"), on=["cutoff", "account_id"]) \
        .merge(SF, on=["cutoff", "link"], how="left") \
        .merge(status[["cutoff", "link", "status"]], on=["cutoff", "link"], how="left")
    cand["p"] = model.predict(cand).p_rpc.to_numpy()
    callable_ = cand.status.isin(CALLABLE)
    new = cand[callable_].sort_values("p", ascending=False).groupby("attempt_id").phone_id.first()
    live = cand[cand.no_rpc_streak < 15]
    rule = pd.concat([live, cand[~cand.attempt_id.isin(live.attempt_id)]]) \
        .sort_values(["priority_slot", "phone_id"]).groupby("attempt_id").phone_id.first()
    safe_rule = cand[callable_].sort_values(["priority_slot", "phone_id"]).groupby("attempt_id").phone_id.first()

    ra["w_new"] = (ra.attempt_id.map(new) == ra.phone_id) / ra.propensity
    ra["w_rule"] = (ra.attempt_id.map(rule) == ra.phone_id) / ra.propensity
    ra["w_safe_rule"] = (ra.attempt_id.map(safe_rule) == ra.phone_id) / ra.propensity
    ra["w_random"] = 1.0
    ra["stranger"] = (ra.y_wrong | ra.y_tp).astype(float)
    # unprotected = a stranger answers and nothing told the agent to verify identity first.
    # The rule engine has no such flag, so every stranger it reaches is unprotected.
    st = cand[["attempt_id", "phone_id", "status"]].drop_duplicates(["attempt_id", "phone_id"])
    ra = ra.merge(st, on=["attempt_id", "phone_id"], how="left")
    ra["unprotected_new"] = ra.stranger * ra.status.eq("dial")
    ra["unprotected_rule"] = ra.stranger

    def snips(df, w, col):
        return float((df[w] * df[col]).sum() / df[w].sum()) if df[w].sum() > 0 else np.nan

    pols = {"new policy": "w_new", "rule engine (approx.)": "w_rule", "rule engine + safety layer": "w_safe_rule",
            "random (as logged)": "w_random"}
    rng = np.random.default_rng(seed)
    accts = ra.account_id.unique()
    groups = {a: g for a, g in ra.groupby("account_id")}
    boots = {k: [] for k in pols}
    for _ in range(n_boot):
        bs = pd.concat([groups[a] for a in rng.choice(accts, len(accts))])
        for k, w in pols.items():
            boots[k].append(snips(bs, w, "y_rpc_soft"))
    table = pd.DataFrame({k: {"rpc_per_1000_dials": 1000 * snips(ra, w, "y_rpc_soft"),
                              "ci_low": 1000 * np.nanpercentile(boots[k], 5),
                              "ci_high": 1000 * np.nanpercentile(boots[k], 95),
                              "stranger_answers_per_1000": 1000 * snips(ra, w, "stranger"),
                              "unprotected_stranger_per_1000": 1000 * snips(
                                  ra, w, "unprotected_rule" if w in ("w_rule", "w_random") else "unprotected_new"),
                              "matched_calls": int((ra[w] > 0).sum())} for k, w in pols.items()}).T
    diff = np.array(boots["new policy"]) - np.array(boots["rule engine (approx.)"])
    return {"table": table, "calls": len(ra), "accounts": len(accts),
            "p_new_better": float(np.nanmean(diff > 0)),
            "lift_ci": (1000 * np.nanpercentile(diff, 5), 1000 * np.nanpercentile(diff, 95))}


# ------------------------------------------------------------------ skip-trace replay
def trace_replay(t: dict, snaps: pd.DataFrame, SF: pd.DataFrame, model, cfg: dict,
                 field_remarks: pd.DataFrame, cutoffs, E: pd.DataFrame | None = None) -> pd.DataFrame:
    from .labels import assign_cutoff
    tr = t["skip_traces"].merge(t["splits"], on="account_id")
    tr["cutoff"] = assign_cutoff(tr.trace_date, cutoffs)
    pay = t["payments"]
    out = []
    for c, grp in tr.groupby("cutoff"):
        rows = snaps[(snaps.cutoff == c) & snaps.account_id.isin(grp.account_id)]
        n, _ = score_numbers(c, rows.drop(columns=[x for x in rows.columns if x.startswith("next_")]), SF, model)
        n = n.join(apply_safety(n, cfg["thresholds"]))
        fields = field_options(c, t, field_remarks, cfg["params"])
        plan, _ = decide(n, t["accounts"][t["accounts"].account_id.isin(grp.account_id)], cfg["params"],
                         cfg["trace_model"], fields)
        out.append(grp.merge(plan[["account_id", "action", "p_find", "ev_trace", "best_owned_live", "avoid_share"]],
                             on="account_id", how="left"))
    r = pd.concat(out, ignore_index=True)
    r["found_phone"] = r.result.eq("new_phone_found")
    r["found_any"] = r.result.ne("no_new_info")
    pm = r[["trace_id", "account_id", "trace_date"]].merge(pay[["account_id", "payment_ts", "amount"]],
                                                           on="account_id", how="left")
    pm["in30"] = (pm.payment_ts > pm.trace_date) & (pm.payment_ts <= pm.trace_date + pd.Timedelta(days=30))
    r = r.merge(pm.assign(amt=pm.amount.where(pm.in30, 0)).groupby("trace_id")
                .agg(paid30=("in30", "any"), paid30_amount=("amt", "sum")), on="trace_id")
    r["we_trace"] = r.action.eq("trace")
    if E is not None:   # borrower reached within 30 days on a number that existed before the trace
        old = E[E.rpc_trusted][["account_id", "phone_id", "attempt_ts"]].merge(
            r[["trace_id", "account_id", "trace_date", "new_contact_point_id"]], on="account_id")
        old = old[(old.attempt_ts > old.trace_date) & (old.attempt_ts <= old.trace_date + pd.Timedelta(days=30))
                  & (old.phone_id != old.new_contact_point_id)]
        r["reached_on_existing_30d"] = r.trace_id.isin(old.trace_id)
    return r


# ------------------------------------------------------------------ wasted dials
def waste(A: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    a = A.merge(status[["cutoff", "link", "status"]], on=["cutoff", "link"], how="left")
    g = a.groupby("status")
    return pd.DataFrame({"dials": g.size(), "share_of_dials": g.size() / len(a),
                         "borrower_reached_rate": g.y_rpc_soft.mean(),
                         "stranger_or_relative_answered": g.apply(lambda x: int((x.y_wrong | x.y_tp).sum()),
                                                                  include_groups=False)}) \
        .sort_values("dials", ascending=False)
