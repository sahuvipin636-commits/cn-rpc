"""Stage 6: safety rules + per-account decisions with reason codes, as of 30 June.
Run from the repo root (after Stage 5):  python scripts/06_decide.py
Writes data/processed/{numbers_plan, account_plan}.parquet, reports/{06_decide.md,
suppression_list.csv, trace_queue.csv, call_plan.csv}
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from rpc.config import PROCESSED, REPORTS, GOLD_STATUSES
from rpc.io import load_all, load_gold
from rpc.labels import weekly_cutoffs, assign_cutoff, SCORING_CUTOFF
from rpc.models import RPCModel
from rpc.safety import apply_safety, tune_threshold, tune_precision, DEFAULTS
from rpc.decide import (fit_value_params, account_aggregates, TraceModel, field_options, decide, assign_hours,
                         schedule_visits)

pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 200)
out = []


def section(title, body):
    text = body if isinstance(body, str) else body.to_string()
    out.append(f"## {title}\n\n```\n{text}\n```\n")
    print(f"\n=== {title}\n{text}")


def numbers_at(D, rows, SF, model):
    """Score every number at its best hour (agent), plus bot probability and factor heads at that hour."""
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


t0 = time.time()
t = load_all()
trust = pd.read_parquet(PROCESSED / "attempt_trust.parquet")
SF = pd.read_parquet(PROCESSED / "state_features.parquet")
snaps = pd.read_parquet(PROCESSED / "snapshots.parquet")
scoring = pd.read_parquet(PROCESSED / "scoring_snapshot.parquet")
rf = pd.read_parquet(PROCESSED / "remarks_field.parquet")
model = RPCModel.load(PROCESSED / "rpc_model.pkl")
train_ids = set(t["splits"].query("split == 'train'").account_id)

# 1. value parameters (train accounts)
params = fit_value_params(t, trust, train_ids)
section("1. Value parameters (train accounts)", "\n".join(f"{k}: {v}" for k, v in params.items()))

# 2. safety thresholds on train gold, checked on validation gold
D = SCORING_CUTOFF
s30 = scoring.merge(SF[SF.cutoff == D].drop(columns="cutoff"), on="link", how="left")
g_tr = load_gold(("train",)).merge(s30, on=["account_id", "phone_id"])
g_va = load_gold(("validation",)).merge(s30, on=["account_id", "phone_id"])
th = dict(DEFAULTS)
for key, status in [("recycled", "not_borrower_number"), ("third_party", "third_party_number")]:
    p, pos = g_tr[f"p_{key}"].to_numpy(), (g_tr.verified_status == status).to_numpy()
    th[f"{key}_low"] = tune_threshold(p, pos)
    th[f"{key}_high"] = max(tune_precision(p, pos), th[f"{key}_low"])
section("2. Safety thresholds (train gold: low tier = 90% recall -> verify first, "
        "high tier = 90% precision -> suppress)", pd.Series(th).round(3))
for name, g in [("train (tuned here)", g_tr), ("validation", g_va)]:
    g = g.join(apply_safety(g, th))
    section(f"3. Safety status vs verified status, {name}",
            pd.crosstab(g.verified_status, g.status).reindex(index=GOLD_STATUSES).fillna(0).astype(int))

# 3. trace model: account state at the time of each historical trace
cutoffs = weekly_cutoffs()
tr = t["skip_traces"].merge(t["splits"], on="account_id")
tr["cutoff"] = assign_cutoff(tr.trace_date, cutoffs)
X = []
for c, grp in tr.groupby("cutoff"):
    rows = snaps[(snaps.cutoff == c) & snaps.account_id.isin(grp.account_id)]
    n = rows.merge(SF[SF.cutoff == c].drop(columns="cutoff"), on="link", how="left")
    n = n.join(apply_safety(n, th))
    X.append(grp.join(account_aggregates(n), on="account_id"))
X = pd.concat(X)
X["y"] = X.result.eq("new_phone_found")
tm = TraceModel().fit(X[X.split == "train"], X[X.split == "train"].y)
xv = X[X.split == "validation"]
section("4. Trace model (finds a new phone)", "\n".join([
    f"train traces {int((X.split == 'train').sum())}, validation traces {len(xv)}",
    f"validation AUC {roc_auc_score(xv.y, tm.predict(xv)):.3f}",
    "coefficients (standardised):", tm.coefficients().round(2).to_string()]))

# 4. the plan for 30 June
n, grid = numbers_at(D, scoring, SF, model)
n = n.join(apply_safety(n, th))
fields = field_options(D, t, rf, params)
plan, nplan = decide(n, t["accounts"], params, tm, fields)

calls = t["dial_attempts"]
days = calls.attempt_ts.dt.normalize().nunique()
capacity = (calls.attempt_ts.dt.hour.value_counts() / days).round().astype(int).to_dict()
chosen = plan[plan.action == "call"][["account_id", "phone_id"]]
bh = grid.assign(account_id=n.account_id, phone_id=n.phone_id).merge(chosen).set_index("account_id")
plan = assign_hours(plan, bh[list(grid.columns)], capacity)
visits_per_day = int(round(t["field_visits"].groupby("visit_date").size().mean()))
plan = schedule_visits(plan, visits_per_day)

section("5. Number status on 30 June", n.status.value_counts())
section("6. Account actions on 30 June", plan.groupby("action").agg(
    accounts=("account_id", "size"), total_ev_rs=("ev", "sum")).round(0))
sched = pd.DataFrame({"capacity": pd.Series(capacity), "scheduled": plan.scheduled_hour.value_counts(),
                      "wanted_as_best_hour": plan[plan.action == "call"].best_hour.value_counts()})
section("7. Today's calls vs dialler capacity, by hour", sched.fillna(0).astype(int).sort_index().to_string() +
        f"\ncalls today {int(plan.scheduled_hour.notna().sum())} of {int((plan.action == 'call').sum())} wanted; "
        f"field: {visits_per_day} visits/day, {int((plan.action == 'visit').sum())} queued visits = "
        f"{plan.visit_queue_day.max() + 1:.0f} days of field work")
ex = plan.groupby("action").head(2)[["account_id", "action", "reason"]]
section("8. Example reason codes", "\n\n".join(f"{r.account_id} [{r.action}] {r.reason}" for r in ex.itertuples()))
blocked = plan[(plan.best_owned_live >= .3) & (plan.avoid_share >= .5)]
section("9. Avoiders: traces prevented", f"{len(blocked)} accounts hold a live borrower number but avoid calls; "
        f"actions: {blocked.action.value_counts().to_dict()}")

nplan.to_parquet(PROCESSED / "numbers_plan.parquet", index=False)
plan.to_parquet(PROCESSED / "account_plan.parquet", index=False)
REPORTS.mkdir(exist_ok=True)
n[n.status.isin(["suppress_recycled", "locate_only"])][["account_id", "phone_id", "status", "safety_reason"]] \
    .to_csv(REPORTS / "suppression_list.csv", index=False)
plan[plan.action == "trace"].sort_values("ev", ascending=False)[["account_id", "ev", "p_find", "reason"]] \
    .to_csv(REPORTS / "trace_queue.csv", index=False)
plan[plan.action == "call"].sort_values("ev", ascending=False)[
    ["account_id", "phone_id", "call_channel", "scheduled_hour", "ev", "reason"]].to_csv(REPORTS / "call_plan.csv", index=False)
section("10. Run time", f"{time.time() - t0:.0f}s")
(REPORTS / "06_decide.md").write_text("# Stage 6: decisions\n\n" + "\n".join(out), encoding="utf-8")
