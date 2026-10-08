"""Stage 1: validate the data and reproduce the findings that shape the design.
Run from the repo root:  python scripts/01_audit.py
Writes reports/01_audit.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from rpc.config import REPORTS
from rpc.io import load_all, load_gold, add_flags, validate

pd.set_option("display.width", 200)
out: list[str] = []


def section(title: str, body) -> None:
    text = body if isinstance(body, str) else body.to_string()
    out.append(f"## {title}\n\n```\n{text}\n```\n")
    print(f"\n=== {title}\n{text}")


t = load_all()
att = add_flags(t["dial_attempts"])
ph, acc, sp = t["phones"], t["accounts"], t["splits"]
gold = load_gold(splits=("train",))                      # train only, by rule

# 1. Validation
issues = validate(t)
section("1. Validation", "clean" if not issues else "\n".join(issues))

# 2. Shape
section("2. Shape", "\n".join([
    f"accounts {len(acc):,} | phone links {len(ph):,} | attempts {len(att):,} | visits {len(t['field_visits']):,}"
    f" | traces {len(t['skip_traces']):,} | payments {len(t['payments']):,}",
    f"call window {att.attempt_ts.min():%Y-%m-%d} to {att.attempt_ts.max():%Y-%m-%d}",
    f"splits {sp.split.value_counts().to_dict()}",
    f"gold labels available to training (train split): {len(gold)}"]))

# 3. Coverage: what the rule engine never tries
n_att = att.groupby(["account_id", "phone_id"]).size().rename("n")
cov = ph.merge(n_att.reset_index(), how="left").fillna({"n": 0})
section("3. Share of numbers never dialled, by source",
        cov.assign(never=cov.n.eq(0)).groupby("source").never.mean().round(2))

# 4. What each verified state looks like to the dialler (train gold)
g = att.merge(gold, on=["account_id", "phone_id"])
section("4. Network response by verified status, row % (train gold)",
        (pd.crosstab(g.verified_status, g.network_response, normalize="index") * 100).round(0))

# 5. Label noise and agents
tele = att[att.agent_id.str.startswith(("TA", "BOT"))]
nonb = g[g.answered & g.verified_status.ne("borrower_number")]
agent = pd.DataFrame({
    "attempts": tele.groupby("agent_id").size(),
    "rpc_rate": tele.groupby("agent_id").logged_rpc.mean().round(3),
    "third_party_rate": tele.groupby("agent_id").logged_tp.mean().round(3),
    "rpc_on_unanswered": tele.groupby("agent_id").impossible_rpc.sum(),
    "nonborrower_answered": nonb.groupby("agent_id").size(),
    "logged_as_rpc": nonb.groupby("agent_id").logged_rpc.sum(),
}).fillna(0)
agent["rpc_share_on_nonborrower"] = (agent.logged_as_rpc / agent.nonborrower_answered.replace(0, np.nan)).round(2)
section("5. Agent logging quality (sorted by RPC share on verified non-borrower numbers)",
        agent.sort_values("rpc_share_on_nonborrower", ascending=False))

# 6. When borrowers answer
own = att.merge(ph[["account_id", "phone_id", "relation_recorded"]]).query("relation_recorded == 'self'")
own = own.merge(acc[["account_id", "income_type"]])
section("6. Logged RPC % by hour and income type (self numbers)",
        (own.pivot_table(index="income_type", columns="hour", values="logged_rpc", aggfunc="mean") * 100).round(0))
out.append(f"Share of all dials before 13:00: {(att.hour < 13).mean():.0%}\n")
print(f"Share of all dials before 13:00: {(att.hour < 13).mean():.0%}")

# 7. Skip-traces by why the account was failing before the trace
st = t["skip_traces"]
att_by_acc = {a: d for a, d in att.groupby("account_id")}
rows = []
for r in st.itertuples():
    h = att_by_acc.get(r.account_id)
    if h is None:
        continue
    h = h[h.attempt_ts < r.trace_date].tail(15)
    if h.empty:
        continue
    pattern = ("mostly_dead" if h.dead_resp.mean() >= .6 else
               "mostly_rejecting" if h.rejected.mean() >= .3 else
               "mostly_third_party" if h.logged_tp.mean() >= .3 else "mostly_no_answer")
    rows.append({"pattern": pattern, "hit": r.result != "no_new_info",
                 "new_phone": r.result == "new_phone_found", "cost": r.cost_inr})
tr = pd.DataFrame(rows)
section("7. Skip-trace results by pre-trace failure pattern",
        tr.groupby("pattern").agg(traces=("hit", "size"), hit_rate=("hit", "mean"),
                                  new_phone_rate=("new_phone", "mean"), spend_inr=("cost", "sum")).round(2))

# 8. Value of an RPC
pay = t["payments"]
r = att.loc[att.logged_rpc, ["account_id", "attempt_ts"]].merge(pay[["account_id", "payment_ts"]], on="account_id")
hit = r[(r.payment_ts > r.attempt_ts) & (r.payment_ts <= r.attempt_ts + pd.Timedelta(days=7))]
share = hit.drop_duplicates(["account_id", "attempt_ts"]).shape[0] / att.logged_rpc.sum()
section("8. Value of an RPC", f"logged RPCs followed by a payment within 7 days: {share:.1%}")

# 9. Shared numbers, exploration arm, field outcomes
shared = ph.groupby("phone_id").account_id.nunique()
s = ph[ph.phone_id.isin(shared[shared > 1].index)]
section("9. Shared numbers", s.groupby(["source", "relation_recorded"]).phone_id.nunique())
arm = att.groupby("dialling_arm").agg(attempts=("attempt_id", "size"), accounts=("account_id", "nunique"),
                                      rpc_rate=("logged_rpc", "mean"),
                                      min_propensity=("selection_propensity", "min")).round(3)
section("10. Dialling arms", arm)
section("11. Field visit outcomes", t["field_visits"].outcome.value_counts())

REPORTS.mkdir(exist_ok=True)
(REPORTS / "01_audit.md").write_text("# Stage 1 data audit\n\n" + "\n".join(out), encoding="utf-8")
print(f"\nreport written to {REPORTS / '01_audit.md'}")
