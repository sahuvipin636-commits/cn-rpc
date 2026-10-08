"""RPC console for collections supervisors.
Run from the repo root (after Stages 6 and 7):  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rpc.config import PROCESSED, REPORTS, RAW  # noqa: E402

st.set_page_config(page_title="RPC Console", layout="wide")

STATUS_LABEL = {"dial": "Dial", "verify_first": "Verify ID", "locate_only": "Locate only",
                "suppress_recycled": "Suppressed", "pause_dead": "Paused", "drop_invalid": "Invalid"}
STATUS_NOTE = {"Dial": "Call about the debt", "Verify ID": "Confirm identity before discussing the debt",
               "Locate only": "Relative or colleague; message only", "Suppressed": "Likely new owner; do not call",
               "Paused": "Dead for a long time; re-check later", "Invalid": "Number does not exist"}
ACTION_LABEL = {"call": "Call", "visit": "Visit", "trace": "Trace", "wait": "Hold"}


def reason_code(r) -> str:
    if r.action == "call":
        if r.status == "verify_first":
            return "Call: verify ID first"
        return "Call: voice bot" if r.call_channel == "voice_bot" else "Call: best hour"
    if r.action == "visit":
        return "Visit: avoiding calls" if (r.avoid_share >= .5 and r.any_owned_live >= .3) else "Visit: no live number"
    if r.action == "trace":
        if r.mean_dead >= .5:
            return "Trace: numbers dead"
        return "Trace: third party only" if r.mean_third_party >= .5 else "Trace: no borrower number"
    return "Hold: no callable number" if not np.isfinite(r.ev_call) else "Hold: below cost"


@st.cache_data
def load():
    if not (PROCESSED / "account_plan.parquet").exists():
        return None
    acc = pd.read_csv(RAW / "accounts.csv")
    plan = pd.read_parquet(PROCESSED / "account_plan.parquet").merge(
        acc[["account_id", "lender_id", "portfolio", "bucket_start", "overdue_start", "outstanding"]],
        on="account_id", how="left")
    plan["action_label"] = plan.action.map(ACTION_LABEL)
    plan["reason_code"] = [reason_code(r) for r in plan.itertuples()]
    nums = pd.read_parquet(PROCESSED / "numbers_plan.parquet")
    nums["status_label"] = nums.status.map(STATUS_LABEL)
    calls = pd.read_csv(RAW / "dial_attempts.csv", parse_dates=["attempt_ts"])
    replay = pd.read_csv(REPORTS / "trace_replay.csv") if (REPORTS / "trace_replay.csv").exists() else None
    return plan, nums, acc, calls, replay


def pct(s):
    return (s * 100).round(1)


data = load()
if data is None:
    st.error("Plan files not found. Run `python scripts/06_decide.py` from the repo root.")
    st.stop()
plan, nums, acc, calls, replay = data

# ------------------------------------------------------------------ sidebar
st.sidebar.title("RPC Console")
st.sidebar.caption("Plan date: 30 Jun 2026  |  Data to 29 Jun")
page = st.sidebar.radio("View", ["Work queue", "Account lookup", "Contact points", "Skip-trace", "Model performance"])
st.sidebar.divider()
f_pf = st.sidebar.multiselect("Portfolio", sorted(plan.portfolio.dropna().unique()))
f_bk = st.sidebar.multiselect("Bucket", ["X", "1-30", "31-60", "61-90", "90+"])
f_ln = st.sidebar.multiselect("Lender", sorted(plan.lender_id.dropna().unique()))
view = plan.copy()
for col, sel in [("portfolio", f_pf), ("bucket_start", f_bk), ("lender_id", f_ln)]:
    if sel:
        view = view[view[col].isin(sel)]

# ------------------------------------------------------------------ work queue
if page == "Work queue":
    st.header("Work queue")
    c = st.columns(5)
    c[0].metric("Accounts", f"{len(view):,}")
    c[1].metric("Calls planned", f"{(view.action == 'call').sum():,}")
    c[2].metric("Calls scheduled today", f"{view.scheduled_hour.notna().sum():,}")
    c[3].metric("Visits planned", f"{(view.action == 'visit').sum():,}")
    c[4].metric("Traces", f"{(view.action == 'trace').sum():,}")

    acts = st.multiselect("Action", ["Call", "Visit", "Trace", "Hold"], default=["Call", "Visit", "Trace"])
    q = view[view.action_label.isin(acts)].sort_values("ev", ascending=False)
    table = pd.DataFrame({
        "Account": q.account_id, "Portfolio": q.portfolio, "Bucket": q.bucket_start, "Action": q.action_label,
        "Channel": q.call_channel.where(q.action == "call").map({"tele_agent": "Agent", "voice_bot": "Bot"}),
        "Number": q.phone_id.where(q.action == "call"), "Slot": q.scheduled_hour,
        "Exp. value (Rs)": q.ev.round(0), "Reason code": q.reason_code})
    st.dataframe(table, hide_index=True, width="stretch", height=460,
                 column_config={"Slot": st.column_config.NumberColumn(format="%d:00"),
                                "Exp. value (Rs)": st.column_config.NumberColumn(format="localized")})
    st.caption("Slot blank = queued (hour at capacity). Exp. value = expected recovery minus action cost.")

    left, right = st.columns(2)
    with left:
        st.subheader("Actions by portfolio")
        st.bar_chart(view.pivot_table(index="portfolio", columns="action_label", values="account_id",
                                      aggfunc="count", fill_value=0), height=300)
    with right:
        st.subheader("Calls scheduled by hour")
        hrs = view.scheduled_hour.dropna().astype(int).value_counts().sort_index()
        hrs.index = [f"{h:02d}:00" for h in hrs.index]
        st.bar_chart(hrs, height=300)

# ------------------------------------------------------------------ account lookup
elif page == "Account lookup":
    st.header("Account lookup")
    ids = plan.sort_values("ev", ascending=False).account_id.tolist()
    aid = st.selectbox("Account ID", ids)
    a = plan[plan.account_id == aid].iloc[0]
    c = st.columns(5)
    c[0].metric("Portfolio", str(a.portfolio).replace("_", " "))
    c[1].metric("Bucket", a.bucket_start)
    c[2].metric("Overdue (Rs)", f"{a.overdue_start:,.0f}")
    c[3].metric("Outstanding (Rs)", f"{a.outstanding:,.0f}")
    c[4].metric("Action", a.action_label)
    st.markdown(f"**{a.reason_code}.** {a.reason}")

    st.subheader("Contact points")
    n = nums[nums.account_id == aid]
    callable_ = n.status.isin(["dial", "verify_first"])
    st.dataframe(pd.DataFrame({
        "Number": n.phone_id, "Source": n.source, "Relation": n.relation_recorded, "Status": n.status_label,
        "P(contact) %": pct(n.p_rpc_agent).where(callable_), "Best hour": n.best_hour.where(callable_),
        "P(alive) %": pct(n.p_alive), "P(owned) %": pct(n.p_owned_given_alive),
        "P(avoiding) %": pct(n.p_avoid_given_owned), "Note": n.safety_reason}),
        hide_index=True, width="stretch",
        column_config={"Best hour": st.column_config.NumberColumn(format="%d:00")})

    st.subheader("Call history (last 25)")
    h = calls[calls.account_id == aid].sort_values("attempt_ts", ascending=False).head(25)
    st.dataframe(h.rename(columns={"attempt_ts": "Time", "phone_id": "Number", "agent_id": "Agent",
                                   "network_response": "Network", "disposition": "Disposition", "remark": "Remark"})
                 [["Time", "Number", "Agent", "Network", "Disposition", "Remark"]],
                 hide_index=True, width="stretch")

# ------------------------------------------------------------------ contact points
elif page == "Contact points":
    st.header("Contact points")
    counts = nums.status_label.value_counts().reindex(list(STATUS_NOTE)).fillna(0).astype(int)
    st.dataframe(pd.DataFrame({"Status": counts.index, "Numbers": counts.values,
                               "Handling": [STATUS_NOTE[s] for s in counts.index]}),
                 hide_index=True, width="content")
    sel = st.multiselect("Status", list(STATUS_NOTE), default=["Suppressed", "Locate only", "Verify ID"])
    lst = nums[nums.status_label.isin(sel)]
    out = pd.DataFrame({"Account": lst.account_id, "Number": lst.phone_id, "Source": lst.source,
                        "Status": lst.status_label, "Note": lst.safety_reason})
    st.dataframe(out, hide_index=True, width="stretch", height=420)
    st.download_button("Download CSV", out.to_csv(index=False), "contact_point_status.csv", "text/csv")

# ------------------------------------------------------------------ skip-trace
elif page == "Skip-trace":
    st.header("Skip-trace queue")
    q = view[view.action == "trace"].sort_values("ev", ascending=False)
    st.dataframe(pd.DataFrame({"Account": q.account_id, "Portfolio": q.portfolio, "Bucket": q.bucket_start,
                               "P(find) %": pct(q.p_find), "Exp. value (Rs)": q.ev.round(0),
                               "Reason code": q.reason_code}),
                 hide_index=True, width="stretch",
                 column_config={"Exp. value (Rs)": st.column_config.NumberColumn(format="localized")})
    st.caption("Traces are not raised while the borrower holds a live number.")
    if replay is not None:
        st.subheader("Backtest: historical traces (validation + test accounts)")
        r = replay[replay.split.isin(["validation", "test"])]
        groups = {"15-failure rule (actual)": r, "Engine: trace": r[r.we_trace], "Engine: skip": r[~r.we_trace]}
        st.dataframe(pd.DataFrame({
            "Traces": {k: len(g) for k, g in groups.items()},
            "Spend (Rs)": {k: int(g.cost_inr.sum()) for k, g in groups.items()},
            "Phone found %": {k: round(100 * g.found_phone.mean(), 1) for k, g in groups.items()},
            "Reached on existing number in 30d %": {k: round(100 * g.reached_on_existing_30d.mean(), 1)
                                                    for k, g in groups.items()},
            "Paid in 30d %": {k: round(100 * g.paid30.mean(), 1) for k, g in groups.items()},
        }), width="stretch")

# ------------------------------------------------------------------ model performance
else:
    st.header("Model performance (test split)")
    sc = REPORTS / "scorecard.md"
    if sc.exists():
        rows = [ln.strip("|").split("|") for ln in sc.read_text(encoding="utf-8").splitlines()
                if ln.startswith("|") and not ln.startswith("|---") and "Metric" not in ln]
        st.dataframe(pd.DataFrame([[c.strip() for c in r] for r in rows], columns=["Metric", "Result"]),
                     hide_index=True, width="stretch")
    else:
        st.info("Run `python scripts/07_evaluate.py` to generate the scorecard.")
    st.caption("Contact-rate lift from 34 randomised accounts; wide interval, needs a live trial. "
               "Dialling-rule baseline is approximated from the data.")
