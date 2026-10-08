"""Right-party contact console for collections supervisors.
Run from the repo root (after Stages 6 and 7):  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rpc.config import PROCESSED, REPORTS, RAW, SCORING_DATE_LABEL  # noqa: E402

st.set_page_config(page_title="Right-party contact console", layout="wide")

STATUS = {   # one colour per status, used everywhere
    "dial": ("#2E7D5B", "Call about the debt"),
    "verify_first": ("#B7791F", "Callable, but confirm identity before mentioning the debt"),
    "locate_only": ("#7B5EA7", "Reaches a relative or colleague: ask them to pass a message, never the debt"),
    "suppress_recycled": ("#A23B2A", "Probably a new owner: never call"),
    "pause_dead": ("#7A8591", "Off or unreachable for a long time: stop dialling, re-check later"),
    "drop_invalid": ("#7A8591", "Number does not exist"),
}
ACTION_LABEL = {"call": "Call", "visit": "Field visit", "trace": "Skip-trace", "wait": "Wait"}

st.markdown("""
<style>
.reason {font-size: 1.05rem; line-height: 1.55; max-width: 72ch; padding: .9rem 1.1rem;
         border-left: 4px solid #1F5F8B; background: #F3F5F7; margin: .4rem 0 1rem 0;}
.chip {display: inline-block; padding: .1rem .55rem; border-radius: .3rem; color: #fff; font-size: .85rem;}
.legend div {margin: .25rem 0;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    need = [PROCESSED / "account_plan.parquet", PROCESSED / "numbers_plan.parquet"]
    if not all(p.exists() for p in need):
        return None
    plan = pd.read_parquet(need[0])
    nums = pd.read_parquet(need[1])
    acc = pd.read_csv(RAW / "accounts.csv")
    calls = pd.read_csv(RAW / "dial_attempts.csv", parse_dates=["attempt_ts"])
    replay = pd.read_csv(REPORTS / "trace_replay.csv") if (REPORTS / "trace_replay.csv").exists() else None
    plan = plan.merge(acc[["account_id", "portfolio", "bucket_start", "outstanding", "overdue_start"]],
                      on="account_id", how="left")
    return plan, nums, acc, calls, replay


def chip(status: str) -> str:
    colour = STATUS.get(status, ("#7A8591", ""))[0]
    return f'<span class="chip" style="background:{colour}">{status.replace("_", " ")}</span>'


def colour_status(v):
    return f"color: {STATUS[v][0]}; font-weight: 600" if v in STATUS else ""


data = load()
if data is None:
    st.error("No plan found. Run `python scripts/06_decide.py` from the repo root, then reload this page.")
    st.stop()
plan, nums, acc, calls, replay = data

st.title(f"Collections plan for {SCORING_DATE_LABEL}")
st.caption("Built from calls up to 29 June. Each account gets the action with the highest expected recovery "
           "net of cost, after compliance rules have removed numbers that must not be called.")

tab_plan, tab_acc, tab_nums, tab_trace, tab_res = st.tabs(
    ["Today's plan", "Account detail", "Numbers and safety", "Skip-traces", "Results"])

# ------------------------------------------------------------------ today's plan
with tab_plan:
    calls_wanted = int((plan.action == "call").sum())
    calls_today = int(plan.scheduled_hour.notna().sum())
    visits = plan[plan.action == "visit"]
    held = int(nums.status.isin(["suppress_recycled", "locate_only", "pause_dead", "drop_invalid"]).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calls scheduled today", f"{calls_today:,}", f"{calls_wanted - calls_today:,} queued", delta_color="off")
    c2.metric("Field visits", f"{int((visits.visit_queue_day == 0).sum()):,} today",
              f"{len(visits):,} in queue", delta_color="off")
    c3.metric("Skip-traces", f"{int((plan.action == 'trace').sum()):,}")
    c4.metric("Numbers held back", f"{held:,}", "see Numbers and safety", delta_color="off")

    f1, f2 = st.columns([2, 2])
    acts = f1.multiselect("Action", list(ACTION_LABEL), default=["call", "visit", "trace"],
                          format_func=ACTION_LABEL.get)
    pfs = f2.multiselect("Portfolio", sorted(plan.portfolio.dropna().unique()))
    view = plan[plan.action.isin(acts)]
    if pfs:
        view = view[view.portfolio.isin(pfs)]
    view = view.sort_values("ev", ascending=False)
    st.dataframe(
        view[["account_id", "action", "call_channel", "scheduled_hour", "phone_id", "ev", "reason"]]
        .rename(columns={"account_id": "Account", "action": "Action", "call_channel": "Channel",
                         "scheduled_hour": "Hour today", "phone_id": "Number", "ev": "Net value (Rs)",
                         "reason": "Why"}),
        hide_index=True, width="stretch", height=420,
        column_config={"Net value (Rs)": st.column_config.NumberColumn(format="%.0f"),
                       "Hour today": st.column_config.NumberColumn(format="%d:00"),
                       "Why": st.column_config.TextColumn(width="large")})

    st.subheader("When borrowers answer vs when we can call")
    cap = pd.read_csv(REPORTS / "call_plan.csv") if (REPORTS / "call_plan.csv").exists() else None
    hours = pd.DataFrame({
        "Best hour for the account": plan[plan.action == "call"].best_hour.value_counts(),
        "Calls scheduled": plan.scheduled_hour.value_counts()}).fillna(0).sort_index()
    hours.index = [f"{int(h):02d}:00" for h in hours.index]
    st.bar_chart(hours, stack=False, height=260)
    st.caption("Most borrowers answer best at 08:00 or 18:00, but dialler capacity is fixed by shift. "
               "Moving agent hours toward early morning and early evening would let more calls land at the best hour.")

# ------------------------------------------------------------------ account detail
with tab_acc:
    ids = plan.sort_values("ev", ascending=False).account_id.tolist()
    aid = st.selectbox("Account", ids, index=0)
    a = plan[plan.account_id == aid].iloc[0]
    info = acc[acc.account_id == aid].iloc[0]
    st.markdown(f"**{aid}**: {info.portfolio.replace('_', ' ')}, bucket {info.bucket_start}, "
                f"overdue Rs {info.overdue_start:,.0f}, outstanding Rs {info.outstanding:,.0f}")
    st.markdown(f'<div class="reason"><b>{ACTION_LABEL[a.action]}.</b> {a.reason}</div>', unsafe_allow_html=True)

    n = nums[nums.account_id == aid].copy()
    n["chance_borrower"] = n.p_rpc_agent.where(n.status.isin(["dial", "verify_first"]))
    st.dataframe(
        n[["phone_id", "source", "relation_recorded", "status", "chance_borrower", "best_hour",
           "p_alive", "p_owned_given_alive", "p_avoid_given_owned", "safety_reason"]]
        .rename(columns={"phone_id": "Number", "source": "Source", "relation_recorded": "Recorded as",
                         "status": "Status", "chance_borrower": "Chance of borrower (best hour)",
                         "best_hour": "Best hour", "p_alive": "Line works",
                         "p_owned_given_alive": "Belongs to borrower", "p_avoid_given_owned": "Borrower avoiding",
                         "safety_reason": "Safety note"})
        .style.map(colour_status, subset=["Status"]),
        hide_index=True, width="stretch",
        column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="%.0f%%")
                       for c in ["Chance of borrower (best hour)", "Line works", "Belongs to borrower",
                                 "Borrower avoiding"]}
        | {"Best hour": st.column_config.NumberColumn(format="%d:00")})

    st.subheader("Call history")
    h = calls[calls.account_id == aid].sort_values("attempt_ts", ascending=False).head(25)
    st.dataframe(h[["attempt_ts", "phone_id", "agent_id", "network_response", "disposition", "remark"]]
                 .rename(columns={"attempt_ts": "When", "phone_id": "Number", "agent_id": "Agent",
                                  "network_response": "Network", "disposition": "Disposition", "remark": "Remark"}),
                 hide_index=True, width="stretch")

# ------------------------------------------------------------------ numbers and safety
with tab_nums:
    st.markdown("Every number gets one status before any value decision. Compliance rules always win.")
    counts = nums.status.value_counts()
    st.markdown('<div class="legend">' + "".join(
        f"<div>{chip(s)} &nbsp;{counts.get(s, 0):,} numbers. {label}</div>" for s, (_, label) in STATUS.items())
        + "</div>", unsafe_allow_html=True)
    held_back = nums[nums.status.isin(["suppress_recycled", "locate_only", "verify_first"])]
    st.subheader("Suppression and verify-first list")
    st.dataframe(held_back[["account_id", "phone_id", "source", "status", "safety_reason"]]
                 .rename(columns={"account_id": "Account", "phone_id": "Number", "source": "Source",
                                  "status": "Status", "safety_reason": "Why"})
                 .style.map(colour_status, subset=["Status"]),
                 hide_index=True, width="stretch", height=380)
    st.download_button("Download list as CSV", held_back.to_csv(index=False), "suppression_list.csv", "text/csv")

# ------------------------------------------------------------------ skip-traces
with tab_trace:
    q = plan[plan.action == "trace"].sort_values("ev", ascending=False)
    st.markdown(f"**{len(q)} accounts** are worth tracing today. Traces are blocked for accounts where the "
                "borrower still holds a working number and is avoiding calls: those get a call or a visit instead.")
    st.dataframe(q[["account_id", "p_find", "ev", "reason"]]
                 .rename(columns={"account_id": "Account", "p_find": "Chance of finding a number",
                                  "ev": "Net value (Rs)", "reason": "Why"}),
                 hide_index=True, width="stretch",
                 column_config={"Chance of finding a number": st.column_config.ProgressColumn(
                     min_value=0, max_value=1, format="%.0f%%"),
                     "Net value (Rs)": st.column_config.NumberColumn(format="%.0f"),
                     "Why": st.column_config.TextColumn(width="large")})
    if replay is not None:
        st.subheader("Replay of past traces (validation and test accounts)")
        r = replay[replay.split.isin(["validation", "test"])]
        k = r[r.we_trace]
        s = r[~r.we_trace]
        st.dataframe(pd.DataFrame({
            "Traces": [len(r), len(k), len(s)],
            "Spend (Rs)": [r.cost_inr.sum(), k.cost_inr.sum(), s.cost_inr.sum()],
            "Found a phone": [r.found_phone.mean(), k.found_phone.mean(), s.found_phone.mean()],
            "Borrower reached on an existing number within 30 days": [
                r.reached_on_existing_30d.mean(), k.reached_on_existing_30d.mean(), s.reached_on_existing_30d.mean()],
        }, index=["All traces (15-failure rule)", "Engine would trace", "Engine would skip"]).round(2),
            width="stretch")

# ------------------------------------------------------------------ results
with tab_res:
    sc = REPORTS / "scorecard.md"
    if sc.exists():
        st.markdown(sc.read_text(encoding="utf-8"))
    else:
        st.info("Run `python scripts/07_evaluate.py` to produce the scorecard.")
    st.markdown("""
**Read with care**
- The RPC lift is measured on the randomised arm (34 accounts in validation and test), so its interval is wide.
  It needs CN's randomised trial to confirm.
- The state model is accurate on verified numbers but was overconfident on two of them, because repeated
  calls to the same number are treated as independent evidence.
- The rule-engine comparison approximates CN's dialling order (lowest priority slot, move on after 15 failures).
""")
