"""Stage 2: tag remarks (3 languages) and score agent reliability.
Run from the repo root:  python scripts/02_remarks_agents.py
Writes data/processed/{remarks_dial, remarks_field, attempt_trust}.parquet,
data/processed/agent_reliability.csv and reports/02_remarks_agents.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from rpc.config import PROCESSED, REPORTS
from rpc.io import load, load_gold, add_flags
from rpc.remarks import tag_dial_remarks, tag_field_remarks, DIAL_TAGS, FIELD_TAGS
from rpc.agents import agent_reliability, attempt_rpc_trust

pd.set_option("display.width", 200)
out = []


def section(title, body):
    text = body if isinstance(body, str) else body.to_string()
    out.append(f"## {title}\n\n```\n{text}\n```\n")
    print(f"\n=== {title}\n{text}")


PROCESSED.mkdir(parents=True, exist_ok=True)
att = add_flags(load("dial_attempts"))
visits = load("field_visits")
gold = load_gold(splits=("train",))

# ---- remarks
rd = tag_dial_remarks(att)
rf = tag_field_remarks(visits)
rd.to_parquet(PROCESSED / "remarks_dial.parquet", index=False)
rf.to_parquet(PROCESSED / "remarks_field.parquet", index=False)

section("1. Match coverage", "\n".join([
    f"dial remarks matched:  {rd.canon.notna().mean():.1%} of {len(rd):,}",
    f"field remarks matched: {(rf.canon.notna() | rf.tag_address_correction).mean():.1%} of {len(rf):,}",
    f"unmatched dial examples:  {att.remark[rd.canon.isna()].unique()[:8].tolist()}",
    f"unmatched field examples: {visits.remark[rf.canon.isna() & ~rf.tag_address_correction].unique()[:8].tolist()}"]))
section("2. Language mix of dial remarks (share)", rd.lang.value_counts(normalize=True).round(3))

# Sanity: tags should agree with the disposition the agent typed
fam = pd.Series("other", index=rd.index)
for tag, name in [("no_answer", "no_answer"), ("rejected", "call_rejected"), ("switched_off", "switched_off"),
                  ("not_reachable", "not_reachable"), ("invalid", "invalid_number"), ("wrong_number", "wrong_number"),
                  ("third_party", "third_party"), ("rpc", "rpc")]:
    fam[rd[f"tag_{tag}"]] = name
disp_fam = att.disposition.where(~att.logged_rpc, "rpc").where(~att.logged_tp, "third_party")
section("3. Remark tag vs typed disposition: agreement", f"{(fam == disp_fam).mean():.1%} "
        "(remarks mirror dispositions, so they cannot expose mislogging)")

# Signal check on train gold: details a disposition loses
x = att.join(rd.drop(columns="attempt_id")).merge(gold, on=["account_id", "phone_id"])
wn = x[x.tag_wrong_number]
section("4. Wrong-number calls: 'new owner' phrasing by verified status (train gold)",
        pd.crosstab(wn.verified_status, wn.tag_new_owner.map({True: "new_owner", False: "other"})))
so = x[x.tag_switched_off]
section("5. Switched-off calls: 'off for months/weeks' phrasing by verified status (train gold)",
        pd.crosstab(so.verified_status, so.tag_long_off.map({True: "long_off", False: "other"})))
cb = rd[rd.tag_callback]
section("6. Callbacks asking for the evening", f"{cb.tag_evening.mean():.0%} of {len(cb):,} callbacks")

tag_counts = rf[[f"tag_{t}" for t in FIELD_TAGS]].sum().sort_values(ascending=False)
section("7. Field remark tags", tag_counts.rename(lambda c: c[4:]))
fo = visits[["outcome"]].join(rf[["tag_at_home", "tag_at_shop", "tag_at_work", "tag_address_correction"]])
section("8. Field detail by outcome (share of visits)", fo.groupby("outcome").mean().round(2))

# ---- agents
rel = agent_reliability(att, gold)
rel.to_csv(PROCESSED / "agent_reliability.csv", index=False)
trust = pd.DataFrame({"attempt_id": att.attempt_id, "rpc_trust": attempt_rpc_trust(att, rel)})
trust.to_parquet(PROCESSED / "attempt_trust.parquet", index=False)
cols = ["agent_id", "attempts", "logged_rpc_rate", "gold_nonborrower_answered", "gold_nonborrower_logged_rpc",
        "p_rpc_on_nonborrower", "rpc_on_unanswered", "rpc_trust", "flag"]
section(f"9. Agent reliability (pooled RPC-on-non-borrower rate {rel.attrs['pooled_rate']:.1%})",
        rel[cols].sort_values("rpc_trust").round(3).set_index("agent_id"))
w = trust.rpc_trust.dropna()
section("10. Logged RPCs after weighting",
        f"{len(w):,} logged RPCs -> {w.sum():,.0f} effective ({1 - w.mean():.1%} down-weighted)")

REPORTS.mkdir(exist_ok=True)
(REPORTS / "02_remarks_agents.md").write_text("# Stage 2: remarks and agents\n\n" + "\n".join(out), encoding="utf-8")
print(f"\nwrote {PROCESSED} outputs and {REPORTS / '02_remarks_agents.md'}")
