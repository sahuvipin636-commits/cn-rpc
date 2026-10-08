"""Stage 7: one-time evaluation on the TEST accounts, plus the CN metrics scorecard.
Run from the repo root (after Stage 6):  python scripts/07_evaluate.py
Writes reports/07_evaluate.md, reports/scorecard.md, reports/trace_replay.csv
"""
import json
import pickle
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from rpc.config import PROCESSED, REPORTS, WINDOW_START, COSTS
from rpc.io import load_all, add_flags
from rpc.features import enrich_attempts
from rpc.labels import weekly_cutoffs
from rpc.models import RPCModel
from rpc.states import StateModel, observation_codes
from rpc import evaluate as ev

pd.set_option("display.width", 220)
out = []


def section(title, body):
    text = body if isinstance(body, str) else body.to_string()
    out.append(f"## {title}\n\n```\n{text}\n```\n")
    print(f"\n=== {title}\n{text}")


t0 = time.time()
t = load_all()
E = enrich_attempts(add_flags(t["dial_attempts"]), pd.read_parquet(PROCESSED / "remarks_dial.parquet"),
                    pd.read_parquet(PROCESSED / "attempt_trust.parquet"))
SF = pd.read_parquet(PROCESSED / "state_features.parquet")
snaps = pd.read_parquet(PROCESSED / "snapshots.parquet")
A = pd.read_parquet(PROCESSED / "attempts_model.parquet")
A = A.drop(columns=[c for c in A.columns if c.startswith("next_")]).merge(SF, on=["cutoff", "link"], how="left")
model = RPCModel.load(PROCESSED / "rpc_model.pkl")
cfg = pickle.load(open(PROCESSED / "decision_config.pkl", "rb"))
sm = StateModel.from_json(json.load(open(PROCESSED / "state_model.json")))
rf = pd.read_parquet(PROCESSED / "remarks_field.parquet")
cutoffs = weekly_cutoffs()

links = t["phones"].merge(t["splits"], on="account_id")
links["link"] = links.account_id + "|" + links.phone_id
links["t0"] = (links.added_date.clip(lower=WINDOW_START) - WINDOW_START).dt.days.astype(float)
obs = observation_codes(E)

# 1. hidden states on test gold
st = ev.eval_states(sm, obs, links)
section(f"1. Hidden-state model on TEST gold ({st['n']} numbers)", st["table"].round(3))
section("2. Test confusion (rows = verified, cols = predicted)", st["confusion"])

# 2. safety on test gold
sa = ev.eval_safety(pd.read_parquet(PROCESSED / "numbers_plan.parquet"))
section("3. Safety layer on TEST gold", sa["table"].to_string() +
        f"\nrisky numbers protected: {sa['risky_protected']} | borrower numbers blocked: {sa['borrower_blocked']}"
        f" (plus {sa['borrower_verify_first']} kept callable with identity check)")

# 3. prediction quality on test calls
Atr, Ate = A[A.split == "train"], A[A.split == "test"].reset_index(drop=True)
rp = ev.eval_rpc(model, Atr, Ate, snaps[snaps.split == "test"], SF)
section("4. Per-call RPC prediction on TEST calls", rp["calls"].round(4))
section("5. Calibration by portfolio (TEST)", rp["by_portfolio"].round(3))
section("6. Factor heads (TEST AUC) and per contact point",
        rp["heads"].round(3).to_string() + f"\ncontact point, next 7 days: AUC {rp['contact_point_auc']:.3f} "
        f"over {rp['contact_point_rows']:,} rows")

# 4. policy value on the randomised arm
status = ev.status_history(snaps, SF, cfg["thresholds"])
o_test = ev.ope(model, A[A.split == "test"], snaps, SF, status)
o_vt = ev.ope(model, A[A.split.isin(["validation", "test"])], snaps, SF, status)
for name, o in [("TEST only", o_test), ("validation + test", o_vt)]:
    section(f"7. RPC per 1,000 dials, randomised arm, {name} ({o['calls']} calls, {o['accounts']} accounts)",
            o["table"].round(1).to_string() +
            f"\nnew minus rule: 90% CI {o['lift_ci'][0]:.1f} to {o['lift_ci'][1]:.1f} per 1,000; "
            f"P(new better) = {o['p_new_better']:.0%}")

# 5. skip-trace replay
tr = ev.trace_replay(t, snaps, SF, model, cfg, rf, cutoffs, E)
tr.to_csv(REPORTS / "trace_replay.csv", index=False)
summ = lambda d: pd.Series({"traces": len(d), "cost_rs": d.cost_inr.sum(), "found_phone": d.found_phone.mean(),
                            "found_any": d.found_any.mean(), "rs_per_phone_found": d.cost_inr.sum() / max(d.found_phone.sum(), 1),
                            "reached_on_existing_30d": d.reached_on_existing_30d.mean(), "paid_30d": d.paid30.mean()})
oos = tr[tr.split.isin(["validation", "test"])]
for name, d in [("TEST", tr[tr.split == "test"]), ("validation + test (out of sample)", oos),
                ("all splits (includes train traces the trace model learned from)", tr)]:
    section(f"8. Skip-trace replay, {name}", pd.DataFrame({
        "historical (15-failure rule)": summ(d), "new engine would trace": summ(d[d.we_trace]),
        "new engine would skip": summ(d[~d.we_trace])}).T.round(3).to_string() +
        f"\nwhat the new engine does instead: {d[~d.we_trace].action.value_counts().to_dict()}")

# 6. wasted dials (test accounts, incumbent's own calls)
w = ev.waste(Ate, status)
section("9. TEST dials by the status the new system gave the number at the time", w.round(3))

# 7. scorecard
ot, d_all, d_te = o_test["table"], oos, tr[tr.split == "test"]
stop = w.loc[w.index.isin(ev.STOP)]
kept = d_all[d_all.we_trace]
score = [
    ("RPC per 1,000 dials, new vs rule (random arm, val+test, SNIPS)",
     f"{o_vt['table'].loc['new policy', 'rpc_per_1000_dials']:.0f} vs "
     f"{o_vt['table'].loc['rule engine (approx.)', 'rpc_per_1000_dials']:.0f} "
     f"(lift 90% CI {o_vt['lift_ci'][0]:.0f} to {o_vt['lift_ci'][1]:.0f})"),
    ("RPC prediction per contact point, next 7 days (test)", f"AUC {rp['contact_point_auc']:.3f}"),
    ("RPC prediction per call (test)", f"AUC {rp['calls'].loc['new model', 'auc']:.3f}, predicted "
     f"{rp['calls'].loc['new model', 'mean_pred']:.1%} vs observed {rp['calls'].loc['new model', 'mean_obs']:.1%}"),
    ("Contact-point state accuracy (test gold)", f"{st['table'].loc['state model', 'accuracy']:.0%} vs "
     f"{st['table'].loc['source prior (train gold)', 'accuracy']:.0%} source prior, "
     f"{st['table'].loc['last-call rule', 'accuracy']:.0%} last-call rule ({st['n']} numbers); log-loss "
     f"{st['table'].loc['state model', 'log_loss']:.2f} vs {st['table'].loc['source prior (train gold)', 'log_loss']:.2f} "
     "(2 overconfident misses: repeated calls treated as independent evidence)"),
    ("Dials on numbers the system would have stopped (test)",
     f"{stop.dials.sum():,} of {w.dials.sum():,} ({stop.dials.sum() / w.dials.sum():.1%}); "
     f"they reached the borrower {np.average(stop.borrower_reached_rate, weights=stop.dials):.1%} of the time "
     f"vs {Ate.y_rpc_soft.mean():.1%} overall"),
    ("Skip-traces (validation + test, out of sample)", f"new engine keeps {len(kept)} of {len(d_all)}; spend "
     f"Rs {kept.cost_inr.sum():,.0f} vs Rs {d_all.cost_inr.sum():,.0f}; phones found {int(kept.found_phone.sum())} vs "
     f"{int(d_all.found_phone.sum())}; borrower reached on existing numbers within 30 days for "
     f"{d_all[~d_all.we_trace].reached_on_existing_30d.mean():.0%} of skipped vs {kept.reached_on_existing_30d.mean():.0%} of kept"),
    ("Strangers reached with no verify-first flag, per 1,000 dials (random arm, val+test)",
     f"{o_vt['table'].loc['new policy', 'unprotected_stranger_per_1000']:.0f} new vs "
     f"{o_vt['table'].loc['rule engine (approx.)', 'unprotected_stranger_per_1000']:.0f} rule"),
    ("Risky numbers protected / borrower numbers blocked (test gold)",
     f"{sa['risky_protected']} protected, {sa['borrower_blocked']} blocked"),
]
sc = "\n".join(f"| {k} | {v} |" for k, v in score)
(REPORTS / "scorecard.md").write_text("# CN PS2 scorecard (test split)\n\n| Metric | Result |\n|---|---|\n" + sc + "\n",
                                      encoding="utf-8")
section("10. Scorecard", "\n".join(f"{k}: {v}" for k, v in score))
section("11. Run time", f"{time.time() - t0:.0f}s")
(REPORTS / "07_evaluate.md").write_text("# Stage 7: evaluation on test\n\n" + "\n".join(out), encoding="utf-8")
