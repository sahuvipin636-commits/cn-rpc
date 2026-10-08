"""Stage 4: fit the hidden-state model on train accounts, check it on validation gold,
and export filtered state probabilities for every weekly cutoff.
Run from the repo root (after Stage 3):  python scripts/04_states.py
Writes data/processed/state_features.parquet, data/processed/state_model.json, reports/04_states.md
"""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from rpc.config import PROCESSED, REPORTS, WINDOW_START, GOLD_STATUSES
from rpc.io import load_all, load_gold, add_flags
from rpc.features import enrich_attempts
from rpc.labels import weekly_cutoffs, SCORING_CUTOFF
from rpc.states import (StateModel, observation_codes, group_probs, factor_columns, STATES, OBS, VERIFY_T)

pd.set_option("display.width", 200)
out = []


def section(title, body):
    text = body if isinstance(body, str) else body.to_string()
    out.append(f"## {title}\n\n```\n{text}\n```\n")
    print(f"\n=== {title}\n{text}")


t0 = time.time()
t = load_all()
E = enrich_attempts(add_flags(t["dial_attempts"]), pd.read_parquet(PROCESSED / "remarks_dial.parquet"),
                    pd.read_parquet(PROCESSED / "attempt_trust.parquet"))
obs = observation_codes(E)

links = t["phones"].merge(t["splits"], on="account_id")
links["link"] = links.account_id + "|" + links.phone_id
links["t0"] = (links.added_date.clip(lower=WINDOW_START) - WINDOW_START).dt.days.astype(float)
gold_tr = load_gold(("train",)).assign(link=lambda d: d.account_id + "|" + d.phone_id)
gold_va = load_gold(("validation",)).assign(link=lambda d: d.account_id + "|" + d.phone_id)
links = links.merge(gold_tr[["link", "verified_status"]].rename(columns={"verified_status": "gold_status"}),
                    on="link", how="left")

train_links = links[links.split == "train"]
seqs_train = StateModel.sequences(obs, train_links)


def predict_gold(model, gold):
    sub = links[links.link.isin(gold.link)].drop(columns="gold_status")
    seqs = StateModel.sequences(obs, sub)
    p = np.vstack([model.filter_at(s, [VERIFY_T])[0] for s in seqs])
    gp = group_probs(p)
    gp.insert(0, "link", [s["link"] for s in seqs])
    return gold[["link", "verified_status"]].merge(gp, on="link")


def score(df, cols=GOLD_STATUSES):
    P = df[cols].to_numpy()
    P = np.clip(P, 1e-6, 1)
    P = P / P.sum(axis=1, keepdims=True)
    y = df.verified_status.map({c: i for i, c in enumerate(cols)}).to_numpy()
    return log_loss(y, P, labels=list(range(len(cols)))), (P.argmax(1) == y).mean()


# 1. grid search the two time scales on train gold (filtered predictions, no anchoring at predict time)
grid = []
for decay in [0.5, 1, 2, 4]:
    for avoid in [0.5, 1, 2]:
        m = StateModel(decay, avoid).fit(seqs_train, VERIFY_T, n_iter=8)
        ll, acc = score(predict_gold(m, gold_tr))
        grid.append({"decay_scale": decay, "avoid_scale": avoid, "train_logloss": ll, "train_acc": acc})
grid = pd.DataFrame(grid).sort_values("train_logloss")
section("1. Time-scale grid search (train gold)", grid.round(3).head(6).to_string(index=False))
best = grid.iloc[0]

# 2. final fit
model = StateModel(best.decay_scale, best.avoid_scale).fit(seqs_train, VERIFY_T, n_iter=30)
section("2. Learned emissions: how each state shows up in calls (row %)",
        (pd.DataFrame(model.B, index=STATES, columns=OBS) * 100).round(1))
section("3. Learned starting mix by source (%)",
        (pd.DataFrame(model.pi, index=STATES).T * 100).round(0))

# 3. validation: model vs baselines
va = predict_gold(model, gold_va)
tr = predict_gold(model, gold_tr)

prior = (gold_tr.merge(links[["link", "source"]], on="link").groupby("source").verified_status
         .value_counts(normalize=True).unstack(fill_value=0).reindex(columns=GOLD_STATUSES, fill_value=0))
prior = (prior + 0.02).div((prior + 0.02).sum(axis=1), axis=0)
va_prior = gold_va.merge(links[["link", "source"]], on="link").join(prior, on="source")

last = obs.groupby("link").code.last()
rule = {OBS.index("does_not_exist"): "invalid_number", OBS.index("third_party"): "third_party_number",
        OBS.index("wrong_new_owner"): "not_borrower_number", OBS.index("wrong_other"): "not_borrower_number",
        OBS.index("switched_off"): "switched_off", OBS.index("switched_off_long"): "switched_off",
        OBS.index("not_reachable"): "switched_off"}
va_rule = gold_va.assign(pred=gold_va.link.map(last).map(rule).fillna("borrower_number"))

ll_m, acc_m = score(va)
ll_p, acc_p = score(va_prior)
section("4. Validation gold (never used in fitting)", "\n".join([
    f"numbers: {len(va)}",
    f"state model     log-loss {ll_m:.3f}   accuracy {acc_m:.0%}",
    f"source prior    log-loss {ll_p:.3f}   accuracy {acc_p:.0%}",
    f"last-call rule                      accuracy {(va_rule.pred == va_rule.verified_status).mean():.0%}",
    f"(train gold, fitted: log-loss {score(tr)[0]:.3f}, accuracy {score(tr)[1]:.0%})"]))
pred = va[GOLD_STATUSES].idxmax(axis=1)
section("5. Validation confusion (rows = verified, cols = predicted)",
        pd.crosstab(va.verified_status, pred).reindex(index=GOLD_STATUSES, columns=GOLD_STATUSES, fill_value=0))

pred_tr = tr[GOLD_STATUSES].idxmax(axis=1)
section("5b. Train confusion (fitted; shows the weak classes)",
        pd.crosstab(tr.verified_status, pred_tr).reindex(index=GOLD_STATUSES, columns=GOLD_STATUSES, fill_value=0))
dialled = set(obs.link)
both = pd.concat([tr.assign(split="train"), va.assign(split="validation")])
both["dialled"] = both.link.isin(dialled)
section("6. Accuracy by whether the number was ever dialled (train + validation)",
        both.assign(correct=both[GOLD_STATUSES].idxmax(axis=1) == both.verified_status)
        .groupby(["split", "dialled"]).correct.agg(["size", "mean"]).round(2))

# 4. export filtered states for every cutoff (all accounts; params come from train only)
cut_days = [(c - WINDOW_START).days for c in weekly_cutoffs() + [SCORING_CUTOFF]]
cut_ts = weekly_cutoffs() + [SCORING_CUTOFF]
all_seqs = StateModel.sequences(obs, links.drop(columns="gold_status"))
rows = []
for s in all_seqs:
    p = model.filter_at(s, cut_days)
    ok = ~np.isnan(p[:, 0])
    if ok.any():
        f = factor_columns(p[ok])
        f.insert(0, "link", s["link"])
        f.insert(0, "cutoff", [c for c, k in zip(cut_ts, ok) if k])
        rows.append(f)
sf = pd.concat(rows, ignore_index=True)
sf.to_parquet(PROCESSED / "state_features.parquet", index=False)
json.dump({"decay_scale": float(best.decay_scale), "avoid_scale": float(best.avoid_scale),
           "Q": model.Q.tolist(), "B": model.B.tolist(), "pi": {k: v.tolist() for k, v in model.pi.items()},
           "states": STATES, "obs": OBS}, open(PROCESSED / "state_model.json", "w"), indent=1)
last_cut = sf[sf.cutoff == SCORING_CUTOFF]
section("7. State mix across all numbers on 30 June (%)",
        (last_cut[[f"p_{s}" for s in STATES]].mean() * 100).round(1))
section("8. Run time", f"{time.time() - t0:.0f}s")

REPORTS.mkdir(exist_ok=True)
(REPORTS / "04_states.md").write_text("# Stage 4: hidden-state model\n\n" + "\n".join(out), encoding="utf-8")
print(f"\nwrote {PROCESSED / 'state_features.parquet'}")
