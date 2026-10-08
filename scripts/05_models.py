"""Stage 5: train the RPC model (factor heads + direct challenger), calibrate per portfolio,
and check it on validation. Test is not touched.
Run from the repo root (after Stage 4):  python scripts/05_models.py
Writes data/processed/rpc_model.pkl, data/processed/valid_predictions.parquet, reports/05_models.md
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from rpc.config import PROCESSED, REPORTS
from rpc.models import RPCModel, ece, feature_columns, PARAMS

pd.set_option("display.width", 200)
out = []


def section(title, body):
    text = body if isinstance(body, str) else body.to_string()
    out.append(f"## {title}\n\n```\n{text}\n```\n")
    print(f"\n=== {title}\n{text}")


def xent(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def metrics(p, y):
    return {"log_loss": xent(p, y), "auc": roc_auc_score(y >= .5, p), "brier": float(((p - y) ** 2).mean()),
            "mean_pred": float(p.mean()), "mean_obs": float(y.mean())}


t0 = time.time()
A = pd.read_parquet(PROCESSED / "attempts_model.parquet")
A = A.drop(columns=[c for c in A.columns if c.startswith("next_")])        # in case Stage 3 is older
SF = pd.read_parquet(PROCESSED / "state_features.parquet")
A = A.merge(SF, on=["cutoff", "link"], how="left")
train, valid = A[A.split == "train"].copy(), A[A.split == "validation"].copy()
yv = valid.y_rpc_soft.to_numpy()

# ---- fit
model = RPCModel().fit(train, valid)
raw = model.raw(valid)

# baselines: (1) historical rate by source x hour band, (2) same LightGBM without Stage 4 state features
band = lambda h: pd.cut(h, [7, 9, 15, 19], labels=["morning", "midday", "evening"])
rate = train.groupby([train.source, band(train.hour)], observed=True).y_rpc_soft.mean()
base1 = pd.Series(list(zip(valid.source, band(valid.hour)))).map(rate).fillna(train.y_rpc_soft.mean()).to_numpy()
no_state = [c for c in feature_columns(train) if not c.startswith("p_")]
m2 = RPCModel(); m2.features = no_state
m2.categories = {c: v for c, v in model.categories.items() if c in no_state}
b2 = lgb.train({**PARAMS, "objective": "cross_entropy"}, lgb.Dataset(m2.X(train), train.y_rpc_soft),
               num_boost_round=model.rounds["direct"])
base2 = b2.predict(m2.X(valid))

rows = {"source x hour-band rate": metrics(base1, yv), "LightGBM without state features": metrics(base2, yv),
        "direct LightGBM": metrics(raw.direct.to_numpy(), yv),
        "factor product (alive x answer x owned)": metrics(raw.factors.to_numpy(), yv)}
section("1. Validation, per call (uncalibrated)", pd.DataFrame(rows).T.round(4))
model.use = "factors" if rows["factor product (alive x answer x owned)"]["log_loss"] <= \
    rows["direct LightGBM"]["log_loss"] else "direct"
section("2. Model used for scoring", f"{model.use}   (boosting rounds per head: {model.rounds})")

heads = {"alive": (valid.y_alive, None), "answer": (valid.y_answered, valid.y_alive),
         "owned": (valid.y_rpc_soft >= .5, valid.y_answered)}
section("3. Factor heads on validation (AUC on their own subsets)", pd.Series({
    k: roc_auc_score(y[m.astype(bool)] if m is not None else y, raw[k][m.astype(bool)] if m is not None else raw[k])
    for k, (y, m) in heads.items()}).round(3))

# ---- calibrate on out-of-fold train predictions, check on validation
model.fit_calibration(train)
pv = model.predict(valid, calibrated=True)
cal = pd.DataFrame({pf: {"calls": len(g), "observed": g.y_rpc_soft.mean(),
                         "pred_raw": pv.loc[g.index, "p_rpc_raw"].mean(), "pred_calibrated": pv.loc[g.index, "p_rpc"].mean(),
                         "ece_raw": ece(pv.loc[g.index, "p_rpc_raw"].to_numpy(), g.y_rpc_soft.to_numpy()),
                         "ece_calibrated": ece(pv.loc[g.index, "p_rpc"].to_numpy(), g.y_rpc_soft.to_numpy())}
                    for pf, g in valid.groupby("portfolio")}).T
section("4. Calibration by portfolio (validation)", cal.round(3))
m_raw, m_cal = metrics(pv.p_rpc_raw.to_numpy(), yv), metrics(pv.p_rpc.to_numpy(), yv)
model.apply_calibration = m_cal["log_loss"] < m_raw["log_loss"]
section("5. Calibrated vs raw (validation decides)", "\n".join([
    pd.DataFrame({"raw": m_raw, "calibrated": m_cal}).T.round(4).to_string(),
    f"apply calibration: {model.apply_calibration}"]))
pv = model.predict(valid)

# random arm: numbers the rule engine would not necessarily have picked
ra = valid.dialling_arm.eq("random_contact_point")
oof_ra = train.dialling_arm.eq("random_contact_point")
section("6. Calibration on random-arm calls (checks selection bias)", pd.DataFrame({
    "validation random arm": {"calls": int(ra.sum()), "observed": valid.y_rpc_soft[ra].mean(),
                              "predicted": pv.p_rpc[ra].mean(), "predicted_raw": pv.p_rpc_raw[ra].mean()},
    "train random arm (out-of-fold)": {"calls": int(oof_ra.sum()), "observed": train.y_rpc_soft[oof_ra].mean(),
                                       "predicted_raw": model.oof[oof_ra].mean()}}).T.round(3))

# per contact point: rank numbers for next week (CN's AUC metric)
snaps = pd.read_parquet(PROCESSED / "snapshots.parquet")
sv = snaps[(snaps.split == "validation") & (snaps.next_att > 0)].merge(SF, on=["cutoff", "link"], how="left")
sv = sv.assign(hour=10, dow=1, is_bot=False, lang_match=True)
sv["p"] = model.predict(sv).p_rpc.to_numpy()
sv["next_rpc_any"] = sv.next_rpc_any.astype(bool)
section("7. Per contact point, next 7 days (validation)",
        f"AUC {roc_auc_score(sv.next_rpc_any, sv.p):.3f} over {len(sv):,} (cutoff, number) rows; "
        f"observed contact rate {sv.next_rpc_any.mean():.1%}")

# best hour: model-estimated gain from moving each validation call to that number's best hour
self_calls = valid[valid.relation_recorded == "self"]
grid = model.by_hour(self_calls, is_bot=False)
actual = model.predict(self_calls.assign(is_bot=False)).p_rpc.to_numpy()
best_h = grid.idxmax(axis=1)
gain = pd.DataFrame({"income": self_calls.income_type.values, "actual": actual, "best": grid.max(axis=1).values,
                     "best_hour": best_h.values})
section("8. Best hour (model estimate, validation self numbers, agent calls)", pd.concat([
    gain.groupby("income")[["actual", "best"]].mean().round(3),
    gain.groupby("income").best_hour.agg(lambda s: s.value_counts(normalize=True).round(2).head(3).to_dict())], axis=1))
section("9. Top drivers by head (share of gain)", pd.concat(
    {h: model.importance(h, 10).round(3).reset_index().rename(columns={"index": "feature", 0: "gain"})
     for h in ["alive", "answer", "owned"]}, axis=1).to_string())

model.save(PROCESSED / "rpc_model.pkl")
pd.concat([valid[["attempt_id", "link", "cutoff", "portfolio", "y_rpc_soft"]].reset_index(drop=True),
           pv.reset_index(drop=True)], axis=1).to_parquet(PROCESSED / "valid_predictions.parquet", index=False)
section("10. Run time", f"{time.time() - t0:.0f}s")
REPORTS.mkdir(exist_ok=True)
(REPORTS / "05_models.md").write_text("# Stage 5: RPC model\n\n" + "\n".join(out), encoding="utf-8")
