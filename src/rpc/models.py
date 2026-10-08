"""Stage 5: probability that a call to this number, at this hour, reaches the borrower.

Three factor heads (LightGBM), multiplied:
  alive   P(line works)                       trained on all calls
  answer  P(someone answers | alive)          trained on alive calls
  owned   P(it is the borrower | answered)    trained on answered calls, trust-weighted label
A direct one-model challenger is trained alongside; the better one on validation is used for
scoring. The factor heads always provide the reason codes.
Calibration: isotonic per portfolio, fitted on out-of-fold train predictions (grouped by account).

Selection bias note: the rule engine picks numbers using priority_slot and failure streaks,
which are both features, so P(RPC | features, dialled) is learnable from logged calls. The
random-arm propensities are used in Stage 7 to evaluate policies, and here to check
calibration on numbers the rule engine would not have chosen.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

NOT_FEATURES = {"attempt_id", "cutoff", "link", "attempt_ts", "agent_id", "agent_lang", "dialling_arm",
                "propensity", "phone_id", "account_id", "split"}
ATTEMPT_LEVEL = ["hour", "dow", "is_bot", "lang_match"]   # chosen at dial time, not properties of the number
HEADS = {"alive": ("y_alive", None), "answer": ("y_answered", "y_alive"), "owned": ("y_rpc_soft", "y_answered")}
PARAMS = dict(learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, seed=7)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in NOT_FEATURES and not c.startswith(("y_", "next_"))]
    assert not [c for c in cols if c.startswith(("y_", "next_"))], "label columns leaked into features"
    return cols


class RPCModel:
    def __init__(self):
        self.features: list[str] = []
        self.categories: dict[str, list] = {}
        self.boosters: dict[str, lgb.Booster] = {}
        self.calibrators: dict[str, IsotonicRegression] = {}
        self.use: str = "factors"
        self.apply_calibration: bool = True   # decided on validation in Stage 5

    # ---------- data ----------
    def X(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.features].copy()
        for c, cats in self.categories.items():
            x[c] = pd.Categorical(x[c].astype(object), categories=cats)
        for c in x.columns:
            if x[c].dtype == bool:
                x[c] = x[c].astype(int)
        return x

    def _setup(self, train: pd.DataFrame):
        self.features = feature_columns(train)
        self.categories = {c: sorted(train[c].dropna().astype(str).unique())
                           for c in self.features
                           if not pd.api.types.is_numeric_dtype(train[c]) and train[c].dtype != bool}

    # ---------- training ----------
    @staticmethod
    def _train_one(Xtr, ytr, Xva, yva, objective, rounds=None):
        dtr = lgb.Dataset(Xtr, ytr)
        if rounds is None:
            dva = lgb.Dataset(Xva, yva, reference=dtr)
            b = lgb.train({**PARAMS, "objective": objective}, dtr, num_boost_round=2000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            return b, b.best_iteration
        return lgb.train({**PARAMS, "objective": objective}, dtr, num_boost_round=rounds), rounds

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame):
        self._setup(train)
        Xtr, Xva = self.X(train), self.X(valid)
        self.rounds = {}
        for name, (y, cond) in HEADS.items():
            mtr = train[cond].astype(bool).to_numpy() if cond else np.ones(len(train), bool)
            mva = valid[cond].astype(bool).to_numpy() if cond else np.ones(len(valid), bool)
            obj = "cross_entropy" if name == "owned" else "binary"
            self.boosters[name], self.rounds[name] = self._train_one(
                Xtr[mtr], train[y].to_numpy(float)[mtr], Xva[mva], valid[y].to_numpy(float)[mva], obj)
        self.boosters["direct"], self.rounds["direct"] = self._train_one(
            Xtr, train.y_rpc_soft.to_numpy(float), Xva, valid.y_rpc_soft.to_numpy(float), "cross_entropy")
        return self

    def raw(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.X(df)
        p = {k: b.predict(X) for k, b in self.boosters.items()}
        p["factors"] = p["alive"] * p["answer"] * p["owned"]
        return pd.DataFrame(p, index=df.index)

    def fit_calibration(self, train: pd.DataFrame, n_folds: int = 5):
        """Out-of-fold predictions on train (grouped by account), then isotonic per portfolio."""
        oof = pd.Series(np.nan, index=train.index)
        for tr_idx, te_idx in GroupKFold(n_folds).split(train, groups=train.account_id):
            sub = RPCModel()
            sub.features, sub.categories = self.features, self.categories
            a, b = train.iloc[tr_idx], train.iloc[te_idx]
            Xa = sub.X(a)
            for name, (y, cond) in HEADS.items():
                m = a[cond].astype(bool).to_numpy() if cond else np.ones(len(a), bool)
                obj = "cross_entropy" if name == "owned" else "binary"
                sub.boosters[name], _ = self._train_one(Xa[m], a[y].to_numpy(float)[m], None, None, obj,
                                                        rounds=self.rounds[name])
            sub.boosters["direct"], _ = self._train_one(Xa, a.y_rpc_soft.to_numpy(float), None, None,
                                                        "cross_entropy", rounds=self.rounds["direct"])
            oof.iloc[te_idx] = sub.raw(b)[self.use].to_numpy()
        self.oof = oof
        for pf, g in train.groupby("portfolio"):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            iso.fit(oof.loc[g.index], g.y_rpc_soft)
            self.calibrators[pf] = iso
        return self

    # ---------- scoring ----------
    def predict(self, df: pd.DataFrame, calibrated: bool | None = None) -> pd.DataFrame:
        calibrated = self.apply_calibration if calibrated is None else calibrated
        r = self.raw(df)
        r["p_rpc_raw"] = r[self.use]
        r["p_rpc"] = r.p_rpc_raw
        if calibrated and self.calibrators:
            for pf, idx in df.groupby("portfolio").groups.items():
                if pf in self.calibrators:
                    r.loc[idx, "p_rpc"] = self.calibrators[pf].predict(r.loc[idx, "p_rpc_raw"])
        return r

    def by_hour(self, rows: pd.DataFrame, hours=range(8, 19), is_bot: bool = False) -> pd.DataFrame:
        """P(RPC) for each row at every allowed hour (contact hours only). Columns = hours."""
        out = {}
        for h in hours:
            x = rows.assign(hour=h, is_bot=is_bot)
            out[h] = self.predict(x).p_rpc.to_numpy()
        return pd.DataFrame(out, index=rows.index)

    def importance(self, head: str, top: int = 15) -> pd.Series:
        b = self.boosters[head]
        return (pd.Series(b.feature_importance("gain"), index=b.feature_name())
                .sort_values(ascending=False).head(top) / b.feature_importance("gain").sum())

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "RPCModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return err / len(p)
