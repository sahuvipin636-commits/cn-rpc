"""Stage 4: hidden-state model for every (account, phone).

States change in continuous time (P(dt) = expm(Q * dt)), every call is a noisy observation.
Fitting (train accounts only):
  - transitions: a few readable daily rates, scaled by a small grid search on train gold
  - emissions and starting mix per source: EM (forward-backward), with Dirichlet smoothing
  - train-split gold labels anchor each sequence's state on the verification date
Output: filtered state probabilities as of any date, using only calls before that date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import expm

from .config import WINDOW_START, VERIFIED_DATE

STATES = ["reachable", "avoiding", "temp_off", "dead", "recycled", "third_party", "invalid"]
S = len(STATES)
GOLD_GROUPS = {"borrower_number": ["reachable", "avoiding"], "third_party_number": ["third_party"],
               "not_borrower_number": ["recycled"], "switched_off": ["temp_off", "dead"],
               "invalid_number": ["invalid"]}
OBS = ["rnr", "rejected", "not_reachable", "switched_off", "switched_off_long", "does_not_exist",
       "rpc", "third_party", "wrong_new_owner", "wrong_other", "answered_other"]
O = {o: i for i, o in enumerate(OBS)}
SOURCES = ["kyc_origination", "borrower_update", "bureau", "reference", "employer", "skip_trace"]

# Starting guesses; EM refines emissions and starting mixes, rates are scaled by grid search
B0 = np.array([
    # rnr  rej  nrch  off  offL  dne   rpc   tp   wNew wOth  oth
    [.55, .06, .04, .01, .002, .0005, .30, .02, .001, .005, .01],   # reachable
    [.50, .30, .04, .02, .005, .0005, .10, .01, .001, .005, .01],   # avoiding
    [.10, .01, .45, .40, .03, .001, .005, .003, .001, .002, .001],  # temp_off
    [.03, .005, .30, .50, .15, .01, .003, .002, .001, .002, .001],  # dead
    [.40, .05, .05, .05, .01, .001, .05, .05, .12, .20, .02],       # recycled
    [.45, .05, .05, .02, .005, .001, .06, .32, .005, .02, .02],     # third_party
    [.02, .001, .02, .02, .005, .90, .001, .001, .005, .02, .001],  # invalid
])
PI0 = {
    "kyc_origination": [.55, .15, .05, .10, .07, .05, .03], "borrower_update": [.65, .15, .05, .05, .03, .05, .02],
    "bureau": [.30, .08, .05, .15, .20, .17, .05], "reference": [.08, .02, .03, .03, .04, .78, .02],
    "employer": [.02, .01, .03, .03, .03, .86, .02], "skip_trace": [.65, .10, .05, .05, .05, .08, .02],
}
BASE_RATES = {  # per day
    ("reachable", "avoiding"): .010, ("avoiding", "reachable"): .030,
    ("reachable", "temp_off"): .010, ("temp_off", "reachable"): .150,
    ("temp_off", "dead"): .020, ("reachable", "dead"): .002, ("avoiding", "dead"): .002,
    ("dead", "recycled"): .005, ("dead", "reachable"): .002,
}
DECAY = {("temp_off", "dead"), ("reachable", "dead"), ("avoiding", "dead"), ("dead", "recycled")}
AVOID = {("reachable", "avoiding"), ("avoiding", "reachable")}


def make_Q(decay_scale: float = 1.0, avoid_scale: float = 1.0) -> np.ndarray:
    Q = np.zeros((S, S))
    for (a, b), r in BASE_RATES.items():
        Q[STATES.index(a), STATES.index(b)] = r * (decay_scale if (a, b) in DECAY else 1) * \
            (avoid_scale if (a, b) in AVOID else 1)
    np.fill_diagonal(Q, -Q.sum(1))
    return Q


def _days(ts) -> np.ndarray:
    return (pd.to_datetime(ts) - WINDOW_START).dt.total_seconds().to_numpy() / 86400


def observation_codes(E: pd.DataFrame) -> pd.DataFrame:
    """Map each call to an observation code and a trust weight (only for logged RPCs)."""
    nr = E.network_response
    code = np.select(
        [nr.eq("ring_no_answer"), nr.eq("busy_rejected"), nr.eq("not_reachable"),
         nr.eq("switched_off") & E.tag_long_off.fillna(False), nr.eq("switched_off"),
         nr.eq("number_does_not_exist"),
         E.logged_rpc, E.logged_tp, E.logged_wrong & E.tag_new_owner.fillna(False), E.logged_wrong],
        [O["rnr"], O["rejected"], O["not_reachable"], O["switched_off_long"], O["switched_off"],
         O["does_not_exist"], O["rpc"], O["third_party"], O["wrong_new_owner"], O["wrong_other"]],
        O["answered_other"])
    w = np.where(code == O["rpc"], E.rpc_trust.fillna(1.0), 1.0)
    return pd.DataFrame({"link": E.link, "t": _days(E.attempt_ts), "code": code, "w": w,
                         "ts": E.attempt_ts}).sort_values(["link", "t"], kind="stable")


class StateModel:
    def __init__(self, decay_scale=1.0, avoid_scale=1.0, smoothing=20.0):
        self.Q = make_Q(decay_scale, avoid_scale)
        self.B = B0 / B0.sum(1, keepdims=True)
        self.pi = {s: np.array(v) / sum(v) for s, v in PI0.items()}
        self.smoothing = smoothing
        self._P = {}

    # ---------- building blocks ----------
    def P(self, dt: float) -> np.ndarray:
        k = round(max(dt, 0.0), 1)
        if k not in self._P:
            self._P[k] = expm(self.Q * k)
        return self._P[k]

    def lik(self, code: int, w: float) -> np.ndarray:
        if code == O["rpc"] and w < 1:
            return w * self.B[:, O["rpc"]] + (1 - w) * self.B[:, O["third_party"]]
        return self.B[:, code]

    @staticmethod
    def sequences(obs: pd.DataFrame, links: pd.DataFrame) -> list[dict]:
        """links: link, source, t0 (days), optional gold_status."""
        by = {k: g for k, g in obs.groupby("link", sort=False)}
        seqs = []
        for r in links.itertuples(index=False):
            g = by.get(r.link)
            seqs.append({"link": r.link, "source": r.source, "t0": r.t0,
                         "t": g.t.to_numpy() if g is not None else np.array([]),
                         "code": g.code.to_numpy() if g is not None else np.array([], int),
                         "w": g.w.to_numpy() if g is not None else np.array([]),
                         "gold": getattr(r, "gold_status", None)})
        return seqs

    def _mask(self, gold) -> np.ndarray:
        m = np.zeros(S)
        for s in GOLD_GROUPS[gold]:
            m[STATES.index(s)] = 1
        return m

    # ---------- inference ----------
    def forward_backward(self, seq, t_end: float):
        times = np.concatenate([[seq["t0"]], seq["t"], [t_end]])
        L = [np.ones(S)] + [self.lik(c, w) for c, w in zip(seq["code"], seq["w"])]
        L.append(self._mask(seq["gold"]) if isinstance(seq["gold"], str) else np.ones(S))
        n = len(times)
        alpha, Ps, logl = np.zeros((n, S)), [None] * n, 0.0
        a = self.pi[seq["source"]] * L[0]
        for i in range(n):
            if i > 0:
                Ps[i] = self.P(times[i] - times[i - 1])
                a = (alpha[i - 1] @ Ps[i]) * L[i]
            c = a.sum()
            alpha[i] = a / c
            logl += np.log(c)
        beta = np.ones((n, S))
        for i in range(n - 2, -1, -1):
            b = Ps[i + 1] @ (L[i + 1] * beta[i + 1])
            beta[i] = b / b.sum()
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        return gamma, logl

    def fit(self, seqs, t_end: float, n_iter: int = 25, tol: float = 1e-3, verbose=False):
        prev = -np.inf
        for it in range(n_iter):
            Bc = self.smoothing * self.B.copy()
            pic = {s: 10.0 * self.pi[s].copy() for s in SOURCES}
            total = 0.0
            for seq in seqs:
                gamma, ll = self.forward_backward(seq, t_end)
                total += ll
                pic[seq["source"]] += gamma[0]
                for j, (c, w) in enumerate(zip(seq["code"], seq["w"])):
                    g = gamma[j + 1]
                    if c == O["rpc"] and w < 1:
                        a = w * self.B[:, O["rpc"]]
                        b = (1 - w) * self.B[:, O["third_party"]]
                        share = a / (a + b)
                        Bc[:, O["rpc"]] += g * share
                        Bc[:, O["third_party"]] += g * (1 - share)
                    else:
                        Bc[:, c] += g
            self.B = Bc / Bc.sum(1, keepdims=True)
            self.pi = {s: v / v.sum() for s, v in pic.items()}
            if verbose:
                print(f"  EM iter {it + 1}: log-lik {total:,.1f}")
            if total - prev < tol * abs(total):
                break
            prev = total
        return self

    def filter_at(self, seq, when: list[float]) -> np.ndarray:
        """State probabilities at each time in `when`, using only calls strictly before it."""
        out = np.full((len(when), S), np.nan)
        a, t_prev, j = self.pi[seq["source"]].copy(), seq["t0"], 0
        for k, tq in enumerate(when):
            if tq < seq["t0"]:
                continue
            while j < len(seq["t"]) and seq["t"][j] < tq:
                a = (a @ self.P(seq["t"][j] - t_prev)) * self.lik(seq["code"][j], seq["w"][j])
                a /= a.sum()
                t_prev = seq["t"][j]
                j += 1
            out[k] = a @ self.P(tq - t_prev)
        return out


def group_probs(p: np.ndarray) -> pd.DataFrame:
    """Collapse state probabilities onto the five verified statuses."""
    return pd.DataFrame({g: p[:, [STATES.index(s) for s in ss]].sum(1) for g, ss in GOLD_GROUPS.items()})


def factor_columns(p: np.ndarray) -> pd.DataFrame:
    """The three factors (and the full state mix) as feature columns."""
    st = {s: p[:, i] for i, s in enumerate(STATES)}
    alive = st["reachable"] + st["avoiding"] + st["recycled"] + st["third_party"]
    owned = st["reachable"] + st["avoiding"]
    return pd.DataFrame({**{f"p_{s}": v for s, v in st.items()},
                         "p_alive": alive, "p_owned_given_alive": owned / np.clip(alive, 1e-9, None),
                         "p_avoid_given_owned": st["avoiding"] / np.clip(owned, 1e-9, None)})


VERIFY_T = float((VERIFIED_DATE - WINDOW_START).days)
