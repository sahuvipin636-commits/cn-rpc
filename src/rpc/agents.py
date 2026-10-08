"""Stage 2b: how far to trust each agent's "RPC" disposition.

Evidence used (train-split gold only, plus logs):
  1. On verified NON-borrower numbers, how often an agent's answered call was logged as RPC.
     Shrunk toward the pooled rate, because some agents have only a handful of such calls.
  2. RPCs logged on calls that were never answered (impossible, so trust = 0).

Not used: peer comparison on shared numbers. Checked in Stage 2 and it disagreed with the
gold labels (agents get different call mixes and hours), so it would add bias, not signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIOR_STRENGTH = 10.0      # pseudo-calls pulling each agent toward the pooled rate
FLAG_TRUST_BELOW = 0.85


def agent_reliability(att: pd.DataFrame, gold_train: pd.DataFrame) -> pd.DataFrame:
    """One row per agent. `att` must have add_flags() columns."""
    g = att[att.answered].merge(gold_train[["account_id", "phone_id", "verified_status"]],
                                on=["account_id", "phone_id"])
    nb = g[g.verified_status != "borrower_number"]
    k = nb.groupby("agent_id").logged_rpc.sum()
    n = nb.groupby("agent_id").size()
    q0 = k.sum() / n.sum()                                   # pooled "RPC on non-borrower" rate

    rel = pd.DataFrame(index=sorted(att.agent_id.unique()))
    rel.index.name = "agent_id"
    rel["attempts"] = att.groupby("agent_id").size()
    rel["logged_rpc_rate"] = att.groupby("agent_id").logged_rpc.mean()
    rel["gold_nonborrower_answered"] = n.reindex(rel.index).fillna(0).astype(int)
    rel["gold_nonborrower_logged_rpc"] = k.reindex(rel.index).fillna(0).astype(int)
    a0, b0 = q0 * PRIOR_STRENGTH, (1 - q0) * PRIOR_STRENGTH
    rel["p_rpc_on_nonborrower"] = (rel.gold_nonborrower_logged_rpc + a0) / (rel.gold_nonborrower_answered + a0 + b0)
    rel["rpc_on_unanswered"] = att.groupby("agent_id").impossible_rpc.sum().reindex(rel.index).fillna(0).astype(int)

    # Trust relative to a typical agent: 1.0 = as reliable as the pool (or better)
    rel["rpc_trust"] = ((1 - rel.p_rpc_on_nonborrower) / (1 - q0)).clip(0, 1)
    rel["flag"] = np.select(
        [rel.rpc_on_unanswered > 0, rel.rpc_trust < FLAG_TRUST_BELOW],
        ["logs RPC on unanswered calls", "often logs non-borrowers as RPC"], "")
    rel.attrs["pooled_rate"] = q0
    return rel.reset_index()


def attempt_rpc_trust(att: pd.DataFrame, rel: pd.DataFrame) -> pd.Series:
    """Probability-style weight that a logged RPC really reached the borrower.
    NaN for attempts not logged as RPC."""
    trust = att.agent_id.map(rel.set_index("agent_id").rpc_trust).astype(float)
    out = pd.Series(np.nan, index=att.index)
    out[att.logged_rpc] = trust[att.logged_rpc]
    out[att.impossible_rpc] = 0.0
    return out
