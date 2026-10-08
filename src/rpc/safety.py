"""Stage 6a: compliance and safety rules, applied BEFORE any value-based decision.

Every number gets one status:
  dial              may be called about the debt
  verify_first      may be called, but identity must be confirmed before the debt is mentioned
                    (the model is unsure whether the borrower still holds this number)
  suppress_recycled likely a new subscriber: never call (debt disclosure risk under FPC)
  locate_only       reaches a relative / colleague: only an FPC-safe "please ask them to call us"
                    script, never the debt, and never counted as a borrower channel
  drop_invalid      number does not exist
  pause_dead        switched off or unreachable for a long time: stop dialling, re-check later
Thresholds are tuned on train-split gold for high recall on the risky classes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULTS = {"recycled_low": 0.15, "recycled_high": 0.50, "third_party_low": 0.15, "third_party_high": 0.50,
            "invalid": 0.50, "dead": 0.70}
RECALL_TARGET = 0.90      # low tier: catch 90% of risky numbers (verify first)
PRECISION_TARGET = 0.90   # high tier: suppress only when 90% sure


def tune_threshold(p: np.ndarray, is_pos: np.ndarray, target: float = RECALL_TARGET) -> float:
    """Highest threshold that still catches `target` share of the positives."""
    pos = np.sort(p[is_pos])[::-1]
    if len(pos) == 0:
        return 0.5
    k = int(np.ceil(target * len(pos))) - 1
    return float(np.clip(pos[k], 0.05, 0.95))


def tune_precision(p: np.ndarray, is_pos: np.ndarray, target: float = PRECISION_TARGET,
                   min_count: int = 5) -> float:
    """Lowest threshold whose flagged set is at least `target` precise (needs min_count flagged)."""
    for th in np.round(np.arange(0.05, 0.96, 0.01), 2):
        flagged = p >= th
        if flagged.sum() >= min_count and is_pos[flagged].mean() >= target:
            return float(th)
    return 0.95


def rule_flags(n: pd.DataFrame) -> pd.DataFrame:
    """Evidence-based triggers that do not depend on the model."""
    stranger_since_borrower = (n.get("wrong_after_rpc", 0) > 0) & (
        n.days_since_wrong.fillna(np.inf) < n.days_since_rpc.fillna(np.inf))
    return pd.DataFrame({
        "new_owner_said": n.get("new_owner_after_rpc", 0) > 0,
        "stranger_since_borrower": stranger_since_borrower,
        "does_not_exist_seen": n.get("n_does_not_exist", 0) > 0,
    }, index=n.index)


def apply_safety(n: pd.DataFrame, th: dict = DEFAULTS) -> pd.DataFrame:
    """n: one row per (account, phone) with state probabilities and history features."""
    f = rule_flags(n)
    dead = n.p_dead + n.p_temp_off
    # hard suppression needs an explicit statement or high model confidence; a single stranger
    # answering (often a relative, or an agent's mislog) only forces identity verification
    recycled = (n.p_recycled >= th["recycled_high"]) | f.new_owner_said
    invalid = (n.p_invalid >= th["invalid"]) | f.does_not_exist_seen
    third = n.p_third_party >= th["third_party_high"]
    unsure = (n.p_recycled >= th["recycled_low"]) | (n.p_third_party >= th["third_party_low"]) | \
        f.stranger_since_borrower
    status = np.select([invalid, recycled, third, dead >= th["dead"], unsure],
                       ["drop_invalid", "suppress_recycled", "locate_only", "pause_dead", "verify_first"], "dial")
    why = np.select(
        [invalid & f.does_not_exist_seen, invalid,
         recycled & f.new_owner_said, recycled,
         third, dead >= th["dead"], unsure & f.stranger_since_borrower, unsure],
        ["network says the number does not exist",
         "model: " + n.p_invalid.mul(100).round().astype(int).astype(str) + "% likely invalid",
         "person answering said it is their new number",
         "model: " + n.p_recycled.mul(100).round().astype(int).astype(str) + "% likely reassigned to a new owner",
         "model: " + n.p_third_party.mul(100).round().astype(int).astype(str) + "% likely a relative or colleague",
         "switched off / unreachable: " + n.dead_streak.fillna(0).astype(int).astype(str) + " dead responses in a row",
         "confirm identity before discussing the debt: a stranger answered after the borrower was last reached",
         "confirm identity before discussing the debt: " + (n.p_recycled + n.p_third_party).mul(100).round()
         .astype(int).astype(str) + "% chance someone else holds this number"],
        "")
    return pd.DataFrame({"status": status, "safety_reason": why}, index=n.index)
