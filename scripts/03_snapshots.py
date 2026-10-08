"""Stage 3: as-of snapshots, labels and the attempt-level modelling table.
Run from the repo root (after Stage 2):  python scripts/03_snapshots.py
Writes data/processed/{snapshots, scoring_snapshot, link_week, attempts_model}.parquet
and reports/03_snapshots.md
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from rpc.config import PROCESSED, REPORTS
from rpc.io import load_all, add_flags
from rpc.features import enrich_attempts, build_snapshot
from rpc.labels import weekly_cutoffs, SCORING_CUTOFF, build_snapshots, link_week_labels, attempt_table

pd.set_option("display.width", 200)
out = []


def section(title, body):
    text = body if isinstance(body, str) else body.to_string()
    out.append(f"## {title}\n\n```\n{text}\n```\n")
    print(f"\n=== {title}\n{text}")


t0 = time.time()
t = load_all()
E = enrich_attempts(add_flags(t["dial_attempts"]),
                    pd.read_parquet(PROCESSED / "remarks_dial.parquet"),
                    pd.read_parquet(PROCESSED / "attempt_trust.parquet"))
rf = pd.read_parquet(PROCESSED / "remarks_field.parquet")
cutoffs = weekly_cutoffs()

snaps = build_snapshots(cutoffs, E, t, rf)
scoring = build_snapshot(SCORING_CUTOFF, E, t, rf)
lw = link_week_labels(E, cutoffs)
snaps = snaps.merge(lw, on=["cutoff", "link"], how="left")
snaps["next_att"] = snaps.next_att.fillna(0)
A = attempt_table(E, snaps, t["agents"], cutoffs)

snaps.to_parquet(PROCESSED / "snapshots.parquet", index=False)
scoring.to_parquet(PROCESSED / "scoring_snapshot.parquet", index=False)
A.to_parquet(PROCESSED / "attempts_model.parquet", index=False)

section("1. Sizes", "\n".join([
    f"cutoffs: {len(cutoffs)} weekly, {cutoffs[0]:%d %b} to {cutoffs[-1]:%d %b}; scoring cutoff {SCORING_CUTOFF:%d %b}",
    f"snapshot rows: {len(snaps):,}  ({snaps.shape[1]} columns)",
    f"snapshot rows with a call in the next 7 days (labelled): {(snaps.next_att > 0).sum():,}",
    f"attempt rows: {len(A):,}  (dropped, number added mid-week: {A.attrs['dropped_no_snapshot']:,})",
    f"scoring snapshot rows: {len(scoring):,}",
    f"built in {time.time() - t0:.0f}s"]))

lab = snaps[snaps.next_att > 0]
section("2. Next-7-day borrower contact rate by split (labelled rows)",
        lab.groupby("split").agg(rows=("link", "size"), rpc_any=("next_rpc_any", "mean"),
                                 rpc_soft=("next_rpc_soft", "mean")).round(3))
section("3. Three factors at attempt level", pd.Series({
    "P(alive)": A.y_alive.mean(), "P(picked | alive)": A[A.y_alive].y_picked.mean(),
    "P(answered | picked)": A[A.y_picked].y_answered.mean(),
    "P(borrower | answered)": A[A.y_answered].y_rpc_soft.mean()}).round(3))

# Do the features carry signal? (train split only)
tr = lab[lab.split == "train"]
bucket = lambda s, bins, labels: pd.cut(s, bins=bins, labels=labels, include_lowest=True)
section("4. Signal check: next-week contact rate by dead streak (train)",
        tr.groupby(bucket(tr.dead_streak, [-1, 0, 2, 5, 1000], ["0", "1-2", "3-5", "6+"]), observed=True)
        .next_rpc_any.agg(["size", "mean"]).round(3))
section("5. Signal check: next-week contact rate by past reject rate (train, 3+ calls)",
        tr[tr.n_att >= 3].groupby(bucket(tr[tr.n_att >= 3].rejected_rate, [-.01, 0, .2, .5, 1],
                                         ["0", "0-20%", "20-50%", "50%+"]), observed=True)
        .next_rpc_any.agg(["size", "mean"]).round(3))
section("6. Signal check: next-week contact rate by source (train)",
        tr.groupby("source").next_rpc_any.agg(["size", "mean"]).round(3))
section("7. Cold start: share of labelled rows with no prior calls on that number",
        f"{(lab.n_att == 0).mean():.1%}")

REPORTS.mkdir(exist_ok=True)
(REPORTS / "03_snapshots.md").write_text("# Stage 3: snapshots and labels\n\n" + "\n".join(out), encoding="utf-8")
print(f"\nwrote outputs to {PROCESSED}")
