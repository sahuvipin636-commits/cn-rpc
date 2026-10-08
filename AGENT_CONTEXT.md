# AGENT_CONTEXT: CreditNirvana PS2, Right-Party Contact Prediction

Read this file and SCHEMA.md before writing any code.

## Goal
For every (account, phone) on file, predict how likely it is to produce a right-party
contact (RPC). A decision layer then picks, per account: keep dialling (at the best hour),
switch number, switch channel (field visit), suppress, or skip-trace. Skip-traces are
ranked by expected value instead of CN's "15 consecutive failed contacts" rule.

## Core idea: three factors
P(RPC) = P(Alive) x P(Owned | Alive) x P(Answers | Alive, Owned, hour)
- Alive low  -> retry later, other number, or trace
- Owned low  -> suppress (recycled / wrong) or locate-only (third party)
- Answers low -> borrower is avoiding: switch channel, never trace
The lowest factor is the reason code.

## Hidden states and the gold labels that check them
| Verified status (2 Jul)  | Model states                       |
|--------------------------|------------------------------------|
| borrower_number          | reachable, avoiding                |
| third_party_number       | third_party                        |
| not_borrower_number      | recycled, wrong_from_origination   |
| switched_off             | temp_unreachable, dead             |
| invalid_number           | invalid                            |

## Data facts that shape the design (from scripts/01_audit.py)
- Window 2026-04-01 to 2026-06-29. Gold verification on 2026-07-02.
- Accounts pre-split: train 1680, validation 360, test 360.
- Random exploration arm: 123 accounts, `selection_propensity` logged per attempt.
- Avoiding signal: busy_rejected and "2 ring ke baad disconnect" style remarks.
- Recycled signal: owned early, then dead responses, then strangers / wrong_number.
- Traces on rejecting accounts rarely find anything; on dead / third-party-only they do.
- Some agents log third parties as RPC; some RPCs are logged on unanswered calls.
- Salaried and daily-wage borrowers answer best at 8am and 6pm; incumbent dials mostly before 1pm.
- Remarks: Hinglish, Kanglish (romanised Kannada), English, with typos.
- Remarks mirror dispositions 99.9%, so they cannot expose mislogging. Their value is
  detail: new_owner vs not_known, long_off, who answered, evening callback, met at home/shop/work.
- Agent trust: shrunk gold-based rate (train only) + RPC-on-unanswered. Peer comparison was
  tested and rejected (disagrees with gold). Outputs: data/processed/agent_reliability.csv,
  attempt_trust.parquet (rpc_trust per logged RPC; 0 if call was unanswered).
- Stage 3 outputs (data/processed): snapshots.parquet (13 weekly cutoffs, 1 Apr to 24 Jun,
  122 cols, next-7-day labels), scoring_snapshot.parquet (as of 30 Jun, for the 2 Jul gold),
  attempts_model.parquet (each call joined to its week's features; y_alive, y_picked,
  y_answered, y_rpc_soft, y_tp, y_wrong; hour, is_bot, propensity).
  A number added on day D is known on day D. tests/test_asof.py proves no future leakage.
  Known assumption: agent trust is an agent-level parameter fitted on train gold.
- Stage 4: continuous-time state model (states.py), fitted on train accounts only, EM emissions,
  train gold anchors the 2 Jul state. Validation gold: 95% accuracy (41 numbers) vs 80% source
  prior, 56% last-call rule. Weakest split: recycled vs switched_off. Outputs:
  state_features.parquet (cutoff, link, p_<state>, p_alive, p_owned_given_alive,
  p_avoid_given_owned) for all 13 cutoffs + 30 Jun; state_model.json.
- Stage 5: RPCModel (models.py) = alive x answer x owned LightGBM heads (beats direct model
  slightly), features = Stage 3 snapshot + Stage 4 states + hour/dow/is_bot/lang_match.
  Validation: per-call AUC 0.685, per contact point next-7-day AUC 0.702, well calibrated raw
  (isotonic per portfolio tested, made log-loss worse, so apply_calibration=False). MFI is
  over-predicted on validation (0.140 vs 0.109). Random arm: no selection bias seen.
  Best hour mostly 08:00, then 18:00 (model estimate). Saved: data/processed/rpc_model.pkl.
  Label columns (y_*, next_*) are blocked from features by feature_columns() + test.
- Not in data: SMS delivery, WhatsApp, MNRL, payment-link events (production add-ons only).

## Stage roadmap
| Stage | Module(s) | Output | Status |
|---|---|---|---|
| 0 | repo, specs | AGENT_CONTEXT.md, SCHEMA.md | done |
| 1 | src/rpc/io.py, scripts/01_audit.py | validated loaders, data audit report | done |
| 2 | src/rpc/remarks.py, src/rpc/agents.py | remark tags (3 languages), agent reliability | done |
| 3 | src/rpc/features.py, src/rpc/labels.py | weekly as-of snapshots + labels | done |
| 4 | src/rpc/states.py | hidden-state filter, checked on gold (train/val) | done |
| 5 | src/rpc/models.py | RPC + best-hour models, IPS weights, calibration | done |
| 6 | src/rpc/safety.py, src/rpc/decide.py | suppression, actions, reason codes, trace EV | next |
| 7 | src/rpc/evaluate.py | AUC/calibration, gold accuracy, IPS policy value, trace replay | |
| 8 | addresses, app/streamlit_app.py, write-up | field extension, dashboard | |

## Non-negotiable rules
1. As-of discipline: a feature for cutoff date D uses only events with timestamp < D.
2. Split discipline: fit only on `train`; tune on `validation`; touch `test` once, in evaluate.py.
3. Gold labels are read only through `io.load_gold(splits=...)`. Only evaluate.py may
   request the test split. tests/test_guards.py enforces this.
4. Compliance is a hard filter, never a trade-off: contact hours 08:00 to 19:00,
   no debt discussion with third parties, suppression list always wins.
5. Every action carries a plain-language reason code.
6. Calibrate per portfolio.

## Conventions
- Python 3.10+, pandas, numpy, scipy, scikit-learn, lightgbm.
- Package code in src/rpc, entry points in scripts/ (run from repo root).
- All constants in src/rpc/config.py.
- Each script prints a short summary and writes any report to reports/.
