# Right-party contact prediction and skip-trace prioritisation (CreditNirvana PS2)

Scores every phone number on file for how likely it is to reach the borrower, explains why,
and turns that into a daily plan: which number to call and when, which accounts to visit or
skip-trace, and which numbers must never be called.

## Results on the untouched test split

| Metric | Result |
|---|---|
| Right-party contacts per 1,000 dials (randomised arm) | 153 vs 135 rule engine (90% CI for lift: -12 to +47) |
| Skip-trace spend, 243 out-of-sample traces | Rs 2,059 vs Rs 25,029 (-92%) |
| Dials on numbers the system would stop | 8.2% of dials, borrower reached 2.6% vs 17.0% overall |
| Strangers reached without identity check | 33 vs 50 per 1,000 dials |
| Verified number state accuracy | 89% vs 75% source guess, 67% last-call rule |
| Verified borrower numbers blocked | 0 of 16 |

Full numbers: `reports/scorecard.md` after running Stage 7.

## Run it

Put CN's CSVs in `data/raw/`, then from this folder:

```
pip install -r requirements.txt
python scripts/01_audit.py
python scripts/02_remarks_agents.py
python scripts/03_snapshots.py
python scripts/04_states.py
python scripts/05_models.py
python scripts/06_decide.py
python scripts/07_evaluate.py
python -m pytest -q tests
streamlit run app/streamlit_app.py
```

## Layout

| Path | What it does |
|---|---|
| `src/rpc/io.py` | Loads and validates data; guards the verified labels |
| `src/rpc/remarks.py` | Parses Hinglish, Kanglish and English remarks |
| `src/rpc/agents.py` | Trust weight for each agent's RPC dispositions |
| `src/rpc/features.py`, `labels.py` | As-of snapshots and labels |
| `src/rpc/states.py` | Continuous-time hidden-state model per number |
| `src/rpc/models.py` | RPC model: alive x answer x owned, best hour |
| `src/rpc/safety.py` | Compliance statuses before any value decision |
| `src/rpc/decide.py` | Rupee-value decisions, reason codes, capacity |
| `src/rpc/evaluate.py` | One-time test evaluation (only module allowed to read test labels) |
| `app/streamlit_app.py` | Supervisor console |
| `AGENT_CONTEXT.md`, `SCHEMA.md` | Design notes and data schema |
