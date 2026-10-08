from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

WINDOW_START = pd.Timestamp("2026-04-01")
WINDOW_END = pd.Timestamp("2026-06-30")       # exclusive
VERIFIED_DATE = pd.Timestamp("2026-07-02")
CONTACT_HOURS = (8, 19)                         # RBI: calls only 08:00 to 19:00

SPLITS = ("train", "validation", "test")

# Fully loaded cost per action in rupees (CN problem statement ranges; trace from data)
COSTS = {"voice_bot": 2.0, "tele_agent": 22.0, "field_visit": 275.0, "skip_trace": 104.0}

# Disposition groups
RPC_DISPOSITIONS = {"rpc_ptp", "rpc_hung_up", "rpc_call_back", "rpc_refused",
                    "rpc_hardship", "rpc_dispute", "rpc_claims_paid"}
THIRD_PARTY_DISPOSITIONS = {"third_party_contact", "third_party_ptp"}
DEAD_RESPONSES = {"switched_off", "not_reachable", "number_does_not_exist"}

GOLD_STATUSES = ["borrower_number", "third_party_number", "not_borrower_number",
                 "switched_off", "invalid_number"]
