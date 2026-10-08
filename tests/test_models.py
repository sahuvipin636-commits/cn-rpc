import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rpc.models import feature_columns  # noqa: E402


def test_labels_never_become_features():
    df = pd.DataFrame(columns=["hour", "p_alive", "n_att", "y_rpc_soft", "y_alive", "next_rpc_any",
                               "next_att", "attempt_id", "split", "propensity", "agent_id"])
    cols = feature_columns(df)
    assert cols == ["hour", "p_alive", "n_att"]
