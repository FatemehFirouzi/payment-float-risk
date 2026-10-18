"""The release strategy: hard policy rules first, then the score cutoff.

Decisioning and threshold analysis use this Python implementation as the
single source of truth. Monitoring independently re-derives the policy in SQL
as a reconciliation control; any difference is surfaced as a policy mismatch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NODE_UNSCORED = "Hold - unscored"
NODE_STATUS_UNKNOWN = "Node 1 - account status unknown"
NODE_DELINQUENT = "Node 1 - delinquent"
NODE_PRIOR_RETURN = "Node 1 - prior return"
NODE_AMOUNT = "Node 1 - amount at or over cap"
NODE_SCORE = "Node 2 - score over cutoff"
NODE_RELEASE = "Release"

NODE_ORDER = [NODE_UNSCORED, NODE_STATUS_UNKNOWN, NODE_DELINQUENT, NODE_PRIOR_RETURN,
              NODE_AMOUNT, NODE_SCORE, NODE_RELEASE]


def node1_hold(df: pd.DataFrame, strategy: dict) -> pd.Series:
    """True where a Node 1 rule keeps the hold regardless of score."""
    prior = df[f"prior_returns_{int(strategy['prior_return_days'])}d"]
    status = df["delinquency_status"]
    return (
        status.isna()
        | ~status.isin(strategy["good_status"])
        | (prior > 0)
        | (df["amount"] >= strategy["amount_cap"])
    )


def assign_node(df: pd.DataFrame, cutoff: int, strategy: dict) -> pd.Series:
    """First matching rule wins. Release when score <= cutoff (inclusive)."""
    prior_col = f"prior_returns_{int(strategy['prior_return_days'])}d"
    if prior_col not in df.columns:
        raise ValueError(f"strategy needs feature {prior_col}")
    status = df["delinquency_status"]
    score = df["score"]
    conditions = [
        score.isna(),
        status.isna(),
        ~status.isin(strategy["good_status"]),
        df[prior_col] > 0,
        df["amount"] >= strategy["amount_cap"],
        score > cutoff,
    ]
    choices = NODE_ORDER[:-1]
    return pd.Series(np.select(conditions, choices, default=NODE_RELEASE), index=df.index)


def decide(df: pd.DataFrame, cutoff: int, strategy: dict) -> pd.DataFrame:
    out = df.copy()
    out["node"] = assign_node(out, cutoff, strategy)
    out["released"] = out["node"] == NODE_RELEASE
    return out
