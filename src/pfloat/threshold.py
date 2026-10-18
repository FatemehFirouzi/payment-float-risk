"""Threshold trade-off: how much to release, at what return cost.

For every candidate cutoff on payments that pass the Node 1 rules:

    released     = score <= cutoff
    value        = value_per_release x good payments released
    loss         = sum(amount x spend_share x (1 - recovery)) over returned, released payments
    net          = value - loss

The recommended cutoff maximises net value. Returns are observed whether or
not a payment was held, so this can be evaluated on any matured period
without reject inference; only the spend and recovery assumptions are modelled.
"""

from __future__ import annotations

import pandas as pd

from .strategy import node1_hold


def tradeoff(
    df: pd.DataFrame,
    strategy: dict,
    economics: dict,
    label_col: str = "label",
    spend_col: str | None = None,
    recovery_col: str | None = None,
) -> pd.DataFrame:
    """One row per candidate cutoff (each distinct score among eligible payments)."""
    d = df[df[label_col].notna() & df["score"].notna()].copy()
    total = len(d)
    d = d[~node1_hold(d, strategy)]
    if d.empty:
        return pd.DataFrame()

    spend = d[spend_col] if spend_col else economics["spend_share"]
    recovery = d[recovery_col] if recovery_col else economics["recovery_rate"]
    d["loss_if_released"] = d["amount"] * spend * (1 - recovery) * d[label_col]
    d["bad_amount"] = d["amount"] * d[label_col]

    g = d.groupby("score").agg(n=(label_col, "size"), bads=(label_col, "sum"),
                               bad_amount=("bad_amount", "sum"), loss=("loss_if_released", "sum"))
    g = g.sort_index()
    c = g.cumsum()
    out = pd.DataFrame({
        "cutoff": g.index.astype(int),
        "released": c["n"].to_numpy(),
        "release_rate_eligible": (c["n"] / len(d)).to_numpy(),
        "release_rate_all": (c["n"] / total).to_numpy(),
        "returns_released": c["bads"].to_numpy(),
        "return_rate_released": (c["bads"] / c["n"]).to_numpy(),
        "returned_amount_released": c["bad_amount"].to_numpy(),
        "band_n": g["n"].to_numpy(),
        "band_return_rate": (g["bads"] / g["n"]).to_numpy(),
        "loss": c["loss"].to_numpy(),
    })
    out["value"] = economics["value_per_release"] * (out["released"] - out["returns_released"])
    out["net_value"] = out["value"] - out["loss"]
    return out


def recommend(curve: pd.DataFrame) -> int:
    return int(curve.loc[curve["net_value"].idxmax(), "cutoff"])
