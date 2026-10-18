"""Decision ledger: one row per payment decision, safe to rerun.

* `decision_id` is a UUIDv5 of the payment id, so a rerun produces the same
  id and a downstream release system can deduplicate on it.
* `upsert` keeps the first decision for a payment and drops duplicates inside
  the new batch too, so a payment can never get two release requests.
* Both holds and releases are logged, so monitoring sees every decision.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

NAMESPACE = uuid.UUID("6f1c2d3e-4b5a-4c7d-8e9f-a0b1c2d3e4f5")

LEDGER_COLUMNS = ["decision_id", "payment_id", "account_id", "decided_on", "model_version",
                  "score", "cutoff", "node", "released"]


def decision_id(payment_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, str(payment_id)))


def empty_ledger() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in LEDGER_COLUMNS})


def to_ledger_rows(decisions: pd.DataFrame, decided_on, model_version: str, cutoff: int) -> pd.DataFrame:
    rows = pd.DataFrame({
        "decision_id": decisions["payment_id"].map(decision_id),
        "payment_id": decisions["payment_id"],
        "account_id": decisions["account_id"],
        "decided_on": pd.Timestamp(decided_on).normalize(),
        "model_version": model_version,
        "score": decisions["score"],
        "cutoff": cutoff,
        "node": decisions["node"],
        "released": decisions["released"].astype(bool),
    })
    return rows[LEDGER_COLUMNS]


def upsert(ledger: pd.DataFrame, new_rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Insert decisions for payments not already in the ledger. Returns (ledger, n_inserted)."""
    new_rows = new_rows.drop_duplicates("payment_id", keep="first")
    fresh = new_rows[~new_rows["payment_id"].isin(ledger["payment_id"])]
    if ledger.empty:
        return fresh.reset_index(drop=True), len(fresh)
    return pd.concat([ledger, fresh], ignore_index=True), len(fresh)


def save(ledger: pd.DataFrame, path: Path) -> None:
    assert ledger["payment_id"].is_unique, "ledger must have one row per payment"
    ledger.to_parquet(path, index=False)
