import numpy as np
import pandas as pd

from pfloat import ledger
from pfloat.strategy import (
    NODE_AMOUNT,
    NODE_DELINQUENT,
    NODE_PRIOR_RETURN,
    NODE_RELEASE,
    NODE_SCORE,
    NODE_STATUS_UNKNOWN,
    NODE_UNSCORED,
    assign_node,
    decide,
)

STRATEGY = {"amount_cap": 300, "prior_return_days": 180, "good_status": ["NOT_DUE", "CURRENT_DUE"]}


def _row(**kw):
    base = {"score": 10.0, "delinquency_status": "NOT_DUE", "prior_returns_180d": 0, "amount": 100.0}
    base.update(kw)
    return base


def nodes(rows, cutoff=20):
    return list(assign_node(pd.DataFrame(rows), cutoff, STRATEGY))


def test_score_cutoff_is_inclusive():
    assert nodes([_row(score=20), _row(score=21)]) == [NODE_RELEASE, NODE_SCORE]


def test_amount_cap_boundary_holds_at_cap():
    assert nodes([_row(amount=299.99), _row(amount=300.0)]) == [NODE_RELEASE, NODE_AMOUNT]


def test_rules_fire_in_order():
    rows = [
        _row(score=np.nan, delinquency_status="PAST_DUE_30"),   # unscored wins over everything
        _row(delinquency_status=None, amount=999),               # unknown status before amount
        _row(delinquency_status="PAST_DUE_60", prior_returns_180d=3),
        _row(prior_returns_180d=1, amount=999),
    ]
    assert nodes(rows) == [NODE_UNSCORED, NODE_STATUS_UNKNOWN, NODE_DELINQUENT, NODE_PRIOR_RETURN]


def test_decide_flags_only_release_node():
    out = decide(pd.DataFrame([_row(), _row(score=99)]), 20, STRATEGY)
    assert list(out["released"]) == [True, False]


def _decisions(ids):
    df = pd.DataFrame({"payment_id": ids, "account_id": 1, "score": 5.0, "node": NODE_RELEASE, "released": True})
    return ledger.to_ledger_rows(df, "2025-09-01", "v1", 20)


def test_decision_id_is_deterministic():
    assert ledger.decision_id("P1") == ledger.decision_id("P1") != ledger.decision_id("P2")


def test_upsert_is_idempotent_and_dedupes_the_batch():
    led, n1 = ledger.upsert(ledger.empty_ledger(), _decisions(["P1", "P2", "P2"]))
    led, n2 = ledger.upsert(led, _decisions(["P1", "P2", "P3"]))  # rerun plus one new payment
    assert (n1, n2) == (2, 1)
    assert led["payment_id"].is_unique and len(led) == 3
