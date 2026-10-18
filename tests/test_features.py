"""Point-in-time correctness: no feature may see information from the future."""

import pandas as pd

from pfloat.features import add_label, count_prior_events

T = pd.Timestamp


def _pay():
    return pd.DataFrame({"account_id": [1, 1, 1], "initiated_at": [T("2025-03-01"), T("2025-03-10"), T("2025-06-15")]})


def test_return_after_the_payment_is_not_counted():
    returns = pd.DataFrame({"account_id": [1], "returned_at": [T("2025-03-05")]})
    got = count_prior_events(_pay(), returns, time_col="initiated_at", event_time_col="returned_at", window_days=90)
    # Mar 1 payment: return on Mar 5 is in the future -> 0. Mar 10: 1. Jun 15: Mar 5 is outside 90 days -> 0.
    assert list(got) == [0, 1, 0]


def test_event_at_exactly_the_payment_time_is_excluded():
    returns = pd.DataFrame({"account_id": [1], "returned_at": [T("2025-03-10")]})
    got = count_prior_events(_pay(), returns, time_col="initiated_at", event_time_col="returned_at", window_days=90)
    assert got[1] == 0


def test_other_accounts_do_not_leak():
    returns = pd.DataFrame({"account_id": [2], "returned_at": [T("2025-03-05")]})
    got = count_prior_events(_pay(), returns, time_col="initiated_at", event_time_col="returned_at", window_days=90)
    assert got.sum() == 0


def test_velocity_excludes_the_current_payment():
    pay = _pay()
    got = count_prior_events(pay, pay, time_col="initiated_at", event_time_col="initiated_at", window_days=30)
    assert list(got) == [0, 1, 0]


def test_label_is_missing_until_mature():
    df = pd.DataFrame({"payment_id": ["a", "b", "c"],
                       "initiated_at": [T("2025-09-01"), T("2025-09-01"), T("2025-12-20")]})
    returns = pd.DataFrame({"payment_id": ["a"], "returned_at": [T("2025-09-03")], "return_code": ["R01"]})
    out = add_label(df, returns, as_of=T("2025-12-31"), window_days=60).set_index("payment_id")
    assert out.loc["a", "label"] == 1 and out.loc["b", "label"] == 0
    assert pd.isna(out.loc["c", "label"])
