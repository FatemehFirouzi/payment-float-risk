"""Point-in-time feature engineering.

Every feature for a payment uses only information available strictly before
that payment's timestamp. Window counts are computed with `merge_asof` over
per-account cumulative event counts, so the whole table builds in a few
vectorised passes instead of a loop over payments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_FEATURES: list[str] = [
    # payment
    "amount", "log_amount", "amount_to_balance", "amount_to_line", "amount_to_min_due",
    "is_min_payment", "is_full_balance", "is_check", "channel_web", "channel_phone",
    "hour", "is_night", "day_of_week",
    # account
    "tenure_days", "utilization", "credit_line", "delinquent", "status_known",
    # funding account
    "funding_age_days", "funding_first_use", "n_funding_linked_90d",
    # velocity
    "n_payments_7d", "n_payments_30d", "amount_sum_7d",
    # prior returns
    "prior_returns_90d", "prior_returns_180d", "days_since_last_return",
]

DELINQUENT_STATUSES = ("PAST_DUE_30", "PAST_DUE_60")


def count_prior_events(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    time_col: str,
    event_time_col: str,
    window_days: int,
    value_col: str | None = None,
    key: str = "account_id",
) -> np.ndarray:
    """Count (or sum `value_col` of) events in [t - window, t) for each row of df.

    Strictly before t: an event at exactly the payment time is not counted,
    which is what makes the feature safe to compute in batch.
    """
    ev = events[[key, event_time_col] + ([value_col] if value_col else [])].dropna(subset=[event_time_col])
    ev = ev.sort_values(event_time_col).copy()
    ev["_v"] = ev[value_col].astype(float) if value_col else 1.0
    ev["_cum"] = ev.groupby(key)["_v"].cumsum()
    ev = ev[[key, event_time_col, "_cum"]]

    left = pd.DataFrame({"_row": np.arange(len(df)), key: df[key].to_numpy(), "_t": df[time_col].to_numpy()})
    left = left.sort_values("_t")

    def cum_before(t: pd.Series) -> np.ndarray:
        l2 = left.assign(_q=t.to_numpy()).sort_values("_q")
        m = pd.merge_asof(l2, ev, left_on="_q", right_on=event_time_col, by=key,
                          direction="backward", allow_exact_matches=False)
        out = np.zeros(len(df))
        out[m["_row"].to_numpy()] = m["_cum"].fillna(0.0).to_numpy()
        return out

    upper = cum_before(left["_t"])
    lower = cum_before(left["_t"] - pd.Timedelta(days=window_days))
    return upper - lower


def last_event_before(df, events, *, time_col, event_time_col, key="account_id") -> np.ndarray:
    """Timestamp of the most recent event strictly before each row's time (NaT if none)."""
    ev = events[[key, event_time_col]].dropna().sort_values(event_time_col)
    ev = ev.assign(_last=ev[event_time_col])
    left = pd.DataFrame({"_row": np.arange(len(df)), key: df[key].to_numpy(), "_t": df[time_col].to_numpy()})
    m = pd.merge_asof(left.sort_values("_t"), ev[[key, event_time_col, "_last"]], left_on="_t",
                      right_on=event_time_col, by=key, direction="backward", allow_exact_matches=False)
    out = np.full(len(df), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    out[m["_row"].to_numpy()] = m["_last"].to_numpy(dtype="datetime64[ns]")
    return out


def build_features(
    payments: pd.DataFrame,
    accounts: pd.DataFrame,
    funding: pd.DataFrame,
    account_status: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    df = payments.sort_values("initiated_at").reset_index(drop=True)
    df = df.merge(accounts[["account_id", "open_date", "credit_line"]], on="account_id", how="left")
    df = df.merge(funding[["funding_account_id", "linked_at"]], on="funding_account_id", how="left")
    t = df["initiated_at"]

    # payment
    df["log_amount"] = np.log1p(df["amount"])
    df["amount_to_balance"] = df["amount"] / df["statement_balance"].clip(lower=1.0)
    df["amount_to_line"] = df["amount"] / df["credit_line"]
    df["amount_to_min_due"] = df["amount"] / df["minimum_due"].clip(lower=1.0)
    df["is_min_payment"] = (df["amount"] <= df["minimum_due"] + 0.01).astype(int)
    df["is_full_balance"] = (df["amount"] >= df["statement_balance"] - 0.01).astype(int)
    df["is_check"] = (df["payment_type"] == "CHECK").astype(int)
    df["channel_web"] = (df["channel"] == "WEB").astype(int)
    df["channel_phone"] = (df["channel"] == "PHONE").astype(int)
    df["hour"] = t.dt.hour
    df["is_night"] = (df["hour"] < 5).astype(int)
    df["day_of_week"] = t.dt.dayofweek

    # account
    df["tenure_days"] = (t - df["open_date"]).dt.days
    df["utilization"] = df["statement_balance"] / df["credit_line"]
    st = account_status.sort_values("as_of_date")
    tmp = pd.merge_asof(
        pd.DataFrame({"_row": np.arange(len(df)), "account_id": df["account_id"], "_t": t}).sort_values("_t"),
        st, left_on="_t", right_on="as_of_date", by="account_id", direction="backward",
    ).sort_values("_row")
    df["delinquency_status"] = tmp["delinquency_status"].to_numpy()
    df["status_known"] = df["delinquency_status"].notna().astype(int)
    df["delinquent"] = df["delinquency_status"].isin(DELINQUENT_STATUSES).astype(int)

    # funding account
    df["funding_age_days"] = (t - df["linked_at"]).dt.total_seconds() / 86400.0
    df["funding_first_use"] = (df.groupby("funding_account_id").cumcount() == 0).astype(int)
    df["n_funding_linked_90d"] = count_prior_events(
        df, funding, time_col="initiated_at", event_time_col="linked_at", window_days=90)

    # velocity (prior payments only; the current one is excluded)
    df["n_payments_7d"] = count_prior_events(
        df, payments, time_col="initiated_at", event_time_col="initiated_at", window_days=7)
    df["n_payments_30d"] = count_prior_events(
        df, payments, time_col="initiated_at", event_time_col="initiated_at", window_days=30)
    df["amount_sum_7d"] = count_prior_events(
        df, payments, time_col="initiated_at", event_time_col="initiated_at", window_days=7, value_col="amount")

    # prior returns: known only once the return has happened
    ret = returns.merge(payments[["payment_id", "account_id"]], on="payment_id", how="left")
    for d in (90, 180):
        df[f"prior_returns_{d}d"] = count_prior_events(
            df, ret, time_col="initiated_at", event_time_col="returned_at", window_days=d)
    last = last_event_before(df, ret, time_col="initiated_at", event_time_col="returned_at")
    df["days_since_last_return"] = (t.to_numpy() - last) / np.timedelta64(1, "D")

    return df


def add_label(df: pd.DataFrame, returns: pd.DataFrame, as_of: pd.Timestamp, window_days: int) -> pd.DataFrame:
    """bad = returned within `window_days` of the payment. NaN if not yet matured."""
    end = as_of + pd.Timedelta(days=1)
    out = df.drop(columns=[c for c in ("returned_at", "return_code") if c in df.columns])
    out = out.merge(returns[["payment_id", "returned_at", "return_code"]], on="payment_id", how="left")
    days = (out["returned_at"] - out["initiated_at"]).dt.total_seconds() / 86400.0
    bad = ((days <= window_days) & (out["returned_at"] <= end)).astype(float)
    matured = out["initiated_at"] + pd.Timedelta(days=window_days) <= end
    out["label"] = bad.where(matured)
    return out
