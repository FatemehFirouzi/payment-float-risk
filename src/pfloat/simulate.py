"""Synthetic card-payment world.

Generates accounts, linked funding (bank) accounts, monthly delinquency
snapshots, ACH/check card payments and their returns. No real data is used.

The generator is built so that the hard parts of the real problem show up:

* Returns are rare (a few percent) and arrive late: R01/R02/R03 within days,
  R10 "unauthorized" returns up to 60 days later, so recent payments look
  clean until they mature.
* A small population of bust-out accounts pays with bad funds from freshly
  linked bank accounts and spends the released credit almost immediately.
* From `drift_start` a wave of dormant "sleeper" accounts wakes up and kites:
  a burst of payments just under the illustrative $300 cap, from aged bank accounts, via
  the mobile app, on accounts with clean history. It hides the signals the
  model and the rules rely on. That is the drift monitoring has to catch.

Observable tables (what a risk team would have) go to data/. The hidden
truth used only for evaluation and loss accounting goes to data/truth/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

DAY = pd.Timedelta(days=1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _uniform_dates(rng, lo: pd.Series | pd.Timestamp, hi: pd.Series | pd.Timestamp, n: int) -> pd.Series:
    lo = pd.to_datetime(pd.Series(lo)) if not isinstance(lo, pd.Timestamp) else lo
    hi = pd.to_datetime(pd.Series(hi)) if not isinstance(hi, pd.Timestamp) else hi
    span = (hi - lo) / pd.Timedelta(seconds=1)
    offset = rng.random(n) * np.asarray(span, dtype=float)
    out = lo + pd.to_timedelta(offset, unit="s")
    return pd.Series(np.asarray(out, dtype="datetime64[ns]"))


def simulate(cfg: Config) -> dict[str, pd.DataFrame]:
    s = cfg["simulation"]
    rng = np.random.default_rng(cfg.seed)
    n = int(s["n_accounts"])
    start = pd.Timestamp(s["start_date"])
    as_of = pd.Timestamp(s["as_of_date"])
    end = as_of + DAY  # payments up to the end of the as-of day
    drift = pd.Timestamp(s["drift_start"])
    open_start = pd.Timestamp(s["account_open_start"])

    # ------------------------------------------------------------------ accounts
    account_id = np.arange(1, n + 1)
    bust = rng.random(n) < s["bust_out_share"]
    wave = ~bust & (rng.random(n) < s["drift_wave_share"])  # dormant "sleeper" accounts
    risk = rng.normal(0, 1, n)            # stable credit-quality latent
    propensity = rng.normal(0, 0.6, n)    # account-level return tendency

    open_date = _uniform_dates(rng, open_start, as_of - 45 * DAY, n)
    bust_open = _uniform_dates(rng, start - 240 * DAY, drift - 30 * DAY, n)
    wave_active = _uniform_dates(rng, drift, drift + 50 * DAY, n)
    open_date = open_date.where(~bust, bust_open)
    # Sleepers were opened months earlier and stayed quiet until the wave.
    open_date = open_date.where(~wave, wave_active - pd.to_timedelta(rng.integers(60, 400, n), unit="D"))

    active_from = open_date.where(open_date > start, start)
    active_from = active_from.where(~wave, wave_active)

    credit_line = np.clip(np.round(rng.lognormal(np.log(2500), 0.7, n), -2), 300, 25000)
    util_base = np.clip(rng.beta(2, 5, n) + 0.08 * risk, 0.02, 0.98)
    util_base = np.where(bust, rng.uniform(0.6, 0.95, n), util_base)

    accounts = pd.DataFrame({
        "account_id": account_id,
        "open_date": open_date.dt.normalize().astype("datetime64[ns]"),
        "credit_line": credit_line,
    })
    accounts_truth = pd.DataFrame({
        "account_id": account_id,
        "is_bust_out": bust,
        "is_drift_wave": wave,
        "risk": risk,
        "propensity": propensity,
    })

    # --------------------------------------------------- monthly delinquency status
    months = pd.date_range(start, as_of, freq="MS")
    status_rows = []
    for m in months:
        live = open_date < m
        k = int(live.sum())
        p = _sigmoid(-3.3 + 1.1 * risk[live] + rng.normal(0, 0.5, k))
        p = np.where((bust | wave)[live], 0.02, p)  # fraud accounts look current until they bust
        delinquent = rng.random(k) < p
        status = np.where(
            delinquent,
            np.where(rng.random(k) < 0.7, "PAST_DUE_30", "PAST_DUE_60"),
            np.where(rng.random(k) < 0.55, "NOT_DUE", "CURRENT_DUE"),
        )
        status_rows.append(pd.DataFrame({"account_id": account_id[live], "as_of_date": m, "delinquency_status": status}))
    account_status = pd.concat(status_rows, ignore_index=True)
    account_status["as_of_date"] = account_status["as_of_date"].astype("datetime64[ns]")

    # ----------------------------------------------------------- funding accounts
    init_link = (open_date - pd.to_timedelta(rng.integers(0, 30, n), unit="D"))
    # The drift wave bought aged bank accounts: linked long before they pay.
    init_link = init_link.where(~wave, active_from - pd.to_timedelta(rng.integers(90, 240, n), unit="D"))
    links = [pd.DataFrame({"account_id": account_id, "linked_at": init_link})]

    years_active = ((end - active_from).dt.days / 365.0).to_numpy()
    n_extra = rng.poisson(np.where(bust, 3.0, 0.25 * years_active))
    rep = np.repeat(np.arange(n), n_extra)
    if len(rep):
        extra_at = _uniform_dates(rng, active_from.iloc[rep].reset_index(drop=True), end, len(rep))
        links.append(pd.DataFrame({"account_id": account_id[rep], "linked_at": extra_at}))
    funding = pd.concat(links, ignore_index=True)
    funding["linked_at"] = funding["linked_at"].astype("datetime64[ns]")
    funding = funding.sort_values(["linked_at", "account_id"]).reset_index(drop=True)
    funding["funding_account_id"] = np.arange(1, len(funding) + 1)

    # ------------------------------------------------------------------ payments
    months_active = ((end - active_from).dt.days / 30.44).to_numpy()
    rate = s["payments_per_month"] * np.exp(rng.normal(0, 0.3, n))
    rate = np.where(bust, rate * 3.0, rate)
    n_pay = rng.poisson(rate * months_active)
    # Sleepers kite: a burst of about 12 payments in their first three weeks.
    n_pay = np.where(wave, rng.poisson(12, n), n_pay)
    pay_end = pd.Series(end, index=active_from.index)
    pay_end = pay_end.where(~wave, (active_from + 21 * DAY).where(active_from + 21 * DAY < end, end))
    idx = np.repeat(np.arange(n), n_pay)
    m = len(idx)
    ts = _uniform_dates(rng, active_from.iloc[idx].reset_index(drop=True), pay_end.iloc[idx].reset_index(drop=True), m)

    pay = pd.DataFrame({"account_id": account_id[idx], "initiated_at": ts.astype("datetime64[ns]"), "_i": idx})
    pay = pay.sort_values("initiated_at").reset_index(drop=True)

    # Each payment draws on the most recently linked funding account.
    f = funding[["account_id", "linked_at", "funding_account_id"]].sort_values("linked_at")
    pay = pd.merge_asof(pay, f, left_on="initiated_at", right_on="linked_at", by="account_id", direction="backward")
    i = pay["_i"].to_numpy()
    w = wave[i]
    b = bust[i] | w
    m = len(pay)

    check_p = np.where(b & ~w, 0.30, s["check_share"])
    pay["payment_type"] = np.where(rng.random(m) < check_p, "CHECK", "ACH")
    ch = rng.random(m)
    web_p = np.where(b & ~w, 0.55, 0.30)
    phone_p = np.where(b & ~w, 0.20, 0.08)
    pay["channel"] = np.where(ch < phone_p, "PHONE", np.where(ch < phone_p + web_p, "WEB", "APP"))
    pay.loc[w, "channel"] = "APP"

    line = credit_line[i]
    balance = np.round(line * np.clip(util_base[i] + rng.normal(0, 0.08, m), 0.01, 1.0), 2)
    min_due = np.where(balance < 35, balance, np.maximum(35.0, np.round(0.02 * balance, 2)))
    kind = rng.random(m)
    partial = min_due + rng.random(m) * np.maximum(balance - min_due, 0)
    amount = np.where(kind < 0.35, min_due, np.where(kind < 0.55, balance, partial))
    # Bust-outs overpay to free as much credit as possible, but some structure
    # payments just under an illustrative $300 review cap. The drift wave always does.
    bust_amt = np.where(rng.random(m) < 0.35, rng.uniform(120, 299.99, m), line * rng.uniform(0.3, 1.0, m))
    amount = np.where(b, bust_amt, amount)
    amount = np.where(w, rng.uniform(180, 299.99, m), amount)
    pay["amount"] = np.round(np.maximum(amount, 5.0), 2)
    pay["statement_balance"] = balance
    pay["minimum_due"] = np.round(min_due, 2)

    # ----------------------------------------------- return mechanism (hidden)
    st = account_status.sort_values("as_of_date")
    pay = pd.merge_asof(pay.sort_values("initiated_at"), st, left_on="initiated_at", right_on="as_of_date",
                        by="account_id", direction="backward").reset_index(drop=True)
    pay["payment_id"] = [f"P{k:08d}" for k in range(1, len(pay) + 1)]
    i = pay["_i"].to_numpy()
    w = wave[i]
    b = bust[i] | w
    funding_age = (pay["initiated_at"] - pay["linked_at"]).dt.days.to_numpy()
    first_use = (pay.groupby("funding_account_id").cumcount() == 0).to_numpy()
    delinquent = pay["delinquency_status"].isin(["PAST_DUE_30", "PAST_DUE_60"]).to_numpy()
    night = pay["initiated_at"].dt.hour.to_numpy() < 5

    logit = (
        -5.2
        + 0.45 * risk[i]
        + 1.0 * propensity[i]
        + 0.9 * (funding_age < 14)
        + 0.6 * first_use
        + 0.30 * np.log(pay["amount"].to_numpy() / 100.0)
        + 0.5 * (pay["payment_type"].to_numpy() == "CHECK")
        + 0.3 * (pay["channel"].to_numpy() == "PHONE")
        + 0.25 * night
        + 0.8 * delinquent
        + np.where(w, 4.2, np.where(b, 3.0, 0.0))
    )
    will_return = rng.random(m) < _sigmoid(logit)

    code_r = rng.random(m)
    code = np.where(
        b,
        np.select([code_r < 0.45, code_r < 0.70, code_r < 0.80], ["R01", "R02", "R03"], "R10"),
        np.select([code_r < 0.80, code_r < 0.88, code_r < 0.92], ["R01", "R02", "R03"], "R10"),
    )
    is_check = pay["payment_type"].to_numpy() == "CHECK"
    fast_lag = np.where(is_check, rng.integers(3, 11, m), rng.integers(1, 5, m))
    lag = np.where(code == "R10", rng.integers(10, 61, m), fast_lag)
    returned_at = pay["initiated_at"] + pd.to_timedelta(lag, unit="D") + pd.to_timedelta(rng.random(m) * 86400, unit="s")

    payments = pay[["payment_id", "account_id", "funding_account_id", "payment_type", "channel",
                    "amount", "statement_balance", "minimum_due", "initiated_at"]].copy()

    observed = will_return & (returned_at <= end).to_numpy()
    returns = pd.DataFrame({
        "payment_id": pay["payment_id"][observed].to_numpy(),
        "return_code": code[observed],
        "returned_at": returned_at[observed].to_numpy(),
    })

    outcomes = pd.DataFrame({
        "payment_id": pay["payment_id"].to_numpy(),
        "will_return": will_return,
        "true_return_code": np.where(will_return, code, None),
        "true_returned_at": returned_at.where(will_return),
        # Share of released credit that would be spent before the return, and
        # share of that spend recovered afterwards (stands in for card
        # transaction and recovery data).
        "potential_spend_share": np.where(b, rng.uniform(0.85, 1.0, m), rng.beta(2, 5, m)),
        "recovery_rate": np.where(b, 0.05, 0.6),
    })

    funding_out = funding[["funding_account_id", "account_id", "linked_at"]]
    return {
        "accounts": accounts,
        "funding_accounts": funding_out,
        "account_status": account_status,
        "payments": payments,
        "returns": returns,
        "truth/accounts_truth": accounts_truth,
        "truth/payment_outcomes": outcomes,
    }


def write_tables(tables: dict[str, pd.DataFrame], cfg: Config) -> None:
    base = cfg.data_dir
    for name, df in tables.items():
        path = base / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


def read_table(cfg: Config, name: str) -> pd.DataFrame:
    return pd.read_parquet(cfg.data_dir / f"{name}.parquet")
