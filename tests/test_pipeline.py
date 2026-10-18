"""End-to-end run on the small CI config. Slowest test; proves the stages fit together."""

from pathlib import Path

import pandas as pd
import pytest

from pfloat.config import Config
from pfloat.pipeline import run_all

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    c = Config.load(ROOT / "config" / "ci.yaml")
    run_all(c)
    return c


def test_simulated_returns_are_never_after_as_of(cfg):
    r = pd.read_parquet(cfg.data_dir / "returns.parquet")
    assert (r["returned_at"] <= cfg.as_of + pd.Timedelta(days=1)).all()


def test_ledger_has_one_decision_per_production_payment(cfg):
    led = pd.read_parquet(cfg.data_dir / "production" / "ledger.parquet")
    pay = pd.read_parquet(cfg.data_dir / "payments.parquet")
    prod = pay[pay["initiated_at"] >= cfg.production_start]
    assert led["payment_id"].is_unique
    assert set(led["payment_id"]) == set(prod["payment_id"])


def test_monitoring_policy_rederivation_matches_ledger(cfg):
    base = pd.read_parquet(cfg.artifacts_dir / "monitoring" / "mon_base.parquet")
    assert int(base["policy_mismatch"].sum()) == 0


def test_immature_payments_have_null_outcomes(cfg):
    base = pd.read_parquet(cfg.artifacts_dir / "monitoring" / "mon_base.parquet")
    recent = base[base["days_since_payment"] < 30]
    assert recent["return_30d"].isna().all()


def test_outage_day_is_flagged_as_unscored(cfg):
    daily = pd.read_parquet(cfg.artifacts_dir / "monitoring" / "mon_daily.parquet")
    day = daily[pd.to_datetime(daily["payment_date"]) == pd.Timestamp("2025-11-18")]
    assert float(day["unscored_rate"].iloc[0]) == 1.0


def test_report_is_written(cfg):
    assert (cfg.reports_dir / "monitoring_report.md").exists()
    assert (cfg.reports_dir / "figures" / "weekly_monitoring.png").exists()
