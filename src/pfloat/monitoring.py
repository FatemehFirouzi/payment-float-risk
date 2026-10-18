"""Run the monitoring SQL in DuckDB.

The SQL files in sql/ are plain DuckDB SQL over named tables, so they port to
Databricks / Spark SQL with small syntax changes (ASOF JOIN and list lambdas
are the DuckDB-specific parts).
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from .config import Config
from .pipeline import load_strategy

log = logging.getLogger("pfloat")

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
OUTPUT_TABLES = ["mon_base", "mon_daily", "mon_performance", "mon_psi", "mon_discrimination"]


def connect(cfg: Config) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    d = cfg.data_dir
    sources = {
        "payments": d / "payments.parquet",
        "returns": d / "returns.parquet",
        "account_status": d / "account_status.parquet",
        "outcomes": d / "truth" / "payment_outcomes.parquet",
        "ledger": d / "production" / "ledger.parquet",
        "scores": d / "production" / "scores.parquet",
        "baseline_scores": cfg.artifacts_dir / "baseline_scores.parquet",
    }
    for name, path in sources.items():
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path.as_posix()}')")

    strategy = load_strategy(cfg)
    con.execute(f"SET VARIABLE as_of = DATE '{cfg.as_of.date()}'")
    con.execute(f"SET VARIABLE prod_start = TIMESTAMP '{cfg.production_start}'")
    con.execute(f"SET VARIABLE cutoff = {int(strategy['score_threshold'])}")
    con.execute(f"SET VARIABLE amount_cap = {float(strategy['amount_cap'])}")
    con.execute(f"SET VARIABLE prior_return_days = {int(strategy['prior_return_days'])}")
    return con


def stage_monitor(cfg: Config) -> None:
    con = connect(cfg)
    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        con.execute(sql_file.read_text())
    out = cfg.artifacts_dir / "monitoring"
    out.mkdir(exist_ok=True)
    for t in OUTPUT_TABLES:
        con.execute(f"COPY {t} TO '{(out / f'{t}.parquet').as_posix()}' (FORMAT parquet)")
    n, mism = con.execute("SELECT COUNT(*), SUM(policy_mismatch::INT) FROM mon_base").fetchone()
    log.info("monitoring: %s payments, %s policy mismatches", f"{n:,}", mism)


def load(cfg: Config, table: str) -> pd.DataFrame:
    return pd.read_parquet(cfg.artifacts_dir / "monitoring" / f"{table}.parquet")
