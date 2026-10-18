"""Pipeline stages. Each reads its inputs from disk and writes its outputs,
so any stage can be rerun on its own."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import ledger as ledger_mod
from . import threshold
from .config import Config
from .features import add_label, build_features
from .model import ScoringModel, split_masks, train
from .simulate import read_table, simulate, write_tables
from .strategy import decide

log = logging.getLogger("pfloat")


def stage_simulate(cfg: Config) -> None:
    tables = simulate(cfg)
    write_tables(tables, cfg)
    log.info("simulated %s payments, %s observed returns", f"{len(tables['payments']):,}", f"{len(tables['returns']):,}")


def stage_features(cfg: Config) -> None:
    returns = read_table(cfg, "returns")
    feats = build_features(read_table(cfg, "payments"), read_table(cfg, "accounts"),
                           read_table(cfg, "funding_accounts"), read_table(cfg, "account_status"), returns)
    feats = add_label(feats, returns, cfg.as_of, int(cfg["label"]["window_days"]))
    feats.to_parquet(cfg.data_dir / "features.parquet", index=False)
    log.info("built %s feature rows", f"{len(feats):,}")


def stage_train(cfg: Config) -> None:
    feats = pd.read_parquet(cfg.data_dir / "features.parquet")
    model, report = train(feats, cfg)
    model.save(cfg.artifacts_dir / "model")
    (cfg.artifacts_dir / "model_report.json").write_text(json.dumps(report, indent=2))
    sel = report["splits"]["selection"]["calibrated"]
    log.info("selection-month AUC %.3f  KS %.3f", sel["auc"], sel["ks"])

    # Baseline score distribution for PSI: calibration + selection months.
    masks = split_masks(feats, cfg)
    base = feats[masks["calibration"] | masks["selection"]]
    pd.DataFrame({"payment_id": base["payment_id"], "score": model.score(base)}).to_parquet(
        cfg.artifacts_dir / "baseline_scores.parquet", index=False)


def stage_select_threshold(cfg: Config) -> None:
    feats = pd.read_parquet(cfg.data_dir / "features.parquet")
    model = ScoringModel.load(cfg.artifacts_dir / "model")
    sel = feats[split_masks(feats, cfg)["selection"]].copy()
    sel["score"] = model.score(sel)
    curve = threshold.tradeoff(sel, cfg["strategy"], cfg["economics"])
    curve.to_parquet(cfg.artifacts_dir / "tradeoff_selection.parquet", index=False)
    recommended = threshold.recommend(curve)
    override = cfg["strategy"].get("score_threshold")
    cutoff = int(override) if override is not None else recommended
    row = curve[curve["cutoff"] == recommended].iloc[0]
    strategy = {
        "score_threshold": cutoff,
        "recommended_on_selection": recommended,
        "selection_release_rate_eligible": float(row["release_rate_eligible"]),
        "selection_return_rate_released": float(row["return_rate_released"]),
        **{k: v for k, v in cfg["strategy"].items() if k != "score_threshold"},
    }
    (cfg.artifacts_dir / "strategy.json").write_text(json.dumps(strategy, indent=2))
    log.info("score cutoff %s (recommended %s)", cutoff, recommended)


def load_strategy(cfg: Config) -> dict:
    return json.loads((cfg.artifacts_dir / "strategy.json").read_text())


def stage_production(cfg: Config) -> None:
    """Replay production one day at a time: score, decide, write to the ledger."""
    feats = pd.read_parquet(cfg.data_dir / "features.parquet")
    model = ScoringModel.load(cfg.artifacts_dir / "model")
    strategy = load_strategy(cfg)
    cutoff = int(strategy["score_threshold"])
    outages = {pd.Timestamp(d) for d in cfg["simulation"].get("feature_outage_dates", [])}

    keep = list(dict.fromkeys(model.features + ["payment_id", "account_id", "initiated_at", "amount",
                                                "delinquency_status", "prior_returns_90d", "prior_returns_180d"]))
    prod = feats.loc[feats["initiated_at"] >= cfg.production_start, keep].copy()
    prod["run_date"] = prod["initiated_at"].dt.normalize()

    ledger = ledger_mod.empty_ledger()
    score_rows = []
    for run_date, day in prod.groupby("run_date", sort=True):
        day = day.copy()
        if run_date in outages:
            # Simulate an upstream outage: the feature columns arrive empty.
            # The scorer refuses to score rather than guess, so these payments
            # are held as unscored, and monitoring has to catch it.
            day["score"] = np.nan
        else:
            day["score"] = model.score(day).astype(float)
            score_rows.append(pd.DataFrame({"payment_id": day["payment_id"], "score": day["score"].astype(int),
                                            "run_date": run_date, "model_version": model.version}))
        decided = decide(day, cutoff, strategy)
        rows = ledger_mod.to_ledger_rows(decided, run_date, model.version, cutoff)
        ledger, _ = ledger_mod.upsert(ledger, rows)

    out = cfg.data_dir / "production"
    out.mkdir(exist_ok=True)
    pd.concat(score_rows, ignore_index=True).to_parquet(out / "scores.parquet", index=False)
    ledger_mod.save(ledger, out / "ledger.parquet")
    log.info("production: %s decisions, %.1f%% released", f"{len(ledger):,}", 100 * ledger["released"].mean())


def run_all(cfg: Config) -> None:
    from .monitoring import stage_monitor
    from .report import stage_report

    for stage in (stage_simulate, stage_features, stage_train, stage_select_threshold,
                  stage_production, stage_monitor, stage_report):
        log.info("== %s", stage.__name__)
        stage(cfg)
