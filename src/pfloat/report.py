"""Monitoring report: answers the two questions with numbers and charts.

1. Is the strategy working as expected?  -> checks against targets in config
2. Are the thresholds still right?       -> trade-off on recent matured data
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import threshold
from .config import Config
from .monitoring import load
from .pipeline import load_strategy

log = logging.getLogger("pfloat")

# Reference palette (light mode): series in fixed order, text never in series colour.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9.5,
    "lines.linewidth": 2.0, "legend.frameon": False,
})


@dataclass
class Check:
    question: str
    name: str
    value: str
    target: str
    status: str
    note: str = ""


def _status(ok: bool, warn: bool = False) -> str:
    return "OK" if ok else ("WATCH" if warn else "ALERT")


def mature_weeks(df: pd.DataFrame, as_of: pd.Timestamp, days: int) -> pd.Series:
    """True for weeks whose every day has `days` of maturity."""
    week_end = pd.to_datetime(df["payment_week"]) + pd.Timedelta(days=6)
    return week_end + pd.Timedelta(days=days) <= as_of


def weekly_outcomes(base: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rel = base[base["released"]]
    w = rel.groupby("payment_week").agg(
        released=("payment_id", "size"),
        mature_30d=("return_30d", "count"),
        returns_30d=("return_30d", "sum"),
        realised_loss=("realised_loss", "sum"),
    ).reset_index()
    w["payment_week"] = pd.to_datetime(w["payment_week"])
    w["return_rate_30d"] = w["returns_30d"] / w["mature_30d"]
    w["loss_per_1k_released"] = 1000 * w["realised_loss"] / w["released"]
    w["mature"] = mature_weeks(w, cfg.as_of, 30)
    return w


def recent_tradeoff(cfg: Config, base: pd.DataFrame, strategy: dict, weeks: int = 6):
    """Trade-off on the most recent fully matured weeks, using realised spend and recovery."""
    feats = pd.read_parquet(cfg.data_dir / "features.parquet",
                            columns=["payment_id", "delinquency_status", "prior_returns_90d", "prior_returns_180d"])
    outcomes = pd.read_parquet(cfg.data_dir / "truth" / "payment_outcomes.parquet",
                               columns=["payment_id", "potential_spend_share", "recovery_rate"])
    b = base[["payment_id", "payment_week", "amount", "score", "return_30d"]].copy()
    b["payment_week"] = pd.to_datetime(b["payment_week"])
    b = b[mature_weeks(b, cfg.as_of, 30)]
    recent = sorted(b["payment_week"].unique())[-weeks:]
    b = b[b["payment_week"].isin(recent)]
    d = b.merge(feats, on="payment_id").merge(outcomes, on="payment_id")
    window = (pd.Timestamp(recent[0]).date(), (pd.Timestamp(recent[-1]) + pd.Timedelta(days=6)).date())

    curve = threshold.tradeoff(d, strategy, cfg["economics"], label_col="return_30d",
                               spend_col="potential_spend_share", recovery_col="recovery_rate")
    caps = []
    for cap in (180, 225, 250, 275, 300, 350):
        s2 = {**strategy, "amount_cap": cap}
        c = threshold.tradeoff(d, s2, cfg["economics"], label_col="return_30d",
                               spend_col="potential_spend_share", recovery_col="recovery_rate")
        at_current = c[c["cutoff"] <= strategy["score_threshold"]].iloc[-1]
        best = c.loc[c["net_value"].idxmax()]
        caps.append({
            "amount_cap": cap,
            "net_value_at_current_cutoff": float(at_current["net_value"]),
            "best_cutoff": int(best["cutoff"]),
            "best_net_value": float(best["net_value"]),
            "release_rate_all_at_best": float(best["release_rate_all"]),
            "return_rate_released_at_best": float(best["return_rate_released"]),
        })
    return curve, pd.DataFrame(caps), window


# ----------------------------------------------------------------------- charts

def _mark_events(ax, cfg: Config) -> None:
    drift = pd.Timestamp(cfg["simulation"]["drift_start"])
    ax.axvline(drift, color=INK_2, linestyle=(0, (3, 3)), linewidth=1)
    for d in cfg["simulation"].get("feature_outage_dates", []):
        ax.axvline(pd.Timestamp(d), color=GRID, linewidth=6, zorder=0)


def fig_weekly(cfg: Config, weekly: pd.DataFrame, psi: pd.DataFrame, disc: pd.DataFrame, daily: pd.DataFrame, path):
    t = cfg["monitoring"]["targets"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True)
    m = weekly[weekly["mature"]]

    ax = axes[0, 0]
    ax.plot(m["payment_week"], 100 * m["return_rate_30d"], color=BLUE, marker="o", markersize=4)
    ax.axhline(100 * t["max_released_return_rate_30d"], color=ORANGE, linewidth=1.2)
    ax.text(m["payment_week"].iloc[0], 100 * t["max_released_return_rate_30d"], " target", color=INK_2,
            va="bottom", fontsize=8.5)
    ax.set_title("Released payments: 30-day return rate (%)")
    _mark_events(ax, cfg)

    ax = axes[0, 1]
    ax.plot(m["payment_week"], m["loss_per_1k_released"], color=BLUE, marker="o", markersize=4)
    ax.set_ylim(bottom=0)
    ax.set_title("Realised loss per 1,000 released payments ($)")
    _mark_events(ax, cfg)

    ax = axes[1, 0]
    p = psi.groupby("payment_week")["psi_week"].first().reset_index()
    p["payment_week"] = pd.to_datetime(p["payment_week"])
    ax.plot(p["payment_week"], p["psi_week"], color=BLUE, marker="o", markersize=4)
    ax.axhline(t["psi_warn"], color=ORANGE, linewidth=1.2)
    ax.text(p["payment_week"].iloc[0], t["psi_warn"], " watch level", color=INK_2, va="bottom", fontsize=8.5)
    ax.set_ylim(0, max(0.15, p["psi_week"].max() * 1.3))
    ax.set_title("Score PSI vs. baseline")
    _mark_events(ax, cfg)

    ax = axes[1, 1]
    d = daily.copy()
    d["payment_date"] = pd.to_datetime(d["payment_date"])
    ax.plot(d["payment_date"], 100 * d["unscored_rate"], color=BLUE, linewidth=1.5)
    ax.axhline(100 * t["max_unscored_rate"], color=ORANGE, linewidth=1.2)
    ax.set_title("Unscored payments per day (%)")
    _mark_events(ax, cfg)

    for ax in axes[1]:
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    fig.text(0.01, 0.005, "Dashed line: start of the sleeper-account wave.  Grey band: feature outage day.  "
             "Top panels show only weeks with full 30-day maturity.", color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_tradeoff(selection: pd.DataFrame, recent: pd.DataFrame, cutoff: int, recommended: int, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for curve, color, label in ((selection, BLUE, "Selection month (Aug, 60-day label)"),
                                (recent, ORANGE, "Recent 6 matured weeks (30-day label)")):
        curve = curve[curve["released"] >= 500]  # skip the noisy low-volume start
        axes[0].plot(100 * curve["release_rate_eligible"], 100 * curve["return_rate_released"], color=color, label=label)
        cur = curve[curve["cutoff"] <= cutoff].iloc[-1]
        axes[0].plot(100 * cur["release_rate_eligible"], 100 * cur["return_rate_released"], "o", color=color,
                     markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    axes[0].set_xlabel("Eligible payments released (%)")
    axes[0].set_ylabel("Return rate among released (%)")
    axes[0].set_title("Release more, return more: where the cutoff sits")
    axes[0].legend(loc="upper left", fontsize=8.5)

    r = recent[recent["cutoff"] <= 120]
    axes[1].plot(r["cutoff"], r["net_value"] / 1000, color=ORANGE)
    axes[1].axvline(cutoff, color=INK_2, linestyle=(0, (3, 3)), linewidth=1)
    axes[1].text(cutoff, axes[1].get_ylim()[0], f" current {cutoff}", color=INK_2, va="bottom", fontsize=8.5)
    axes[1].axvline(recommended, color=BLUE, linewidth=1.2)
    axes[1].text(recommended, axes[1].get_ylim()[1], f" best {recommended}", color=INK_2, va="top", fontsize=8.5)
    axes[1].set_xlabel("Score cutoff (release if score <= cutoff)")
    axes[1].set_ylabel("Net value, $ thousands")
    axes[1].set_title("Recent weeks: net value by cutoff")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_calibration(base: pd.DataFrame, path):
    b = base[base["scored"] & base["return_60d"].notna()].copy()
    b["bin"] = pd.qcut(b["score"], 10, duplicates="drop")
    g = b.groupby("bin", observed=True).agg(pred=("predicted_return_prob", "mean"), actual=("return_60d", "mean"),
                                            n=("payment_id", "size"))
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    lim = max(g["pred"].max(), g["actual"].max()) * 110
    ax.plot([0, lim], [0, lim], color=GRID, linewidth=1.2)
    ax.plot(100 * g["pred"], 100 * g["actual"], "o-", color=BLUE, markersize=6)
    ax.set_xlabel("Predicted return rate (%)")
    ax.set_ylabel("Actual 60-day return rate (%)")
    ax.set_title("Calibration on production months (score deciles)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_importance(importance: dict, path, top: int = 12):
    items = list(importance.items())[:top][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.barh([k for k, _ in items], [v for _, v in items], color=BLUE, height=0.6)
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.set_title("Top features by mean absolute SHAP")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------- report

def stage_report(cfg: Config) -> dict:
    t = cfg["monitoring"]["targets"]
    strategy = load_strategy(cfg)
    cutoff = int(strategy["score_threshold"])
    model_report = json.loads((cfg.artifacts_dir / "model_report.json").read_text())
    base, daily = load(cfg, "mon_base"), load(cfg, "mon_daily")
    psi, disc = load(cfg, "mon_psi"), load(cfg, "mon_discrimination")
    weekly = weekly_outcomes(base, cfg)
    drift = pd.Timestamp(cfg["simulation"]["drift_start"])

    checks: list[Check] = []
    Q1, Q2 = "Working as expected?", "Thresholds still right?"

    bad_days = daily[daily["unscored_rate"] > t["max_unscored_rate"]]
    checks.append(Check(Q1, "Unscored rate (worst day)", f"{daily['unscored_rate'].max():.1%}",
                        f"<= {t['max_unscored_rate']:.0%}", _status(bad_days.empty),
                        ", ".join(str(pd.Timestamp(d).date()) for d in bad_days["payment_date"]) or ""))
    mism = int(base["policy_mismatch"].sum())
    checks.append(Check(Q1, "Policy mismatches (ledger vs. rules)", f"{mism:,}", f"<= {t['max_policy_mismatches']}",
                        _status(mism <= t["max_policy_mismatches"])))
    rr = base["released"].mean()
    lo, hi = t["release_rate_band"]
    checks.append(Check(Q1, "Release rate (all production)", f"{rr:.1%}", f"{lo:.0%} to {hi:.0%}",
                        _status(lo <= rr <= hi)))

    m = weekly[weekly["mature"]]
    pre = m[m["payment_week"] + pd.Timedelta(days=6) < drift]
    post = m[m["payment_week"] >= drift]
    pre_rate = pre["returns_30d"].sum() / pre["mature_30d"].sum()
    post_rate = post["returns_30d"].sum() / post["mature_30d"].sum() if len(post) else float("nan")
    worst = m.loc[m["return_rate_30d"].idxmax()]
    checks.append(Check(Q1, "Released 30-day return rate, weeks after drift", f"{post_rate:.2%}",
                        f"<= {t['max_released_return_rate_30d']:.2%}",
                        _status(post_rate <= t["max_released_return_rate_30d"]),
                        f"before drift {pre_rate:.2%}; worst week {pd.Timestamp(worst['payment_week']).date()} "
                        f"at {worst['return_rate_30d']:.2%}"))
    pre_loss = weekly[weekly["payment_week"] + pd.Timedelta(days=6) < drift]
    post_loss = weekly[(weekly["payment_week"] >= drift) & weekly["mature"]]
    pre_l = 1000 * pre_loss["realised_loss"].sum() / pre_loss["released"].sum()
    post_l = 1000 * post_loss["realised_loss"].sum() / post_loss["released"].sum()
    checks.append(Check(Q1, "Realised loss per 1,000 released", f"${post_l:,.0f}", f"about ${pre_l:,.0f} (pre-drift)",
                        _status(post_l <= 1.5 * pre_l, warn=post_l <= 2.5 * pre_l), f"{post_l / pre_l:.1f}x pre-drift"))
    psi_max = psi.groupby("payment_week")["psi_week"].first().max()
    checks.append(Check(Q1, "Score PSI (worst week)", f"{psi_max:.3f}", f"< {t['psi_warn']}",
                        _status(psi_max < t["psi_warn"], warn=psi_max < t["psi_alert"])))
    dm = disc[disc["bads"] >= 50]
    auc_min = dm["auc"].min()
    checks.append(Check(Q2, "Weekly AUC (worst week, 30-day label)", f"{auc_min:.3f}", f">= {t['min_auc']}",
                        _status(auc_min >= t["min_auc"])))

    sel_curve = pd.read_parquet(cfg.artifacts_dir / "tradeoff_selection.parquet")
    rec_curve, caps, window = recent_tradeoff(cfg, base, strategy)
    recommended = threshold.recommend(rec_curve)
    cur_row = rec_curve[rec_curve["cutoff"] <= cutoff].iloc[-1]
    best_row = rec_curve[rec_curve["cutoff"] == recommended].iloc[0]
    gain = best_row["net_value"] - cur_row["net_value"]
    checks.append(Check(Q2, "Score cutoff vs. best on recent data", f"current {cutoff}, best {recommended}",
                        "gap within 10% of net value", _status(abs(gain) <= 0.1 * abs(best_row["net_value"])),
                        f"moving to {recommended} changes net value by ${gain:,.0f} over {window[0]} to {window[1]}"))
    best_cap = caps.loc[caps["best_net_value"].idxmax()]
    cur_cap = caps[caps["amount_cap"] == strategy["amount_cap"]].iloc[0]
    cap_gain = best_cap["best_net_value"] - cur_cap["best_net_value"]
    checks.append(Check(Q2, "Amount cap vs. best on recent data",
                        f"current ${strategy['amount_cap']}, best ${int(best_cap['amount_cap'])}",
                        "gap within 10% of net value",
                        _status(abs(cap_gain) <= 0.1 * abs(best_cap["best_net_value"])),
                        f"best cap with its best cutoff ({int(best_cap['best_cutoff'])}) adds ${cap_gain:,.0f}"))

    fig_dir = cfg.reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_weekly(cfg, weekly, psi, disc, daily, fig_dir / "weekly_monitoring.png")
    fig_tradeoff(sel_curve, rec_curve, cutoff, recommended, fig_dir / "threshold_tradeoff.png")
    fig_calibration(base, fig_dir / "calibration.png")
    fig_importance(model_report["feature_importance"], fig_dir / "feature_importance.png")

    summary = _write_markdown(cfg, checks, model_report, strategy, caps, window, weekly)
    log.info("report written: %s", cfg.reports_dir / "monitoring_report.md")
    return summary


def _write_markdown(cfg, checks, model_report, strategy, caps, window, weekly) -> dict:
    sel = model_report["splits"]["selection"]["calibrated"]
    tr = model_report["splits"]["train"]["calibrated"]
    q = lambda name: [c for c in checks if c.question == name]

    def table(rows):
        out = ["| Check | Value | Target | Status | Note |", "| --- | --- | --- | --- | --- |"]
        out += [f"| {c.name} | {c.value} | {c.target} | **{c.status}** | {c.note} |" for c in rows]
        return "\n".join(out)

    caps_md = ["| Amount cap | Net value at current cutoff | Best cutoff | Best net value | Release rate (all) | Return rate released |",
               "| --- | --- | --- | --- | --- | --- |"]
    for _, r in caps.iterrows():
        caps_md.append(f"| ${int(r.amount_cap)} | ${r.net_value_at_current_cutoff:,.0f} | {int(r.best_cutoff)} | "
                       f"${r.best_net_value:,.0f} | {r.release_rate_all_at_best:.1%} | {r.return_rate_released_at_best:.2%} |")

    md = f"""# Monitoring report

As of {cfg.as_of.date()}. Production from {cfg.production_start.date()}. Score cutoff {strategy['score_threshold']}, amount cap ${strategy['amount_cap']}, prior-return window {strategy['prior_return_days']} days.

## 1. Is the strategy working as expected?

{table(q("Working as expected?"))}

![Weekly monitoring](figures/weekly_monitoring.png)

## 2. Are the thresholds still right?

{table(q("Thresholds still right?"))}

![Threshold trade-off](figures/threshold_tradeoff.png)

Amount-cap sweep on {window[0]} to {window[1]} (30-day label, realised spend and recovery):

{chr(10).join(caps_md)}

## Model

| Split | AUC | KS | Bad rate | Mean predicted |
| --- | --- | --- | --- | --- |
| Train (Jan to Jun) | {tr['auc']:.3f} | {tr['ks']:.3f} | {tr['bad_rate']:.2%} | {tr['mean_predicted']:.2%} |
| Selection (Aug, out of time) | {sel['auc']:.3f} | {sel['ks']:.3f} | {sel['bad_rate']:.2%} | {sel['mean_predicted']:.2%} |

![Calibration](figures/calibration.png) ![Feature importance](figures/feature_importance.png)
"""
    (cfg.reports_dir / "monitoring_report.md").write_text(md)
    summary = {
        "checks": [c.__dict__ for c in checks],
        "model": {"train": tr, "selection": sel},
        "strategy": strategy,
        "amount_cap_sweep": caps.to_dict(orient="records"),
        "window": [str(window[0]), str(window[1])],
    }
    (cfg.reports_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
