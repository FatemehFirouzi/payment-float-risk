# Monitoring report

As of 2025-12-31. Production from 2025-09-01. Score cutoff 11, amount cap $300, prior-return window 180 days.

## 1. Is the strategy working as expected?

| Check | Value | Target | Status | Note |
| --- | --- | --- | --- | --- |
| Unscored rate (worst day) | 100.0% | <= 1% | **ALERT** | 2025-11-18 |
| Policy mismatches (ledger vs. rules) | 0 | <= 0 | **OK** |  |
| Release rate (all production) | 47.0% | 45% to 80% | **OK** |  |
| Released 30-day return rate, weeks after drift | 0.69% | <= 0.90% | **OK** | before drift 0.60%; worst week 2025-11-03 at 0.80% |
| Realised loss per 1,000 released | $446 | about $58 (pre-drift) | **ALERT** | 7.7x pre-drift |
| Score PSI (worst week) | 0.006 | < 0.1 | **OK** |  |

![Weekly monitoring](figures/weekly_monitoring.png)

## 2. Are the thresholds still right?

| Check | Value | Target | Status | Note |
| --- | --- | --- | --- | --- |
| Weekly AUC (worst week, 30-day label) | 0.755 | >= 0.7 | **OK** |  |
| Score cutoff vs. best on recent data | current 11, best 7 | gap within 10% of net value | **ALERT** | moving to 7 changes net value by $2,068 over 2025-10-20 to 2025-11-30 |
| Amount cap vs. best on recent data | current $300, best $180 | gap within 10% of net value | **ALERT** | best cap with its best cutoff (75) adds $7,231 |

![Threshold trade-off](figures/threshold_tradeoff.png)

Amount-cap sweep on 2025-10-20 to 2025-11-30 (30-day label, realised spend and recovery):

| Amount cap | Net value at current cutoff | Best cutoff | Best net value | Release rate (all) | Return rate released |
| --- | --- | --- | --- | --- | --- |
| $180 | $17,193 | 75 | $18,278 | 44.7% | 0.55% |
| $225 | $14,865 | 12 | $15,183 | 45.2% | 0.60% |
| $250 | $13,908 | 12 | $14,257 | 46.4% | 0.62% |
| $275 | $11,275 | 12 | $11,679 | 47.5% | 0.67% |
| $300 | $8,979 | 7 | $11,047 | 34.8% | 0.58% |
| $350 | $9,440 | 7 | $11,195 | 35.2% | 0.57% |

## Model

| Split | AUC | KS | Bad rate | Mean predicted |
| --- | --- | --- | --- | --- |
| Train (Jan to Jun) | 0.801 | 0.446 | 1.77% | 1.74% |
| Selection (Aug, out of time) | 0.778 | 0.429 | 1.90% | 1.92% |

![Calibration](figures/calibration.png) ![Feature importance](figures/feature_importance.png)
