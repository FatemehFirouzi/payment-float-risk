-- =============================================================================
-- mon_psi: weekly population stability of the score vs. the frozen baseline
-- (calibration + selection months), on fixed decile cutoffs.
-- Rule of thumb: < 0.10 stable, 0.10 - 0.25 watch, > 0.25 act.
-- =============================================================================
CREATE OR REPLACE TABLE mon_psi AS
WITH b AS (
    SELECT quantile_cont(score, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) AS cutoffs
    FROM baseline_scores
),
baseline_dist AS (
    SELECT 1 + len(list_filter(b.cutoffs, c -> c < s.score)) AS decile,
           COUNT(*) / SUM(COUNT(*)) OVER () AS expected
    FROM baseline_scores s CROSS JOIN b
    GROUP BY 1
),
weekly AS (
    SELECT payment_week, score_decile AS decile, COUNT(*) AS n
    FROM mon_base
    WHERE scored
    GROUP BY 1, 2
),
grid AS (
    SELECT w.payment_week, d.decile
    FROM (SELECT DISTINCT payment_week FROM weekly) w
    CROSS JOIN (SELECT range AS decile FROM range(1, 11)) d
),
weekly_dist AS (
    SELECT g.payment_week, g.decile,
           COALESCE(w.n, 0) AS n,
           COALESCE(w.n, 0) / SUM(COALESCE(w.n, 0)) OVER (PARTITION BY g.payment_week) AS actual
    FROM grid g
    LEFT JOIN weekly w USING (payment_week, decile)
)
SELECT
    wd.payment_week,
    wd.decile,
    wd.n,
    wd.actual,
    bd.expected,
    (greatest(wd.actual, 1e-4) - greatest(bd.expected, 1e-4))
      * ln(greatest(wd.actual, 1e-4) / greatest(bd.expected, 1e-4)) AS psi_contribution,
    SUM((greatest(wd.actual, 1e-4) - greatest(bd.expected, 1e-4))
      * ln(greatest(wd.actual, 1e-4) / greatest(bd.expected, 1e-4))) OVER (PARTITION BY wd.payment_week) AS psi_week
FROM weekly_dist wd
JOIN baseline_dist bd USING (decile)
ORDER BY wd.payment_week, wd.decile;
