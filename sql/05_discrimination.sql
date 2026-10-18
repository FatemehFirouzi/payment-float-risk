-- =============================================================================
-- mon_discrimination: weekly AUC, KS and Gini on scored payments with 30 days
-- of maturity. Bad = returned within 30 days.
-- AUC uses the Mann-Whitney rank sum with tied scores given average ranks.
-- =============================================================================
CREATE OR REPLACE TABLE mon_discrimination AS
WITH m AS (
    SELECT payment_week, score, return_30d AS bad
    FROM mon_base
    WHERE scored AND return_30d IS NOT NULL
),
by_score AS (
    SELECT payment_week, score, COUNT(*) AS n, SUM(bad) AS bads, COUNT(*) - SUM(bad) AS goods
    FROM m
    GROUP BY 1, 2
),
cum AS (
    SELECT *,
        COALESCE(SUM(n) OVER (PARTITION BY payment_week ORDER BY score
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS n_below,
        SUM(bads)  OVER (PARTITION BY payment_week ORDER BY score ROWS UNBOUNDED PRECEDING) AS cum_bads,
        SUM(goods) OVER (PARTITION BY payment_week ORDER BY score ROWS UNBOUNDED PRECEDING) AS cum_goods,
        SUM(bads)  OVER (PARTITION BY payment_week) AS total_bads,
        SUM(goods) OVER (PARTITION BY payment_week) AS total_goods
    FROM by_score
),
week AS (
    SELECT
        payment_week,
        SUM(n)                                   AS payments,
        MAX(total_bads)                          AS bads,
        MAX(total_goods)                         AS goods,
        SUM(bads * (n_below + (n + 1) / 2.0))    AS bad_rank_sum,
        MAX(abs(cum_bads / NULLIF(total_bads, 0) - cum_goods / NULLIF(total_goods, 0))) AS ks
    FROM cum
    GROUP BY 1
)
SELECT
    payment_week,
    payments,
    bads,
    bads / payments                                                        AS bad_rate_30d,
    (bad_rank_sum - bads * (bads + 1) / 2.0) / NULLIF(bads * goods, 0)      AS auc,
    2 * (bad_rank_sum - bads * (bads + 1) / 2.0) / NULLIF(bads * goods, 0) - 1 AS gini,
    ks
FROM week
ORDER BY payment_week;
