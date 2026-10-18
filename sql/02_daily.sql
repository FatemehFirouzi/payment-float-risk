-- =============================================================================
-- mon_daily: pipeline health and decision volume per payment day.
-- Counts and sums only; rates are derived from them, never averaged.
-- =============================================================================
CREATE OR REPLACE TABLE mon_daily AS
SELECT
    payment_date,
    COUNT(*)                                        AS payments,
    SUM(scored::INT)                                AS scored,
    1 - SUM(scored::INT) / COUNT(*)                 AS unscored_rate,
    SUM(released::INT)                              AS released,
    SUM(released::INT) / COUNT(*)                   AS release_rate,
    SUM(CASE WHEN released THEN amount ELSE 0 END)  AS released_amount,
    SUM(policy_mismatch::INT)                       AS policy_mismatches,
    AVG(score)                                      AS avg_score,
    SUM(CASE WHEN node = 'Node 2 - score over cutoff' THEN 1 ELSE 0 END) AS held_by_score
FROM mon_base
GROUP BY payment_date
ORDER BY payment_date;
