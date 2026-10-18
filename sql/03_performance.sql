-- =============================================================================
-- mon_performance: matured outcomes by week x score band x node x decision.
-- mature_Nd counts only payments old enough to judge at that window.
-- =============================================================================
CREATE OR REPLACE TABLE mon_performance AS
SELECT
    payment_week,
    score_band,
    node,
    released,
    COUNT(*)                         AS payments,
    SUM(amount)                      AS amount,
    SUM(predicted_return_prob)       AS sum_predicted,
    COUNT(predicted_return_prob)     AS n_predicted,
    COUNT(return_14d)                AS mature_14d,
    SUM(return_14d)                  AS returns_14d,
    COUNT(return_30d)                AS mature_30d,
    SUM(return_30d)                  AS returns_30d,
    SUM(CASE WHEN return_30d IS NOT NULL THEN predicted_return_prob END) AS sum_predicted_mature_30d,
    COUNT(return_60d)                AS mature_60d,
    SUM(return_60d)                  AS returns_60d,
    SUM(realised_loss)               AS realised_loss,
    SUM(avoided_loss)                AS avoided_loss
FROM mon_base
GROUP BY ALL;
