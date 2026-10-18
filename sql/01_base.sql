-- =============================================================================
-- mon_base: one row per production payment.
--
-- Joins what was decided (ledger) with what happened (returns), and
-- re-derives the policy independently from raw tables so that any gap
-- between the rules and the production decision shows up as a mismatch.
--
-- Variables: as_of, prod_start, cutoff, amount_cap, prior_return_days
-- =============================================================================
CREATE OR REPLACE TABLE mon_base AS
WITH prod AS (
    SELECT
        p.*,
        CAST(p.initiated_at AS DATE)                 AS payment_date,
        CAST(date_trunc('week', p.initiated_at) AS DATE) AS payment_week
    FROM payments p
    WHERE p.initiated_at >= getvariable('prod_start')
),

-- Latest monthly delinquency snapshot at or before the payment.
status_asof AS (
    SELECT p.payment_id, s.delinquency_status
    FROM prod p
    ASOF LEFT JOIN account_status s
      ON p.account_id = s.account_id
     AND p.initiated_at >= s.as_of_date
),

-- Returns known strictly before each payment, same account, within the window.
account_returns AS (
    SELECT rp.account_id, r.returned_at
    FROM returns r
    JOIN payments rp USING (payment_id)
),
prior_returns AS (
    SELECT p.payment_id, COUNT(*) AS prior_returns
    FROM prod p
    JOIN account_returns ar
      ON ar.account_id = p.account_id
     AND ar.returned_at <  p.initiated_at
     AND ar.returned_at >= p.initiated_at - to_days(CAST(getvariable('prior_return_days') AS INTEGER))
    GROUP BY p.payment_id
),

joined AS (
    SELECT
        p.payment_id,
        p.account_id,
        p.payment_type,
        p.channel,
        p.amount,
        p.initiated_at,
        p.payment_date,
        p.payment_week,
        s.delinquency_status,
        COALESCE(pr.prior_returns, 0)             AS prior_returns,

        l.decision_id,
        l.score,
        l.node,
        COALESCE(l.released, FALSE)               AS released,
        l.decision_id IS NOT NULL                 AS has_decision,

        r.return_code,
        r.returned_at,
        date_diff('day', p.initiated_at, r.returned_at) AS days_to_return,
        date_diff('day', p.payment_date, getvariable('as_of')) AS days_since_payment,

        -- Spend/recovery stand in for card-transaction and recovery data.
        p.amount * o.potential_spend_share * (1 - o.recovery_rate) AS loss_if_released_and_returned
    FROM prod p
    LEFT JOIN status_asof   s  USING (payment_id)
    LEFT JOIN prior_returns pr USING (payment_id)
    LEFT JOIN ledger        l  USING (payment_id)
    LEFT JOIN returns       r  USING (payment_id)
    LEFT JOIN outcomes      o  USING (payment_id)
),

policy AS (
    SELECT
        j.*,
        CASE
            WHEN j.score IS NULL                                       THEN 'Hold - unscored'
            WHEN j.delinquency_status IS NULL                          THEN 'Node 1 - account status unknown'
            WHEN j.delinquency_status NOT IN ('NOT_DUE', 'CURRENT_DUE') THEN 'Node 1 - delinquent'
            WHEN j.prior_returns > 0                                   THEN 'Node 1 - prior return'
            WHEN j.amount >= getvariable('amount_cap')                 THEN 'Node 1 - amount at or over cap'
            WHEN j.score > getvariable('cutoff')                       THEN 'Node 2 - score over cutoff'
            ELSE 'Release'
        END AS expected_node
    FROM joined j
),

baseline AS (
    SELECT quantile_cont(score, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) AS cutoffs
    FROM baseline_scores
)

SELECT
    p.*,
    p.expected_node = 'Release'                                   AS expected_release,
    NOT p.has_decision OR (p.expected_node = 'Release') <> p.released AS policy_mismatch,
    p.score IS NOT NULL                                           AS scored,
    p.score / 1000.0                                              AS predicted_return_prob,

    -- Performance: NULL until the payment is old enough to judge.
    CASE WHEN p.days_since_payment >= 7  THEN COALESCE(p.days_to_return <= 7,  FALSE)::INT END AS return_7d,
    CASE WHEN p.days_since_payment >= 14 THEN COALESCE(p.days_to_return <= 14, FALSE)::INT END AS return_14d,
    CASE WHEN p.days_since_payment >= 30 THEN COALESCE(p.days_to_return <= 30, FALSE)::INT END AS return_30d,
    CASE WHEN p.days_since_payment >= 60 THEN COALESCE(p.days_to_return <= 60, FALSE)::INT END AS return_60d,
    p.returned_at IS NOT NULL                                     AS returned_to_date,

    -- Realised loss only where credit was released; for held payments this
    -- is the loss that holding avoided.
    CASE WHEN p.returned_at IS NOT NULL AND p.released     THEN p.loss_if_released_and_returned ELSE 0 END AS realised_loss,
    CASE WHEN p.returned_at IS NOT NULL AND NOT p.released THEN p.loss_if_released_and_returned ELSE 0 END AS avoided_loss,

    CASE
        WHEN p.score IS NULL  THEN '0 unscored'
        WHEN p.score <= 5     THEN '01-05'
        WHEN p.score <= 10    THEN '06-10'
        WHEN p.score <= 15    THEN '11-15'
        WHEN p.score <= 20    THEN '16-20'
        WHEN p.score <= 30    THEN '21-30'
        WHEN p.score <= 50    THEN '31-50'
        WHEN p.score <= 100   THEN '51-100'
        ELSE '101+'
    END AS score_band,

    -- Fixed-cutoff decile (1 = lowest risk), same rule as metrics.assign_decile.
    CASE WHEN p.score IS NULL THEN NULL
         ELSE 1 + len(list_filter(b.cutoffs, c -> c < p.score)) END AS score_decile
FROM policy p
CROSS JOIN baseline b;
