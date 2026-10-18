import numpy as np
from sklearn.metrics import roc_auc_score

from pfloat import metrics


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    s = rng.normal(size=5000)
    assert metrics.psi_from_scores(s, s) < 1e-9


def test_psi_detects_a_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(size=20000)
    assert metrics.psi_from_scores(base, base + 0.5) > 0.1


def test_ks_and_auc_with_ties_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.random(5000) < 0.05
    score = np.round(rng.random(5000) * 20 + 5 * y)  # integer scores => many ties
    assert abs(metrics.auc(y, score) - roc_auc_score(y, score)) < 1e-12
    assert 0 < metrics.ks(y, score) < 1


def test_assign_decile_is_monotonic_and_bounded():
    cut = metrics.decile_cutoffs(np.arange(1000))
    d = metrics.assign_decile(np.array([-5, 0, 500, 999, 5000]), cut)
    assert list(d) == sorted(d)
    assert d.min() == 1 and d.max() == 10
