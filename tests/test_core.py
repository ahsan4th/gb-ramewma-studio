"""Unit tests of the computational core (manuscript Eqs. (7)-(14)).

Each test checks a value that a wrong implementation would get wrong; the
``invalid`` tests feed settings that must raise an error.
"""
import math

import numpy as np
import pandas as pd
import pytest

from gbramewma.config import CASE_CHART, CASE_FORECAST, CASE_RESTART
from gbramewma.core import (ChartSettings, PhaseIEstimates, RestartSettings,
                            clean_records, contributions, lagged_design,
                            ra_mewma, restart_mask, robust_t2, run_scheme,
                            type1_quantile)
from gbramewma.demo import make_demo_frame


def _est(mu, sigma, d_med):
    sigma = np.asarray(sigma, float)
    return PhaseIEstimates(np.asarray(mu, float), sigma, np.linalg.inv(sigma), 10, 20, d_med)


def test_type1_quantile_is_the_kth_order_statistic():
    x = np.arange(1, 350)                      # 349 values, k = ceil(0.95 * 349) = 332
    assert type1_quantile(x, 0.95) == 332
    assert type1_quantile(x[::-1], 0.95) == 332


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
def test_type1_quantile_invalid_q(q):
    with pytest.raises(ValueError):
        type1_quantile([1.0, 2.0, 3.0], q)


def test_contributions_match_eq14():
    s_inv = np.diag([1 / 4, 1.0, 1 / 9])
    np.testing.assert_allclose(contributions(np.array([2.0, 1.0, 3.0]), s_inv)[0], [1 / 3] * 3)
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 4))
    si = np.linalg.inv(a @ a.T + 4 * np.eye(4))
    v = rng.normal(size=(5, 4))
    expected = np.array([[v[i, j] * (si @ v[i])[j] / (v[i] @ si @ v[i]) for j in range(4)]
                         for i in range(5)])
    np.testing.assert_allclose(contributions(v, si), expected)
    np.testing.assert_allclose(contributions(v, si).sum(axis=1), 1.0)


def test_robust_t2_is_squared_mahalanobis_distance():
    sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    est = _est([1.0, -1.0], sigma, 1.0)
    e = np.array([[1.0, -1.0], [3.0, 0.0]])
    d = e[1] - est.mu
    np.testing.assert_allclose(robust_t2(e, est), [0.0, d @ np.linalg.inv(sigma) @ d])


def test_ra_mewma_weight_and_recursion():
    cs = ChartSettings(0.7, 0.95, 0.05, 0.35, 0)
    est = _est([0.0, 0.0], np.eye(2), d_med=1.0)
    e = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    lam, z, a = ra_mewma(e, est, cs)
    # D = 0 gives lambda_min, D = D_med the midpoint, D = 3 gives 0.05 + 0.30 * 3/4 (Eq. 11)
    np.testing.assert_allclose(lam, [0.05, 0.20, 0.05 + 0.30 * 0.75])
    z2 = lam[2] * e[2] + (1 - lam[2]) * (lam[1] * e[1])     # Eq. 12, z_0 = 0
    np.testing.assert_allclose(z[2], z2)
    np.testing.assert_allclose(a[2], z2 @ z2)


@pytest.mark.parametrize("lmin,lmax", [(0.35, 0.05), (0.2, 0.2), (0.0, 0.3), (0.1, 1.2)])
def test_ra_mewma_invalid_bounds(lmin, lmax):
    with pytest.raises(ValueError):
        ra_mewma(np.zeros((3, 2)), _est([0, 0], np.eye(2), 1.0), ChartSettings(0.7, 0.95, lmin, lmax, 0))


def test_lagged_design_puts_lag_one_first():
    y = np.arange(12, dtype=float).reshape(6, 2)
    x, t = lagged_design(y, 2)
    np.testing.assert_array_equal(x[0], [2, 3, 0, 1])          # target row 2: lag 1 = row 1
    np.testing.assert_array_equal(t[0], [4, 5])


def test_restart_window_counts_samples_after_each_gap():
    t = pd.Series(pd.to_datetime(["2025-01-01 00:00", "2025-01-01 04:00", "2025-01-03 08:00",
                                  "2025-01-05 08:00", "2025-01-05 12:00", "2025-01-05 16:00"]))
    m = restart_mask(t, RestartSettings(24.0, 2))
    np.testing.assert_array_equal(m, [False, False, True, True, True, False])


def test_dates_month_first_and_day_first():
    df = pd.DataFrame({"t": ["01/02/25 00:00:00", "01/02/25 04:00:00"], "a": [1, 2], "b": [3, 4]})
    assert clean_records(df, "t", ["a", "b"], False).frame["t"].iloc[0] == pd.Timestamp("2025-01-02")
    assert clean_records(df, "t", ["a", "b"], True).frame["t"].iloc[0] == pd.Timestamp("2025-02-01")


def test_duplicate_times_invalid():
    df = pd.DataFrame({"t": ["2025-01-01 00:00", "2025-01-01 00:00"], "a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        clean_records(df, "t", ["a", "b"], False)


def test_limits_come_from_the_calibration_subset():
    clean = clean_records(make_demo_frame(), "Time", ["H2O", "N", "P2O5", "K2O"], False)
    res = run_scheme(clean, int(math.floor(0.8 * len(clean.frame))), CASE_FORECAST,
                     CASE_CHART, CASE_RESTART)
    cal, ref = res.phase == "calibration", res.phase == "reference"
    n_cal = int(cal.sum())
    k = math.ceil(CASE_CHART.quantile * n_cal)
    assert (res.t2[cal] > res.ucl_t2).sum() == n_cal - k
    assert (res.a[cal] > res.ucl_a).sum() == n_cal - k
    # failure control: a limit taken from the reference subset breaks this identity
    wrong = type1_quantile(res.t2[ref], CASE_CHART.quantile)
    assert (res.t2[cal] > wrong).sum() != n_cal - k
