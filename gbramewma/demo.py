"""Synthetic demo data with the structure of a four-hourly NPK quality record.

The plant data of the manuscript are confidential and are not included in this
repository. ``make_demo_frame`` builds an artificial series with the same
columns and similar summary statistics (means, standard deviations and
correlations of Table 1 of the manuscript), serial dependence, level changes
between months, heavy-tailed noise, stoppage gaps and two injected faults, so
that every step of the application can be tried. The values are not plant
measurements and cannot reproduce the reported results.
"""
import numpy as np
import pandas as pd

SEED = 20250101
COLS = ["H2O", "N", "P2O5", "K2O"]
MEAN = np.array([1.106, 15.899, 11.512, 12.443])      # Table 1 (published summary)
SD = np.array([0.309, 0.823, 1.399, 1.553])
CORR = np.array([[1.000, -0.216, -0.125, 0.148],
                 [-0.216, 1.000, 0.127, -0.477],
                 [-0.125, 0.127, 1.000, -0.229],
                 [0.148, -0.477, -0.229, 1.000]])
PHI = np.array([0.35, 0.45, 0.55, 0.20])                # lag-1 coefficients
N_SCHEDULED = 2136                                       # one year of 4-h slots


def make_demo_frame() -> pd.DataFrame:
    """Return the synthetic series (time column plus four characteristics)."""
    rng = np.random.default_rng(SEED)
    chol = np.linalg.cholesky(np.outer(SD, SD) * CORR)
    # Heavy-tailed innovations: multivariate t with 5 degrees of freedom.
    g = rng.standard_normal((N_SCHEDULED, 4)) @ chol.T
    scale = np.sqrt(rng.chisquare(5, N_SCHEDULED) / 3.0)[:, None]
    innov = 0.75 * g / scale
    x = np.zeros((N_SCHEDULED, 4))
    for t in range(1, N_SCHEDULED):
        x[t] = PHI * x[t - 1] + innov[t]
    # Month-to-month level changes (production campaigns).
    times = pd.date_range("2024-01-01 00:00", periods=N_SCHEDULED, freq="4h")
    month = times.month.to_numpy()
    level = rng.normal(0.0, 0.35, (12, 4)) * SD
    y = MEAN + x + level[month - 1]
    # Fault 1: joint N-P2O5 excursion (slurry-type) at one sample.
    y[1800, 1] += 5.0
    y[1800, 2] += 9.0
    # Fault 2: drift of the potassium chloride feed over two days.
    y[1950:1962, 3] += np.linspace(0.5, 3.0, 12)
    y[1950:1962, 1] -= np.linspace(0.2, 1.0, 12)
    y = np.round(np.clip(y, [0.2, 10.0, 5.0, 5.0], None), 1)
    df = pd.DataFrame(y, columns=COLS)
    df.insert(0, "Time", times)
    # Stoppages: remove runs of records, plus a few isolated incomplete rows.
    keep = np.ones(N_SCHEDULED, dtype=bool)
    for start, length in [(70, 160), (430, 9), (520, 15), (700, 120), (880, 18),
                          (1040, 80), (1250, 21), (1790, 9), (1802, 70), (2050, 50)]:
        keep[start:start + length] = False
    df = df.loc[keep].reset_index(drop=True)
    for i in rng.choice(len(df), 12, replace=False):
        df.loc[i, COLS] = np.nan
    return df
