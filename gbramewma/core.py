"""Computational core of the GB-RAMEWMA monitoring procedure.

Equation numbers refer to the manuscript "Interactive monitoring-scheme design
and fault diagnosis for compound NPK granulation" (Sections 4.2-4.7):

* Gradient Boosting residual generator ........ Eqs. (4)-(5)
* Fast-MCD reference estimates ................. Eq. (6)
* Robust Hotelling-type T^2 and its limit ...... Eqs. (7)-(8)
* Robust adaptive-lambda MEWMA (RA-MEWMA) ...... Eqs. (9)-(13)
* Variable contributions ........................ Eq. (14)

Functions take every setting that can change a conclusion as an explicit
argument; the values used in the case study live in ``config.py``.
Invalid inputs raise ``ValueError`` instead of being repaired silently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ForecastSettings:
    """Residual generator (Sect. 4.2). ``window`` is w, the number of lags."""

    window: int
    n_estimators: int
    learning_rate: float
    max_depth: int
    random_state: int


@dataclass(frozen=True)
class ChartSettings:
    """Phase I split, calibration quantile and RA-MEWMA weight bounds."""

    reference_share: float
    quantile: float
    lambda_min: float
    lambda_max: float
    mcd_random_state: int


@dataclass(frozen=True)
class RestartSettings:
    """A gap longer than ``gap_hours`` is treated as a stoppage; the first
    ``window_samples`` samples after it are flagged as start-up samples."""

    gap_hours: float
    window_samples: int


# ---------------------------------------------------------------------------
# Data preparation (Sect. 3.2)
# ---------------------------------------------------------------------------
@dataclass
class CleanData:
    frame: pd.DataFrame          # time column + quality columns, complete rows only
    time_col: str
    quality_cols: list[str]
    n_raw: int
    n_incomplete: int
    dayfirst: bool


def clean_records(df: pd.DataFrame, time_col: str, quality_cols: list[str],
                  dayfirst: bool) -> CleanData:
    """Keep rows with all quality characteristics measured, sorted by time.

    Incomplete rows are removed, not imputed (as in the case study).
    ``dayfirst`` resolves dates such as 01/02/25 (1 February if True)."""
    if len(quality_cols) < 2:
        raise ValueError("Select at least two quality characteristics.")
    missing = [c for c in [time_col, *quality_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in the data: {missing}")
    out = df[[time_col, *quality_cols]].copy()
    out[time_col] = parse_times(out[time_col], dayfirst)
    for c in quality_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    n_raw = len(out)
    complete = out[quality_cols].notna().all(axis=1) & out[time_col].notna()
    out = out.loc[complete].sort_values(time_col, kind="stable").reset_index(drop=True)
    if out[time_col].duplicated().any():
        raise ValueError("Duplicate time stamps found; each sample needs a unique time.")
    return CleanData(out, time_col, list(quality_cols), n_raw, int((~complete).sum()), dayfirst)


_MONTH_FIRST = ["%m/%d/%y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m/%d/%y %H:%M",
                "%m/%d/%Y %H:%M", "%m/%d/%Y", "%m/%d/%y"]


def parse_times(values: pd.Series, dayfirst: bool) -> pd.Series:
    """Parse a time column with one format for every row.

    ISO 8601 is tried first, then slash formats (month first, or day first
    when ``dayfirst`` is True). Raises instead of guessing row by row."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    text = values.astype("string").str.strip()
    try:
        return pd.to_datetime(text, format="ISO8601")
    except (ValueError, TypeError):
        pass
    formats = [f.replace("%m/%d", "%d/%m") for f in _MONTH_FIRST] if dayfirst else _MONTH_FIRST
    for fmt in formats:
        try:
            return pd.to_datetime(text, format=fmt)
        except (ValueError, TypeError):
            continue
    raise ValueError("The time column could not be read with a single date format; "
                     "write times as YYYY-MM-DD HH:MM.")


def gap_starts(times: pd.Series, gap_hours: float) -> np.ndarray:
    """Indices of the first sample after a gap longer than ``gap_hours``."""
    dt = pd.Series(times).diff()
    return np.flatnonzero((dt > pd.Timedelta(hours=gap_hours)).to_numpy())


def restart_mask(times: pd.Series, rs: RestartSettings) -> np.ndarray:
    """Boolean mask of start-up samples: the first ``rs.window_samples``
    samples after each gap (24 h of four-hourly sampling in the case study)."""
    if rs.window_samples < 1:
        raise ValueError("The start-up window must contain at least one sample.")
    n = len(times)
    mask = np.zeros(n, dtype=bool)
    for s in gap_starts(times, rs.gap_hours):
        mask[s:min(n, s + rs.window_samples)] = True
    return mask


# ---------------------------------------------------------------------------
# Residual generator (Sects. 4.2-4.3)
# ---------------------------------------------------------------------------
def lagged_design(Y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Lagged inputs and targets.

    Row i (target time t = i + w) holds [y_{t-1}', y_{t-2}', ..., y_{t-w}'],
    i.e. lag 1 first, all characteristics of one lag next to each other."""
    n, p = Y.shape
    if window < 1 or n <= window:
        raise ValueError(f"Need more than w = {window} samples, got {n}.")
    X = np.empty((n - window, window * p))
    for i, t in enumerate(range(window, n)):
        X[i] = Y[t - window:t][::-1].reshape(-1)
    return X, Y[window:].copy()


@dataclass
class ForecastModel:
    model: MultiOutputRegressor
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    settings: ForecastSettings
    n_train: int


def fit_forecaster(Y: np.ndarray, n_train: int, fs: ForecastSettings) -> ForecastModel:
    """Fit one Gradient Boosting regressor per characteristic on the training
    period; feature and target standardization are fitted on it only."""
    X, T = lagged_design(Y, fs.window)
    n_fit = n_train - fs.window
    if n_fit < 10 * Y.shape[1]:
        raise ValueError("The training period is too short for the chosen window.")
    xs = StandardScaler().fit(X[:n_fit])
    ys = StandardScaler().fit(T[:n_fit])
    gb = GradientBoostingRegressor(
        n_estimators=fs.n_estimators,
        learning_rate=fs.learning_rate,
        max_depth=fs.max_depth,
        random_state=fs.random_state,
    )
    model = MultiOutputRegressor(gb).fit(xs.transform(X[:n_fit]), ys.transform(T[:n_fit]))
    return ForecastModel(model, xs, ys, fs, n_train)


def predict_one_step(fm: ForecastModel, Y: np.ndarray) -> np.ndarray:
    """Fitted values (training period) and rolling one-step-ahead forecasts
    (later samples) for targets w..n-1. The forecast of y_t uses only
    y_{t-1}, ..., y_{t-w}; parameters are not refitted (Eq. 5)."""
    X, _ = lagged_design(Y, fm.settings.window)
    return fm.y_scaler.inverse_transform(fm.model.predict(fm.x_scaler.transform(X)))


def accuracy(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    """MAPE (pairs with |y| > 1e-8), RMSE and MAE over all target pairs (Eqs. 1-3)."""
    e = y - yhat
    ok = np.abs(y) > 1e-8
    return {
        "MAPE (%)": float(100.0 * np.mean(np.abs(e[ok] / y[ok]))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAE": float(np.mean(np.abs(e))),
    }


# ---------------------------------------------------------------------------
# Phase I estimates and charts (Sects. 4.4-4.7)
# ---------------------------------------------------------------------------
def type1_quantile(x: np.ndarray, q: float) -> float:
    """Type-1 sample quantile: order statistic x_(k), k = ceil(q n) (Eq. 8)."""
    if not 0.0 < q < 1.0:
        raise ValueError("The calibration quantile must lie strictly between 0 and 1.")
    xs = np.sort(np.asarray(x, dtype=float))
    if xs.size == 0:
        raise ValueError("Empty calibration subset.")
    return float(xs[math.ceil(q * xs.size) - 1])


@dataclass
class PhaseIEstimates:
    mu: np.ndarray
    sigma: np.ndarray
    sigma_inv: np.ndarray
    n_ref: int
    n_train: int
    d_med: float


def phase1_estimates(E_train: np.ndarray, cs: ChartSettings) -> PhaseIEstimates:
    """Fast-MCD location and scatter from the reference subset (first
    ``reference_share`` of the training residuals, Eq. 6) and D_med (Eq. 10)."""
    n_train, p = E_train.shape
    n_ref = int(math.floor(cs.reference_share * n_train))
    if n_ref < 5 * p or n_train - n_ref < 20:
        raise ValueError("Reference or calibration subset too small; change the reference share.")
    mcd = MinCovDet(random_state=cs.mcd_random_state).fit(E_train[:n_ref])
    sigma = mcd.covariance_
    eig = np.linalg.eigvalsh(sigma)
    if eig.min() <= 0:
        raise ValueError("The Fast-MCD scatter matrix is not positive definite.")
    sigma_inv = np.linalg.inv(sigma)
    d = E_train[:n_ref] - mcd.location_
    t2_ref = np.einsum("ij,jk,ik->i", d, sigma_inv, d)
    d_med = float(np.median(np.sqrt(np.maximum(t2_ref, 0.0))))
    if not np.isfinite(d_med) or d_med <= 0:
        d_med = 1e-12  # numerical safeguard stated in Sect. 4.6
    return PhaseIEstimates(mcd.location_, sigma, sigma_inv, n_ref, n_train, d_med)


def robust_t2(E: np.ndarray, est: PhaseIEstimates) -> np.ndarray:
    """Squared robust Mahalanobis distances (Eq. 7)."""
    d = E - est.mu
    return np.einsum("ij,jk,ik->i", d, est.sigma_inv, d)


def ra_mewma(E: np.ndarray, est: PhaseIEstimates, cs: ChartSettings,
             z0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive weights, smoothed vectors and statistics A_t (Eqs. 9, 11-12).

    The recursion runs over the residuals in time order; ``z0`` continues a
    previous run (used when new samples are scored)."""
    if not 0.0 < cs.lambda_min < cs.lambda_max <= 1.0:
        raise ValueError("Require 0 < lambda_min < lambda_max <= 1.")
    d = E - est.mu
    D = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, est.sigma_inv, d), 0.0))
    lam = cs.lambda_min + (cs.lambda_max - cs.lambda_min) * D / (D + est.d_med)
    Z = np.empty_like(d)
    z = np.zeros(d.shape[1]) if z0 is None else np.asarray(z0, dtype=float).copy()
    for t in range(d.shape[0]):
        z = lam[t] * d[t] + (1.0 - lam[t]) * z
        Z[t] = z
    A = np.einsum("ij,jk,ik->i", Z, est.sigma_inv, Z)
    return lam, Z, A


def contributions(v: np.ndarray, sigma_inv: np.ndarray) -> np.ndarray:
    """Share of each characteristic in v' S^-1 v (Eq. 14); rows sum to one."""
    v = np.atleast_2d(v)
    q = np.einsum("ij,jk,ik->i", v, sigma_inv, v)
    return (v * (v @ sigma_inv)) / q[:, None]


# ---------------------------------------------------------------------------
# Whole scheme
# ---------------------------------------------------------------------------
@dataclass
class SchemeResult:
    times: pd.Series              # times of the residuals (targets w..n-1)
    observed: np.ndarray          # y_t for the residual rows
    predicted: np.ndarray         # fitted / one-step forecasts
    residuals: np.ndarray
    phase: np.ndarray             # "reference", "calibration" or "monitoring"
    restart: np.ndarray           # start-up flag per residual row
    t2: np.ndarray
    a: np.ndarray
    lam: np.ndarray
    z: np.ndarray
    ucl_t2: float
    ucl_a: float
    estimates: PhaseIEstimates
    model: ForecastModel
    accuracy_test: dict
    accuracy_by_char: pd.DataFrame
    quality_cols: list[str]
    sample_index: np.ndarray      # index in the cleaned series (0-based)
    chart_settings: ChartSettings
    restart_settings: RestartSettings


def run_scheme(clean: CleanData, n_train: int, fs: ForecastSettings,
               cs: ChartSettings, rs: RestartSettings) -> SchemeResult:
    """Steps 2-7 of the procedure on one data set."""
    Y = clean.frame[clean.quality_cols].to_numpy(dtype=float)
    n = len(Y)
    if not fs.window < n_train < n:
        raise ValueError("The training period must be longer than w and shorter than the data.")
    fm = fit_forecaster(Y, n_train, fs)
    P = predict_one_step(fm, Y)
    obs = Y[fs.window:]
    E = obs - P
    idx = np.arange(fs.window, n)
    is_train = idx < n_train
    est = phase1_estimates(E[is_train], cs)
    t2 = robust_t2(E, est)
    lam, Z, A = ra_mewma(E, est, cs)
    cal = slice(est.n_ref, est.n_train)
    ucl_t2 = type1_quantile(t2[:est.n_train][cal], cs.quantile)
    ucl_a = type1_quantile(A[:est.n_train][cal], cs.quantile)
    phase = np.where(~is_train, "monitoring",
                     np.where(np.arange(len(E)) < est.n_ref, "reference", "calibration"))
    restart = restart_mask(clean.frame[clean.time_col], rs)[fs.window:]
    test = ~is_train
    acc = accuracy(obs[test], P[test]) if test.any() else {}
    by_char = pd.DataFrame(
        [accuracy(obs[test, j], P[test, j]) for j in range(Y.shape[1])],
        index=clean.quality_cols) if test.any() else pd.DataFrame()
    return SchemeResult(
        times=clean.frame[clean.time_col].iloc[fs.window:].reset_index(drop=True),
        observed=obs, predicted=P, residuals=E, phase=phase, restart=restart,
        t2=t2, a=A, lam=lam, z=Z, ucl_t2=ucl_t2, ucl_a=ucl_a, estimates=est,
        model=fm, accuracy_test=acc, accuracy_by_char=by_char,
        quality_cols=list(clean.quality_cols), sample_index=idx,
        chart_settings=cs, restart_settings=rs)


def recalibrate(res: SchemeResult, cs: ChartSettings) -> tuple[float, float, np.ndarray, np.ndarray, PhaseIEstimates]:
    """Charts and limits for other Phase I settings, keeping the residuals.

    Used for the design-alternatives table (no refitting of the forecaster)."""
    is_train = res.phase != "monitoring"
    est = phase1_estimates(res.residuals[is_train], cs)
    t2 = robust_t2(res.residuals, est)
    _, _, A = ra_mewma(res.residuals, est, cs)
    cal = slice(est.n_ref, est.n_train)
    return (type1_quantile(t2[:est.n_train][cal], cs.quantile),
            type1_quantile(A[:est.n_train][cal], cs.quantile), t2, A, est)


def design_alternatives(res: SchemeResult, base: ChartSettings,
                        reference_shares: list[float], quantiles: list[float]) -> pd.DataFrame:
    """Monitoring-period signal counts for a grid of Phase I settings."""
    mon = res.phase == "monitoring"
    rows = []
    for r in reference_shares:
        for q in quantiles:
            cs = ChartSettings(r, q, base.lambda_min, base.lambda_max, base.mcd_random_state)
            u_t, u_a, t2, A, _ = recalibrate(res, cs)
            rows.append({"Reference share": r, "Quantile q": q,
                         "Robust T² signals": int((t2[mon] > u_t).sum()),
                         "RA-MEWMA signals": int((A[mon] > u_a).sum()),
                         "Expected if in control": round((1 - q) * int(mon.sum()), 1)})
    return pd.DataFrame(rows)


def signal_table(res: SchemeResult, phases: tuple[str, ...]) -> pd.DataFrame:
    """One row per flagged sample with its contributions (Eq. 14)."""
    sel = np.isin(res.phase, phases)
    flag_t = res.t2 > res.ucl_t2
    flag_a = res.a > res.ucl_a
    keep = sel & (flag_t | flag_a)
    d = res.residuals - res.estimates.mu
    c_t = contributions(d[keep], res.estimates.sigma_inv)
    c_a = contributions(res.z[keep], res.estimates.sigma_inv)
    rows = []
    for k, i in enumerate(np.flatnonzero(keep)):
        by = "Both" if flag_t[i] and flag_a[i] else ("Robust T²" if flag_t[i] else "RA-MEWMA")
        row = {"Sample": int(res.sample_index[i]), "Time": res.times.iloc[i],
               "Phase": res.phase[i], "Robust T²": float(res.t2[i]),
               "RA-MEWMA A": float(res.a[i]), "Flagged by": by,
               "Start-up": bool(res.restart[i]), "lambda_t": float(res.lam[i])}
        for j, c in enumerate(res.quality_cols):
            row[f"T² contribution {c} (%)"] = 100.0 * float(c_t[k, j])
        for j, c in enumerate(res.quality_cols):
            row[f"RA contribution {c} (%)"] = 100.0 * float(c_a[k, j])
        for j, c in enumerate(res.quality_cols):
            row[f"Residual {c}"] = float(res.residuals[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def score_new_samples(res: SchemeResult, clean: CleanData, new: pd.DataFrame) -> pd.DataFrame:
    """Phase II scoring of new laboratory results with the fixed scheme.

    The forecaster, the Fast-MCD estimates and both limits are not changed.
    New rows must be later than the last stored sample."""
    new_clean = clean_records(new, clean.time_col, clean.quality_cols, clean.dayfirst).frame
    last = clean.frame[clean.time_col].iloc[-1]
    if (new_clean[clean.time_col] <= last).any():
        raise ValueError("New samples must be later than the last sample already loaded.")
    w = res.model.settings.window
    allY = np.vstack([clean.frame[clean.quality_cols].to_numpy(float),
                      new_clean[clean.quality_cols].to_numpy(float)])
    hist = allY[-(len(new_clean) + w):]
    P = predict_one_step(res.model, hist)
    E = new_clean[clean.quality_cols].to_numpy(float) - P
    t2 = robust_t2(E, res.estimates)
    lam, Z, A = ra_mewma(E, res.estimates, res.chart_settings, z0=res.z[-1])
    c = contributions(E - res.estimates.mu, res.estimates.sigma_inv)
    out = new_clean.copy()
    out["Robust T²"] = t2
    out["RA-MEWMA A"] = A
    out["Robust T² signal"] = t2 > res.ucl_t2
    out["RA-MEWMA signal"] = A > res.ucl_a
    for j, col in enumerate(clean.quality_cols):
        out[f"T² contribution {col} (%)"] = 100 * c[:, j]
    return out
