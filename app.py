"""GB-RAMEWMA Studio.

Design, run and diagnose a residual-based multivariate quality-monitoring
scheme: Gradient Boosting forecast residuals monitored with a robust
Hotelling-type T^2 chart and a robust adaptive-lambda MEWMA (RA-MEWMA) chart.

Run locally with:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import math
import platform
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import sklearn
import streamlit as st
from plotly.subplots import make_subplots

from gbramewma import __version__
from gbramewma.config import (CASE_CHART, CASE_FORECAST, CASE_RESTART,
                              DESIGN_GRID_QUANTILE, DESIGN_GRID_REFERENCE,
                              TRAINING_SHARE)
from gbramewma.core import (ChartSettings, ForecastSettings, RestartSettings,
                            clean_records, design_alternatives, gap_starts,
                            run_scheme, score_new_samples, signal_table)
from gbramewma.demo import make_demo_frame
from gbramewma.diagnosis import (default_pair_table, default_single_table,
                                 suggest_unit)


# Design tokens ---------------------------------------------------------------
TEAL = "#0F5E5A"
INK = "#14211F"
MUTED = "#5B6B68"
LIMIT = "#B42318"
POINT = "#8A9A97"
CHAR_COLORS = ["#0072B2", "#009E73", "#E69F00", "#CC79A7",
               "#56B4E9", "#D55E00", "#7F7F7F", "#000000"]
BAND_REF = "rgba(15,94,90,0.05)"
BAND_CAL = "rgba(15,94,90,0.12)"
BAND_START = "rgba(230,159,0,0.16)"
FONT = dict(family="Source Sans Pro, Arial, sans-serif", size=14, color=INK)

st.set_page_config(page_title="GB-RAMEWMA Studio", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
      h1 {font-weight: 700; letter-spacing: -0.01em; margin-bottom: 0;}
      [data-testid="stMetricValue"] {font-size: 1.55rem;}
      [data-testid="stMetricLabel"] p {color: #5B6B68;}
      .lede {color: #5B6B68; font-size: 1.02rem; margin-top: .2rem;}
      .unit-box {border-left: 4px solid #0F5E5A; background: #EEF3F2;
                 padding: .7rem .9rem; border-radius: 4px;}
      .unit-box b {color: #0F5E5A;}
    </style>
    """,
    unsafe_allow_html=True,
)


# Data helpers ----------------------------------------------------------------
@st.cache_data(show_spinner=False)
def demo_frame() -> pd.DataFrame:
    return make_demo_frame()


@st.cache_data(show_spinner=False)
def excel_sheets(data: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


@st.cache_data(show_spinner=False)
def read_table(name: str, data: bytes, sheet: str | None) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet if sheet else 0)
    return pd.read_csv(io.BytesIO(data))


def guess_time_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if any(k in str(c).lower() for k in ("time", "date", "waktu", "tanggal")):
            return c
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.9 and not pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[0]


def guess_quality_columns(df: pd.DataFrame, time_col: str) -> list[str]:
    out = []
    for c in df.columns:
        if c == time_col or str(c).lower() in ("no", "nomor", "id", "index", "sample"):
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().mean() < 0.8:
            continue
        d = x.dropna().diff().dropna()
        if len(d) and (d == 1).all():   # running sample number
            continue
        out.append(c)
    return out


def with_gaps(t: pd.Series, y: np.ndarray, starts: np.ndarray) -> tuple[list, list]:
    """Insert a missing value inside each gap so that plotted lines break there."""
    tt, yy = list(pd.Series(t)), list(np.asarray(y, float))
    for s_ in sorted(starts, reverse=True):
        tt.insert(s_, tt[s_ - 1] + (tt[s_] - tt[s_ - 1]) / 2)
        yy.insert(s_, np.nan)
    return tt, yy


def lag1(x: np.ndarray) -> float:
    x = np.asarray(x, float) - np.mean(x)
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))


@st.cache_data(show_spinner=False, max_entries=6)
def fit_scheme(frame: pd.DataFrame, time_col: str, cols: tuple[str, ...], dayfirst: bool,
               training_share: float, fs: ForecastSettings, cs: ChartSettings,
               rs: RestartSettings):
    clean = clean_records(frame, time_col, list(cols), dayfirst)
    n_train = int(math.floor(training_share * len(clean.frame)))
    return clean, n_train, run_scheme(clean, n_train, fs, cs, rs)


@st.cache_data(show_spinner=False, max_entries=6)
def alternatives(_res, key: tuple, cs: ChartSettings) -> pd.DataFrame:
    return design_alternatives(_res, cs, DESIGN_GRID_REFERENCE, DESIGN_GRID_QUANTILE)


def fmt_time(t) -> str:
    return pd.Timestamp(t).strftime("%d %b %Y, %H:%M")


# Sidebar: data ---------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Upload a file", "Synthetic demo data"],
                      label_visibility="collapsed")
    raw, data_name = None, None
    if source == "Upload a file":
        up = st.file_uploader("Laboratory results (CSV or Excel)",
                              type=["csv", "xlsx", "xlsm", "xls"])
        if up is not None:
            data = up.getvalue()
            sheet = None
            if up.name.lower().endswith((".xlsx", ".xlsm", ".xls")):
                sheets = excel_sheets(data)
                sheet = st.selectbox("Sheet", sheets) if len(sheets) > 1 else sheets[0]
            raw, data_name = read_table(up.name, data, sheet), up.name
    else:
        raw, data_name = demo_frame(), "synthetic demo data"
        st.caption("Artificial data with the structure of a four-hourly NPK "
                   "quality record. Not plant measurements.")
    if raw is not None:
        st.subheader("Columns")
        tc_default = guess_time_column(raw)
        time_col = st.selectbox("Time of sampling", list(raw.columns),
                                index=list(raw.columns).index(tc_default))
        qc_default = guess_quality_columns(raw, time_col)
        quality_cols = st.multiselect(
            "Quality characteristics",
            [c for c in raw.columns if c != time_col], default=qc_default)
        dayfirst = st.checkbox("Dates are written day first (dd/mm/yy)", value=False)
    st.divider()
    st.caption(f"GB-RAMEWMA Studio {__version__}. Source code: "
               "[github.com/ahsan4th/gb-ramewma-studio]"
               "(https://github.com/ahsan4th/gb-ramewma-studio)")

# Header ----------------------------------------------------------------------
st.title("GB-RAMEWMA Studio")
st.markdown(
    '<p class="lede">Design a quality-monitoring scheme on your own laboratory data, '
    "run it, and trace each alarm to the unit of the line that should be inspected "
    "first.</p>", unsafe_allow_html=True)

if raw is None:
    st.info("Load a CSV or Excel file with one row per laboratory sample: a time "
            "column and two or more numeric quality characteristics. To look around "
            "first, choose the synthetic demo data in the sidebar.")
    st.stop()
if len(quality_cols) < 2:
    st.warning("Select at least two quality characteristics in the sidebar.")
    st.stop()

try:
    clean = clean_records(raw, time_col, quality_cols, dayfirst)
except ValueError as err:
    st.error(f"The data could not be prepared: {err}")
    st.stop()

data_key = (data_name, time_col, tuple(quality_cols), dayfirst, len(raw))
if st.session_state.get("data_key") != data_key:
    st.session_state["data_key"] = data_key
    st.session_state.pop("fit", None)
    st.session_state["single"] = default_single_table(quality_cols)
    st.session_state["pairs"] = default_pair_table(quality_cols)

tab_data, tab_design, tab_mon, tab_diag, tab_exp = st.tabs(
    ["1  Data", "2  Design the scheme", "3  Monitor", "4  Diagnose alarms", "5  Export"])

# 1 Data ----------------------------------------------------------------------
with tab_data:
    f = clean.frame
    times = f[time_col]
    gap_hours_now = st.session_state.get("gap_hours", CASE_RESTART.gap_hours)
    gs = gap_starts(times, gap_hours_now)
    share_now = st.session_state.get("training_share", TRAINING_SHARE)
    n_train_now = int(math.floor(share_now * len(f)))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Samples kept", f"{len(f):,}")
    m2.metric("Incomplete records removed", f"{clean.n_incomplete:,}")
    m3.metric(f"Gaps longer than {gap_hours_now:g} h", f"{len(gs)}")
    t0, t1 = times.iloc[0], times.iloc[-1]
    m4.metric("Period", f"{t0:%-d %b}–{t1:%-d %b %Y}" if t0.year == t1.year
              else f"{t0:%b %Y}–{t1:%b %Y}")

    fig = make_subplots(rows=len(quality_cols), cols=1, shared_xaxes=True,
                        vertical_spacing=0.025, subplot_titles=quality_cols)
    for j, c in enumerate(quality_cols):
        xg, yg = with_gaps(times, f[c].to_numpy(), gs)
        fig.add_trace(go.Scatter(x=xg, y=yg, mode="lines", connectgaps=False,
                                 line=dict(color=CHAR_COLORS[j % 8], width=1.1),
                                 name=c, showlegend=False), row=j + 1, col=1)
    for s in gs:
        fig.add_vrect(x0=times.iloc[s - 1], x1=times.iloc[s], fillcolor="#D9DEDD",
                      opacity=0.55, line_width=0)
    if 0 < n_train_now < len(f):
        fig.add_vline(x=times.iloc[n_train_now], line_dash="dash", line_color=INK,
                      line_width=1)
    fig.update_layout(height=150 * len(quality_cols) + 40, margin=dict(l=10, r=10, t=30, b=10),
                      font=FONT, plot_bgcolor="white", paper_bgcolor="white")
    fig.update_annotations(font=dict(size=13, color=INK), x=0.005, xanchor="left")
    fig.update_xaxes(showgrid=False, linecolor="#C9D1CF")
    fig.update_yaxes(gridcolor="#EEF1F0", zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption("Grey bands: gaps in the records. Dashed line: start of the "
               "monitoring period for the training share set in step 2.")

    desc = pd.DataFrame({
        "Mean": f[quality_cols].mean(), "SD": f[quality_cols].std(),
        "Min": f[quality_cols].min(), "Median": f[quality_cols].median(),
        "Max": f[quality_cols].max(),
        "Lag-1 autocorrelation": [lag1(f[c].to_numpy()) for c in quality_cols]})
    st.table(desc.style.format("{:.3f}"))

# 2 Design --------------------------------------------------------------------
with tab_design:
    left, right = st.columns([1, 2.3], gap="large")
    with left:
        with st.form("design"):
            st.markdown("**Training and residual generator**")
            training_share = st.slider("Training share", 0.50, 0.95,
                                       float(st.session_state.get("training_share", TRAINING_SHARE)),
                                       0.01)
            n_tr = int(math.floor(training_share * len(clean.frame)))
            st.caption(f"{n_tr:,} training samples, up to "
                       f"{fmt_time(clean.frame[time_col].iloc[n_tr - 1])}")
            window = st.number_input("Lagged samples w", 2, 96, CASE_FORECAST.window)
            c1, c2 = st.columns(2)
            n_est = c1.number_input("Trees M", 20, 2000, CASE_FORECAST.n_estimators, 10)
            lr = c2.number_input("Learning rate", 0.005, 1.0, CASE_FORECAST.learning_rate,
                                 0.005, format="%.3f")
            depth = c1.number_input("Tree depth", 1, 8, CASE_FORECAST.max_depth)
            seed = c2.number_input("Random seed", 0, 10_000, CASE_FORECAST.random_state)
            st.markdown("**Phase I estimates and limits**")
            ref_share = st.slider("Reference share", 0.50, 0.90, CASE_CHART.reference_share, 0.05)
            q = st.select_slider("Calibration quantile q", [0.90, 0.95, 0.975, 0.99, 0.995],
                                 value=CASE_CHART.quantile)
            c3, c4 = st.columns(2)
            lmin = c3.number_input("λ min", 0.01, 0.90, CASE_CHART.lambda_min, 0.01, format="%.2f")
            lmax = c4.number_input("λ max", 0.02, 1.00, CASE_CHART.lambda_max, 0.01, format="%.2f")
            st.markdown("**Start-up periods**")
            c5, c6 = st.columns(2)
            gap_h = c5.number_input("Gap longer than (h)", 1.0, 720.0, CASE_RESTART.gap_hours, 1.0)
            win_s = c6.number_input("Start-up samples", 1, 48, CASE_RESTART.window_samples)
            submitted = st.form_submit_button("Fit scheme", type="primary", width="stretch")
        if submitted:
            st.session_state["training_share"] = training_share
            st.session_state["gap_hours"] = gap_h
            fs = ForecastSettings(int(window), int(n_est), float(lr), int(depth), int(seed))
            cs = ChartSettings(float(ref_share), float(q), float(lmin), float(lmax),
                               CASE_CHART.mcd_random_state)
            rs = RestartSettings(float(gap_h), int(win_s))
            try:
                with st.spinner("Fitting the residual generator and the Phase I estimates"):
                    st.session_state["fit"] = fit_scheme(
                        raw, time_col, tuple(quality_cols), dayfirst, float(training_share),
                        fs, cs, rs)
                st.session_state["fit_settings"] = (fs, cs, rs, float(training_share))
            except ValueError as err:
                st.session_state.pop("fit", None)
                st.error(f"The scheme could not be fitted: {err}")

    with right:
        if "fit" not in st.session_state:
            st.info("Set the parameters and press **Fit scheme**. The values shown are "
                    "those of the case study in the accompanying paper.")
        else:
            _, n_train, res = st.session_state["fit"]
            mon = res.phase == "monitoring"
            trn = ~mon
            ft, fa = res.t2 > res.ucl_t2, res.a > res.ucl_a
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Robust T² limit", f"{res.ucl_t2:.3f}")
            k2.metric("RA-MEWMA limit", f"{res.ucl_a:.3f}")
            k3.metric("Signals T² / RA", f"{ft[mon].sum()} / {fa[mon].sum()}",
                      help="Monitoring period: robust T² / RA-MEWMA")
            k4.metric("Expected count", f"{(1 - res.chart_settings.quantile) * mon.sum():.1f}",
                      help="If the process is in control: (1 − q) × number of monitoring samples")
            summary = pd.DataFrame({
                "Robust T²": [f"{res.ucl_t2:.3f}", f"{ft[trn].sum()} of {trn.sum()}",
                              f"{ft[mon].sum()} of {mon.sum()}",
                              f"{(ft & res.restart & mon).sum()} of {(res.restart & mon).sum()}"],
                "RA-MEWMA": [f"{res.ucl_a:.3f}", f"{fa[trn].sum()} of {trn.sum()}",
                             f"{fa[mon].sum()} of {mon.sum()}",
                             f"{(fa & res.restart & mon).sum()} of {(res.restart & mon).sum()}"]},
                index=["Control limit", "Signals, training (Phase I)",
                       "Signals, monitoring (Phase II)", "Signals in start-up windows"])
            st.table(summary)
            a1, a2 = st.columns(2)
            with a1:
                st.markdown("**Forecast accuracy, monitoring period**")
                acc = res.accuracy_by_char.copy()
                acc.loc["All"] = res.accuracy_test
                st.table(acc.style.format({"MAPE (%)": "{:.3f}", "RMSE": "{:.4f}",
                                           "MAE": "{:.4f}"}))
            with a2:
                st.markdown("**Lag-1 autocorrelation, training period**")
                Y = clean.frame[quality_cols].to_numpy(float)
                ac = pd.DataFrame({
                    "Measurements": [lag1(Y[:n_train, j]) for j in range(len(quality_cols))],
                    "Residuals": [lag1(res.residuals[trn][:, j]) for j in range(len(quality_cols))]},
                    index=quality_cols)
                st.table(ac.style.format("{:.3f}"))
            st.markdown("**Compare design alternatives**")
            st.caption("Monitoring-period signals for other reference shares and quantiles, "
                       "with the residual generator kept fixed. Use it to choose the alarm "
                       "load before fixing the scheme.")
            if st.button("Compute alternatives"):
                st.session_state["alt"] = alternatives(
                    res, (data_key, st.session_state["fit_settings"]), res.chart_settings)
            if "alt" in st.session_state:
                alt = st.session_state["alt"].copy()
                alt["Reference share"] = (100 * alt["Reference share"]).round().astype(int).astype(str) + "%"
                st.table(alt.style.hide(axis="index").format(
                    {"Quantile q": "{:.3f}", "Expected if in control": "{:.1f}"}))

# Shared plotting -------------------------------------------------------------
def dominant(res, which: str) -> np.ndarray:
    v = (res.residuals - res.estimates.mu) if which == "t2" else res.z
    c = v * (v @ res.estimates.sigma_inv)
    return np.argmax(c, axis=1)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def chart_figure(res, show_all: bool, shade_start: bool) -> go.Figure:
    sel = np.ones(len(res.t2), bool) if show_all else (res.phase == "monitoring")
    t = res.times[sel].reset_index(drop=True)
    gstarts = gap_starts(t, res.restart_settings.gap_hours)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=["Robust Hotelling-type T²", "RA-MEWMA statistic A"])
    for r, (stat, ucl, which) in enumerate([(res.t2, res.ucl_t2, "t2"), (res.a, res.ucl_a, "a")], 1):
        s = stat[sel]
        dom = dominant(res, which)[sel]
        top = max(3.2 * ucl, float(np.quantile(s, 0.995)) * 1.08)
        clipped = np.minimum(s, top)
        xg, yg = with_gaps(t, clipped, gstarts)
        fig.add_trace(go.Scatter(x=xg, y=yg, mode="lines", connectgaps=False,
                                 line=dict(color="#C3CCCA", width=0.8),
                                 hoverinfo="skip", showlegend=False), row=r, col=1)
        normal = s <= ucl
        fig.add_trace(go.Scattergl(x=t[normal], y=s[normal], mode="markers",
                                   marker=dict(color=POINT, size=4), name="Below limit",
                                   showlegend=(r == 1), legendgroup="ok",
                                   hovertemplate="%{x|%d %b %Y %H:%M}<br>%{y:.3f}<extra></extra>"),
                      row=r, col=1)
        for j, c in enumerate(res.quality_cols):
            m = (~normal) & (dom == j)
            if not m.any():
                continue
            off = s[m] > top
            fig.add_trace(go.Scatter(
                x=t[m], y=np.minimum(s[m], top), mode="markers",
                marker=dict(color=CHAR_COLORS[j % 8], size=np.where(off, 11, 8),
                            symbol=np.where(off, "triangle-up", "circle"),
                            line=dict(color=INK, width=0.6)),
                name=f"Signal, largest contribution {c}", legendgroup=c, showlegend=(r == 1),
                customdata=s[m],
                hovertemplate="%{x|%d %b %Y %H:%M}<br>value %{customdata:.3f}<extra>" + c + "</extra>"),
                row=r, col=1)
        fig.add_hline(y=ucl, line=dict(color=LIMIT, dash="dash", width=1.4), row=r, col=1)
        xa, ya = ("x", "y") if r == 1 else (f"x{r}", f"y{r}")
        fig.add_annotation(x=1, xref=f"{xa} domain", y=ucl, yref=ya,
                           text=f"UCL {ucl:.3f}", showarrow=False, xanchor="right",
                           yanchor="bottom", font=dict(color=LIMIT, size=12))
        fig.update_yaxes(range=[0, top * 1.04], row=r, col=1)
        if (s > top).any():
            fig.add_annotation(x=0.995, xref=f"{xa} domain", y=1.0, yref=f"{ya} domain",
                               text=f"▲ off scale, max {s.max():.1f}", showarrow=False,
                               xanchor="right", yanchor="top", font=dict(size=11, color=MUTED))
    phase = res.phase[sel]
    if show_all:
        for name, color in [("reference", BAND_REF), ("calibration", BAND_CAL)]:
            for a, b in runs(phase == name):
                fig.add_vrect(x0=t.iloc[a], x1=t.iloc[b], fillcolor=color, line_width=0, layer="below")
    if shade_start:
        # one band per start-up window; a band never spans a gap in the records
        rest = res.restart[sel]
        new_run = np.zeros(len(t), bool)
        new_run[gstarts] = True
        i = 0
        while i < len(t):
            if rest[i]:
                j = i
                while j + 1 < len(t) and rest[j + 1] and not new_run[j + 1]:
                    j += 1
                fig.add_vrect(x0=t.iloc[i] - pd.Timedelta(hours=2),
                              x1=t.iloc[j] + pd.Timedelta(hours=2), fillcolor=BAND_START,
                              line_width=0, layer="below")
                i = j + 1
            else:
                i += 1
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=36, b=70), font=FONT,
                      plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(orientation="h", yanchor="top", y=-0.07, x=0))
    fig.update_annotations(selector=dict(text="Robust Hotelling-type T²"), x=0.005, xanchor="left")
    fig.update_annotations(selector=dict(text="RA-MEWMA statistic A"), x=0.005, xanchor="left")
    fig.update_xaxes(showgrid=False, linecolor="#C9D1CF")
    fig.update_yaxes(gridcolor="#EEF1F0", zeroline=False)
    return fig


def need_fit() -> bool:
    if "fit" not in st.session_state:
        st.info("Fit the scheme in step 2 first.")
        return True
    return False


# 3 Monitor -------------------------------------------------------------------
with tab_mon:
    if not need_fit():
        _, n_train, res = st.session_state["fit"]
        c1, c2, _ = st.columns([2.2, 1.6, 2.2])
        show_all = c1.radio("Samples", ["Monitoring period", "All samples"], horizontal=True) == "All samples"
        shade = c2.checkbox("Shade start-up windows", value=True)
        st.plotly_chart(chart_figure(res, show_all, shade), width="stretch",
                        config={"displaylogo": False})
        st.caption("Coloured points exceed the limit; the colour is the characteristic with "
                   "the largest contribution (Eq. 14). Amber bands: first samples after a gap "
                   "in the records. With all samples shown, the shaded blocks mark the "
                   "reference and calibration subsets.")
        with st.expander("Score new laboratory samples with the fixed scheme"):
            st.caption("Upload results taken after the last loaded sample, with the same "
                       "columns. The residual generator, the Fast-MCD estimates and both "
                       "limits are not changed.")
            newf = st.file_uploader("New samples (CSV or Excel)", type=["csv", "xlsx"], key="new")
            if newf is not None:
                try:
                    new_df = read_table(newf.name, newf.getvalue(), None)
                    scored = score_new_samples(res, clean, new_df)
                    st.dataframe(scored, hide_index=True, width="stretch")
                except ValueError as err:
                    st.error(f"The new samples could not be scored: {err}")

# 4 Diagnose ------------------------------------------------------------------
with tab_diag:
    if not need_fit():
        _, n_train, res = st.session_state["fit"]
        with st.expander("Process knowledge used for diagnosis"):
            st.caption("Which unit of the line can move which characteristic. Pre-filled for "
                       "an NPK granulation line; edit for your own line.")
            st.session_state["single"] = st.data_editor(
                st.session_state["single"], hide_index=True, width="stretch",
                disabled=["Characteristic"], key="single_editor")
            st.session_state["pairs"] = st.data_editor(
                st.session_state["pairs"], hide_index=True, width="stretch", num_rows="dynamic",
                column_config={"Residual signs": st.column_config.SelectboxColumn(
                    options=["same", "opposite"])}, key="pairs_editor")
            pair_thr = st.slider("Use a pair rule when the second-largest contribution is at least (%)",
                                 10, 50, 25, 5)
        include_p1 = st.checkbox("Include training-period signals", value=False)
        tab = signal_table(res, ("reference", "calibration", "monitoring") if include_p1 else ("monitoring",))
        if tab.empty:
            st.success("No sample exceeds either limit in the selected period.")
        else:
            cols = res.quality_cols
            dom, unit = [], []
            for _, r in tab.iterrows():
                cp = {c: r[f"T² contribution {c} (%)"] for c in cols}
                rr = {c: r[f"Residual {c}"] for c in cols}
                d, u = suggest_unit(cp, rr, st.session_state["single"], st.session_state["pairs"],
                                    float(pair_thr))
                dom.append(d)
                unit.append(u)
            tab["Largest contribution"] = dom
            tab["Inspect first (hypothesis)"] = unit
            show = tab[["Time", "Sample", "Flagged by", "Start-up", "Robust T²", "RA-MEWMA A",
                        "Largest contribution", "Inspect first (hypothesis)"]]
            st.dataframe(show, hide_index=True, width="stretch", height=260,
                         column_config={"Time": st.column_config.DatetimeColumn(format="DD MMM YYYY, HH:mm"),
                                        "Robust T²": st.column_config.NumberColumn(format="%.2f"),
                                        "RA-MEWMA A": st.column_config.NumberColumn(format="%.2f"),
                                        "Inspect first (hypothesis)": st.column_config.TextColumn(width="large")})
            order = tab.sort_values("Robust T²", ascending=False).index
            pick = st.selectbox(
                "Signal to inspect", list(order),
                format_func=lambda i: (f"{fmt_time(tab.loc[i, 'Time'])} (sample {tab.loc[i, 'Sample']}, "
                                       f"robust T² {tab.loc[i, 'Robust T²']:.1f})"))
            r = tab.loc[pick]
            st.markdown(f'<div class="unit-box">Inspect first: <b>{r["Inspect first (hypothesis)"]}</b>'
                        f'<br><span style="color:#5B6B68">Largest contribution: {r["Largest contribution"]}. '
                        "A hypothesis to check against operating records, not a confirmed cause.</span></div>",
                        unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            for col_box, prefix, title in [(b1, "T² contribution", "Contributions to robust T² (%)"),
                                           (b2, "RA contribution", "Contributions to RA-MEWMA (%)")]:
                vals = [r[f"{prefix} {c} (%)"] for c in cols]
                fb = go.Figure(go.Bar(x=vals, y=cols, orientation="h",
                                      marker_color=[CHAR_COLORS[j % 8] for j in range(len(cols))],
                                      text=[f"{v:.0f}" if abs(v) >= 0.5 else "0" for v in vals],
                                      textposition="outside"))
                fb.update_layout(title=dict(text=title, font=dict(size=14)), height=260,
                                 margin=dict(l=10, r=30, t=40, b=10), font=FONT,
                                 plot_bgcolor="white", paper_bgcolor="white")
                fb.update_yaxes(autorange="reversed")
                fb.update_xaxes(gridcolor="#EEF1F0", zeroline=True, zerolinecolor="#9AA7A4",
                                range=[min(0.0, min(vals)) * 1.25 - 4, max(vals) * 1.2 + 6])
                col_box.plotly_chart(fb, width="stretch", config={"displaylogo": False})
            i = int(np.flatnonzero(res.sample_index == r["Sample"])[0])
            detail = pd.DataFrame({"Observed": res.observed[i], "Forecast": res.predicted[i],
                                   "Residual": res.residuals[i]}, index=cols)
            st.table(detail.style.format("{:.3f}"))
            st.caption(f"Adaptive weight λ_t at this sample: {res.lam[i]:.3f}. Contributions can be "
                       "negative when correlated characteristics move in opposite directions.")

# 5 Export --------------------------------------------------------------------
with tab_exp:
    if not need_fit():
        _, n_train, res = st.session_state["fit"]
        fs, cs, rs, share = st.session_state["fit_settings"]
        allstats = pd.DataFrame({"Sample": res.sample_index, "Time": res.times, "Phase": res.phase,
                                 "Start-up": res.restart, "Robust T²": res.t2, "RA-MEWMA A": res.a,
                                 "lambda_t": res.lam,
                                 "Robust T² signal": res.t2 > res.ucl_t2,
                                 "RA-MEWMA signal": res.a > res.ucl_a})
        for j, c in enumerate(res.quality_cols):
            allstats[f"Residual {c}"] = res.residuals[:, j]
        sig = signal_table(res, ("reference", "calibration", "monitoring"))
        scheme = {
            "application": f"GB-RAMEWMA Studio {__version__}",
            "data": {"file": data_name, "samples": int(len(clean.frame)),
                     "removed_incomplete": int(clean.n_incomplete),
                     "first": str(clean.frame[time_col].iloc[0]),
                     "last": str(clean.frame[time_col].iloc[-1]),
                     "quality_characteristics": res.quality_cols},
            "training": {"share": share, "n_train": int(n_train)},
            "forecast_settings": asdict(fs), "chart_settings": asdict(cs),
            "restart_settings": asdict(rs),
            "phase1": {"n_reference": int(res.estimates.n_ref),
                       "n_calibration": int(res.estimates.n_train - res.estimates.n_ref),
                       "mcd_location": res.estimates.mu.tolist(),
                       "mcd_scatter": res.estimates.sigma.tolist(),
                       "D_med": res.estimates.d_med},
            "limits": {"robust_T2": res.ucl_t2, "RA_MEWMA": res.ucl_a},
            "software": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__,
                         "plotly": plotly.__version__, "streamlit": st.__version__},
        }
        st.markdown("Download the results of the current scheme. Keep the scheme file with "
                    "the plant records: it documents the reference data, the estimates and "
                    "the limits that were fixed for routine use.")
        d1, d2, d3 = st.columns(3)
        d1.download_button("Signals (CSV)", sig.to_csv(index=False), "gbramewma_signals.csv",
                           "text/csv", width="stretch")
        d2.download_button("All statistics (CSV)", allstats.to_csv(index=False),
                           "gbramewma_statistics.csv", "text/csv", width="stretch")
        d3.download_button("Scheme summary (JSON)", json.dumps(scheme, indent=2, default=str),
                           "gbramewma_scheme.json", "application/json", width="stretch")
        st.json(scheme, expanded=False)
