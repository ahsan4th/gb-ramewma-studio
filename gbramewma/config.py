"""Settings used in the case study of the manuscript (Sects. 3.2, 4.1-4.6).

The application shows these values as the starting point of the design
step; every one of them can be changed in the interface.
"""
from .core import ChartSettings, ForecastSettings, RestartSettings

#: Share of the cleaned samples used for training (1,187 of 1,484 samples).
TRAINING_SHARE = 0.80

#: Gradient Boosting residual generator (Table 4 of the manuscript).
CASE_FORECAST = ForecastSettings(
    window=24,          # w, four days of four-hourly samples
    n_estimators=200,   # M
    learning_rate=0.04,  # nu
    max_depth=2,
    random_state=42,
)

#: Phase I split, calibration quantile and RA-MEWMA bounds (Sects. 4.4-4.6).
CASE_CHART = ChartSettings(
    reference_share=0.70,
    quantile=0.95,
    lambda_min=0.05,
    lambda_max=0.35,
    mcd_random_state=42,
)

#: Start-up samples: the first 24 h (six samples) after a gap longer than one day.
CASE_RESTART = RestartSettings(gap_hours=24.0, window_samples=6)

#: Grid of the sensitivity analysis (Table 12 of the manuscript).
DESIGN_GRID_REFERENCE = [0.60, 0.70, 0.80]
DESIGN_GRID_QUANTILE = [0.95, 0.975, 0.99]
