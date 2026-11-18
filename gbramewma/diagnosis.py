"""Link an alarm to the unit of the line that should be inspected first.

The rules encode process knowledge from the design of the line (which unit can
move which characteristic). The defaults below restate Sects. 3.1 and 7.1 of
the manuscript for a compound NPK granulation line; for any other line the
user edits the tables in the application. A suggestion is a hypothesis to be
checked against operating records, not a confirmed cause.
"""
from __future__ import annotations

import re

import pandas as pd

NPK_SINGLE = {
    "h2o": "Dryer and furnace",
    "n": "Ammonia feed, or the solid urea and ammonium sulfate feeders",
    "p2o5": "Slurry stage: phosphoric acid dosing",
    "k2o": "Potassium chloride feeder",
}
NPK_PAIRS = [
    # (characteristic A, characteristic B, residual signs, unit)
    ("n", "p2o5", "same", "Slurry stage: acid or ammonia dosing (N and P2O5 enter together)"),
    ("n", "p2o5", "opposite", "Feed proportions (mass balance between nutrient feeds)"),
    ("n", "k2o", "opposite", "Feed proportions (mass balance between nutrient feeds)"),
    ("p2o5", "k2o", "opposite", "Feed proportions (mass balance between nutrient feeds)"),
]


def _key(name: str) -> str:
    return re.sub(r"[^0-9a-z]", "", str(name).lower())


def default_single_table(quality_cols: list[str]) -> pd.DataFrame:
    """Characteristic -> unit, pre-filled for NPK column names."""
    return pd.DataFrame({
        "Characteristic": quality_cols,
        "Unit to inspect first": [NPK_SINGLE.get(_key(c), "") for c in quality_cols],
    })


def default_pair_table(quality_cols: list[str]) -> pd.DataFrame:
    """Pairs of characteristics that point to one unit, pre-filled for NPK."""
    by_key = {_key(c): c for c in quality_cols}
    rows = [{"Characteristic A": by_key[a], "Characteristic B": by_key[b],
             "Residual signs": s, "Unit to inspect first": u}
            for a, b, s, u in NPK_PAIRS if a in by_key and b in by_key]
    return pd.DataFrame(rows, columns=["Characteristic A", "Characteristic B",
                                       "Residual signs", "Unit to inspect first"])


def suggest_unit(contrib_pct: dict[str, float], residual: dict[str, float],
                 single: pd.DataFrame, pairs: pd.DataFrame,
                 pair_threshold_pct: float) -> tuple[str, str]:
    """Return (dominant characteristic(s), suggested unit).

    A pair rule applies when the second-largest contribution is at least
    ``pair_threshold_pct`` per cent; otherwise the dominant characteristic
    decides."""
    ranked = sorted(contrib_pct, key=contrib_pct.get, reverse=True)
    first, second = ranked[0], ranked[1]
    if contrib_pct[second] >= pair_threshold_pct:
        signs = "same" if residual[first] * residual[second] > 0 else "opposite"
        for _, r in pairs.iterrows():
            if {r["Characteristic A"], r["Characteristic B"]} == {first, second} \
                    and r["Residual signs"] == signs and str(r["Unit to inspect first"]).strip():
                return f"{first} + {second}", str(r["Unit to inspect first"])
    unit = single.loc[single["Characteristic"] == first, "Unit to inspect first"]
    text = str(unit.iloc[0]).strip() if len(unit) else ""
    return first, text if text else "No rule defined for this characteristic"
