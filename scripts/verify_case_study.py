"""Recompute the checks reported in Section 6.4 of the manuscript.

The plant data are confidential and are NOT part of this repository. They are
available from the corresponding author on reasonable request with the
permission of the company. Usage:

    python scripts/verify_case_study.py PATH_TO_PLANT_DATA.xlsx

Exit status 0: every value matched. Exit status 1: the mismatches are listed.
``--quantile`` changes q and is meant as a failure control (q = 0.975 must fail).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gbramewma.config import (CASE_CHART, CASE_FORECAST, CASE_RESTART,  # noqa: E402
                              DESIGN_GRID_QUANTILE, DESIGN_GRID_REFERENCE, TRAINING_SHARE)
from gbramewma.core import (ChartSettings, clean_records, design_alternatives,  # noqa: E402
                            run_scheme)

EXPECTED = {
    "samples": 1484, "removed incomplete": 15, "training samples": 1187,
    "robust T2 limit": 12.934, "RA-MEWMA limit": 2.994,
    "training signals T2": 61, "training signals RA": 37,
    "test signals T2": 30, "test signals RA": 27, "test signals both": 14,
    "T2 training start-up": "7/50", "T2 training other": "54/1113",
    "T2 test start-up": "6/13", "T2 test other": "24/284",
    "RA training start-up": "9/50", "RA training other": "28/1113",
    "RA test start-up": "5/13", "RA test other": "22/284",
    "sample 1249 T2": 571.7, "sample 1249 A": 55.02, "test MAPE (%)": 9.083,
    "Table 12 grid (T2/RA)": "30/26 15/19 3/4 | 30/27 16/22 2/4 | 31/33 20/22 15/3",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--quantile", type=float, default=CASE_CHART.quantile)
    args = ap.parse_args()
    raw = pd.read_excel(args.data) if args.data.endswith("xlsx") else pd.read_csv(args.data)
    clean = clean_records(raw, "Time", ["H2O", "N", "P2O5", "K2O"], False)
    n_train = int(math.floor(TRAINING_SHARE * len(clean.frame)))
    cs = ChartSettings(CASE_CHART.reference_share, args.quantile, CASE_CHART.lambda_min,
                       CASE_CHART.lambda_max, CASE_CHART.mcd_random_state)
    res = run_scheme(clean, n_train, CASE_FORECAST, cs, CASE_RESTART)
    mon = res.phase == "monitoring"
    trn = ~mon
    ft, fa = res.t2 > res.ucl_t2, res.a > res.ucl_a
    got = {"samples": len(clean.frame), "removed incomplete": clean.n_incomplete,
           "training samples": n_train, "robust T2 limit": round(res.ucl_t2, 3),
           "RA-MEWMA limit": round(res.ucl_a, 3), "training signals T2": int(ft[trn].sum()),
           "training signals RA": int(fa[trn].sum()), "test signals T2": int(ft[mon].sum()),
           "test signals RA": int(fa[mon].sum()), "test signals both": int((ft & fa & mon).sum())}
    for name, flag in [("T2", ft), ("RA", fa)]:
        for period, m in [("training", trn), ("test", mon)]:
            s, o = res.restart & m, ~res.restart & m
            got[f"{name} {period} start-up"] = f"{(flag & s).sum()}/{s.sum()}"
            got[f"{name} {period} other"] = f"{(flag & o).sum()}/{o.sum()}"
    i = int(np.flatnonzero(res.sample_index == 1249)[0])
    got["sample 1249 T2"] = round(float(res.t2[i]), 1)
    got["sample 1249 A"] = round(float(res.a[i]), 2)
    got["test MAPE (%)"] = round(res.accuracy_test["MAPE (%)"], 3)
    grid = design_alternatives(res, cs, DESIGN_GRID_REFERENCE, DESIGN_GRID_QUANTILE)
    cells = [f"{a}/{b}" for a, b in zip(grid["Robust T² signals"], grid["RA-MEWMA signals"])]
    got["Table 12 grid (T2/RA)"] = " | ".join(" ".join(cells[k:k + 3]) for k in (0, 3, 6))
    bad = [(k, EXPECTED[k], got[k]) for k in EXPECTED if got[k] != EXPECTED[k]]
    for k in EXPECTED:
        print(f"{'OK  ' if got[k] == EXPECTED[k] else 'FAIL'} {k}: {got[k]}")
    if bad:
        print(f"\n{len(bad)} mismatch(es)")
        return 1
    print("\nAll values match Section 6.4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
