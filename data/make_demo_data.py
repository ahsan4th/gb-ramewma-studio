"""Write the synthetic demo data set to data/demo_granulation_synthetic.csv.

The plant data of the manuscript are confidential and are NOT included in this
repository; gbramewma/demo.py describes how the artificial series is built.

Run:  python data/make_demo_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gbramewma.demo import make_demo_frame  # noqa: E402

if __name__ == "__main__":
    out = Path(__file__).with_name("demo_granulation_synthetic.csv")
    frame = make_demo_frame()
    frame.to_csv(out, index=False, date_format="%Y-%m-%d %H:%M")
    print(f"wrote {out} with {len(frame)} rows")
