"""Part 2 of the bundle, run in parallel with `export_bundle.py --skip-test`: LightGBM p for every test pair
(kept at home in work/) and the uncertain band (bundle/rr_test.parquet). Same logic as export_bundle.py."""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_bundle import predict_lgb  # noqa: E402

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--models", default="full_v3")
ap.add_argument("--test-feat", default="cache/feat/test.parquet")
ap.add_argument("--band", type=float, nargs=2, default=[0.02, 0.98])
ap.add_argument("--out", default=str(HERE / "bundle_test"))
a = ap.parse_args()
boosters = [lgb.Booster(model_file=str(f)) for f in sorted((R / "runs/models").glob(f"{a.models}_fold*.txt"))]
tp = predict_lgb(pl.scan_parquet(R / a.test_feat), boosters, keep=("s1", "cand"))
(HERE / "work").mkdir(exist_ok=True)
tp.write_parquet(HERE / "work" / f"test_p_{a.models}.parquet")
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tb = tp.filter(pl.col("p").is_between(*a.band))
tb.write_parquet(out / "rr_test.parquet")
print(f"rr_test {tb.height:,} of {tp.height:,} test pairs ({tb.height / tp.height * 100:.2f}%)")
