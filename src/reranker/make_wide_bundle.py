"""Wide rerank band: pairs with LightGBM p in [0.003, 0.997] that the first bundle (band [0.02, 0.98]) did not
include -> bundle/rr_stack_wide.parquet (OOF, for the stacker) and bundle/rr_test_wide.parquet (test).
Test pairs: union over the given test-p tables (base model and the French re-scored variants).
Usage: python make_wide_bundle.py --oof cache/oof/full_v3.parquet \
           --test-p work/test_p_full_v3.parquet work/test_p_full_v3fr.parquet work/test_p_full_v3fr2.parquet
"""
import argparse
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="cache/oof/full_v3.parquet")
    ap.add_argument("--test-p", nargs="+", default=["work/test_p_full_v3.parquet"])
    ap.add_argument("--band", type=float, nargs=2, default=[0.003, 0.997])
    a = ap.parse_args()
    band = lambda d: d.filter(pl.col("p").is_between(*a.band))  # noqa: E731
    B = HERE / "bundle"
    old_s = pl.read_parquet(B / "rr_stack.parquet").select("s1", "cand")
    val = pl.read_parquet(R / "cache/split/val_source1.parquet", columns=["entity_id"]).rename({"entity_id": "s1"})
    o = band(pl.read_parquet(R / a.oof, columns=["s1", "cand", "y", "p"])).join(old_s, on=["s1", "cand"], how="anti")
    o = o.join(val.with_columns(is_val=pl.lit(True)), on="s1", how="left").with_columns(pl.col("is_val").fill_null(False))
    o.write_parquet(B / "rr_stack_wide.parquet")
    old_t = pl.read_parquet(B / "rr_test.parquet").select("s1", "cand")
    ts = [pl.read_parquet(HERE / f) for f in a.test_p]
    t = (pl.concat([band(x).select("s1", "cand") for x in ts]).unique()
           .join(old_t, on=["s1", "cand"], how="anti").join(ts[0], on=["s1", "cand"]))
    t.write_parquet(B / "rr_test_wide.parquet")
    print(f"rr_stack_wide {o.height:,}  rr_test_wide {t.height:,}")


if __name__ == "__main__":
    main()
