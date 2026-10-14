"""Re-score the French test pairs from a France-only feature table and patch them into the test p table.
Usage: python fr_rescore.py --feat cache/feat/test_fr.parquet --out research/gpu_exp/work/test_p_full_v3fr2.parquet
"""
import argparse

import lightgbm as lgb
import numpy as np
import polars as pl

from io_utils import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--models", default="full_v3")
    ap.add_argument("--p-in", default="research/gpu_exp/work/test_p_full_v3.parquet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ms = [lgb.Booster(model_file=str(f)) for f in sorted((ROOT / "runs/models").glob(f"{a.models}_fold*.txt"))]
    cols = ms[0].feature_name()
    d = pl.read_parquet(ROOT / a.feat, columns=["s1", "cand"] + cols)
    X = d.select(pl.col(c).cast(pl.Float32) for c in cols).to_numpy()
    p = np.mean([m.predict(X, num_threads=48) for m in ms], axis=0).astype(np.float32)
    tp = pl.read_parquet(ROOT / a.p_in)
    new = d.select("s1", "cand", p_new=pl.Series(p))
    c = tp.join(new, on=["s1", "cand"])
    print(f"{a.feat}: {c.height:,} pairs; crossed 0.75 up {((c['p'] < .75) & (c['p_new'] >= .75)).sum():,}, "
          f"down {((c['p'] >= .75) & (c['p_new'] < .75)).sum():,}")
    tp.join(new, on=["s1", "cand"], how="left").with_columns(p=pl.coalesce("p_new", "p")).drop("p_new") \
      .write_parquet(ROOT / a.out)


if __name__ == "__main__":
    main()
