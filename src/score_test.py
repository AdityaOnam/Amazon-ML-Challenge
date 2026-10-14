"""Score a test feature table with the averaged LightGBM fold models -> research/gpu_exp/work/test_p_<models>.parquet
(the per-pair test probability table used by the reranker stacker and stage 2).
Usage: python score_test.py --feat cache/feat/test.parquet --models full_v3b
"""
import argparse

import lightgbm as lgb
import numpy as np
import polars as pl

from io_utils import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="cache/feat/test.parquet")
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ms = [lgb.Booster(model_file=str(f)) for f in sorted((ROOT / "runs/models").glob(f"{a.models}_fold*.txt"))]
    cols = ms[0].feature_name()
    d = pl.read_parquet(ROOT / a.feat, columns=["s1", "cand"] + cols)
    out = []
    for i in range(0, d.height, 10_000_000):  # bounded memory
        X = d[i:i + 10_000_000].select(pl.col(c).cast(pl.Float32) for c in cols).to_numpy()
        out.append(np.mean([m.predict(X, num_threads=64) for m in ms], axis=0).astype(np.float32))
    path = ROOT / (a.out or f"research/gpu_exp/work/test_p_{a.models}.parquet")
    d.select("s1", "cand").with_columns(p=pl.Series(np.concatenate(out))).write_parquet(path)
    print(f"wrote {path}: {d.height:,} pairs")


if __name__ == "__main__":
    main()
