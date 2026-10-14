"""Phase 3b: LightGBM pair scorer with GroupKFold by S1 -> out-of-fold probabilities.

Usage: python train.py --feat cache/feat/val.parquet [--folds 5] [--rounds 1500]
Writes cache/oof/{name}.parquet (s1, cand, y, p) and runs/models/{name}_fold{k}.txt
"""
import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from io_utils import ROOT

W = int(os.environ.get("SLURM_CPUS_PER_TASK", 32))
NON_FEAT = {"s1", "cand", "y"}
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=W, verbose=-1, max_bin=255)


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEAT]


def to_numpy(df, cols):
    return df.select(pl.col(c).cast(pl.Float32) for c in cols).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--name", default=None)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--s1-frac", type=float, default=1.0, help="train on this fraction of S1 groups (hash-sampled)")
    a = ap.parse_args()
    name = a.name or os.path.basename(a.feat).replace(".parquet", "")
    PARAMS["learning_rate"] = a.lr
    t0 = time.time()
    d = pl.read_parquet(ROOT / a.feat if not a.feat.startswith("/") else a.feat)
    if a.s1_frac < 1.0:
        d = d.filter((pl.col("s1").hash(seed=11) % 1000) < int(a.s1_frac * 1000))
    cols = feature_cols(d)
    # fold id by hashing S1 so every S1's candidates stay together
    fold = (d["s1"].hash(seed=7) % a.folds).to_numpy()
    X, y = to_numpy(d, cols), d["y"].to_numpy().astype(np.float32)
    print(f"{d.height:,} pairs, {len(cols)} features, pos rate {y.mean():.4f}", flush=True)
    oof = np.zeros(len(y), dtype=np.float32)
    mdir = ROOT / "runs" / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    imp = np.zeros(len(cols))
    for k in range(a.folds):
        tr, va = fold != k, fold == k
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=cols, free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(PARAMS, dtr, a.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        imp += m.feature_importance("gain")
        m.save_model(str(mdir / f"{name}_fold{k}.txt"), num_iteration=m.best_iteration)
        print(f"  fold {k}: best_iter {m.best_iteration}, logloss {m.best_score['valid_0']['binary_logloss']:.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    odir = ROOT / "cache" / "oof"
    odir.mkdir(parents=True, exist_ok=True)
    d.select("s1", "cand", "y").with_columns(p=pl.Series(oof)).write_parquet(odir / f"{name}.parquet")
    frac = int(a.s1_frac * 1000)
    s1_all = pl.read_parquet(ROOT / "cache/split" / f"{name.split('_')[0]}_source1.parquet", columns=["entity_id"]) \
               if (ROOT / "cache/split" / f"{name.split('_')[0]}_source1.parquet").exists() else None
    if s1_all is not None:  # the S1 universe this OOF covers (incl. S1s with no candidates)
        s1_all.filter((pl.col("entity_id").hash(seed=11) % 1000) < frac).rename({"entity_id": "s1"}) \
              .write_parquet(odir / f"{name}_s1.parquet")
    top = sorted(zip(imp, cols), reverse=True)[:25]
    print("top features by gain:", ", ".join(f"{c}" for _, c in top))
    print(f"wrote oof in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
