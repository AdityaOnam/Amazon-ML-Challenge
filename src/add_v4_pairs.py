"""Widen the candidate set with the v4-only pairs (mix10,addr10,name10,emb20 minus v3's mix10,addr5,name5,emb5).

v3 pairs keep their p; the extra pairs get the v4 LightGBM p (research/10_residual_errors_v5.md fix #2: adding
the pairs gains +0.0003 honest, while replacing the v3 model with v4 lost). French pairs can come from a
France-renormalised v4 feature table (--fr-feat).
Usage: python add_v4_pairs.py --p-in research/gpu_exp/work/test_p_full_v3fr2.parquet \
           --fr-feat cache/feat/test_v4_fr.parquet --out research/gpu_exp/work/test_p_full_v3fr2w.parquet
"""
import argparse

import lightgbm as lgb
import numpy as np
import polars as pl

from io_utils import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-in", required=True)
    ap.add_argument("--feat", default="cache/feat/test_v4.parquet")
    ap.add_argument("--fr-feat", default=None)
    ap.add_argument("--models", default="full_v4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ms = [lgb.Booster(model_file=str(f)) for f in sorted((ROOT / "runs/models").glob(f"{a.models}_fold*.txt"))]
    cols = ms[0].feature_name()
    tp = pl.read_parquet(ROOT / a.p_in)
    fr = (pl.read_parquet(ROOT / "cache/norm_trn/test_source1.parquet", columns=["entity_id", "country"])
            .filter(pl.col("country") == "France").select(s1="entity_id"))
    parts = []
    srcs = [(a.feat, "anti" if a.fr_feat else None)] + ([(a.fr_feat, "semi")] if a.fr_feat else [])
    for path, fr_mode in srcs:
        q = pl.scan_parquet(ROOT / path).select(["s1", "cand"] + cols).join(tp.lazy().select("s1", "cand"),
                                                                            on=["s1", "cand"], how="anti")
        if fr_mode:
            q = q.join(fr.lazy(), on="s1", how=fr_mode)  # base table: non-French only; fr table: French only
        d = q.collect()
        X = d.select(pl.col(c).cast(pl.Float32) for c in cols).to_numpy()
        p = np.mean([m.predict(X, num_threads=48) for m in ms], axis=0).astype(np.float32)
        parts.append(d.select("s1", "cand").with_columns(p=pl.Series(p)))
        print(f"{path}: {d.height:,} extra pairs, p>=0.75: {(p >= 0.75).sum():,}", flush=True)
        del d, X
    out = pl.concat([tp] + parts)
    out.write_parquet(ROOT / a.out)
    print(f"wrote {a.out}: {out.height:,} pairs ({tp.height:,} base)")


if __name__ == "__main__":
    main()
