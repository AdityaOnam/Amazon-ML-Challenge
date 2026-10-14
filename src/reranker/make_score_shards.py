"""Pair tables for reranker scoring on several GPUs: the union of the given bundle tables, split into N shards
by S1 hash -> bundle/<prefix>_<k>.parquet; with --swap also the swapped order (candidate first) as
bundle/<prefix>sw_<k>.parquet. Afterwards `concat_scores.py` joins the shard results back.
Usage: python make_score_shards.py --tables rr_stack rr_test rr_stack_wide rr_test_wide --prefix rr_all --n 4 --swap
"""
import argparse
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--swap", action="store_true")
    a = ap.parse_args()
    B = HERE / "bundle"
    u = pl.concat([pl.read_parquet(B / f"{t}.parquet", columns=["s1", "cand"]) for t in a.tables]).unique(maintain_order=True)
    u = u.with_columns(sh=pl.col("s1").hash(seed=5) % a.n)
    for k in range(a.n):
        x = u.filter(pl.col("sh") == k).drop("sh")
        x.write_parquet(B / f"{a.prefix}_{k}.parquet")
        if a.swap:
            x.select(s1="cand", cand="s1").write_parquet(B / f"{a.prefix}sw_{k}.parquet")
    print(f"{u.height:,} pairs in {a.n} shards")


if __name__ == "__main__":
    main()
