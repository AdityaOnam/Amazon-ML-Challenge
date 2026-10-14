"""Country-wise combination of two stacked test tables: France rows from one stack, all other countries from
another (the final submission keeps France on the inner-band bge+Qwen stack of the French-renormalised base,
which the leaderboard confirmed, and uses the wide-band multi-reranker stack elsewhere).
Usage: python fr_hybrid.py --main work/test_stacked_v3b_lsbqw.parquet --france work/test_stacked_bq_fr2.parquet \
           --out work/test_stacked_v12.parquet      (paths relative to research/gpu_exp)
"""
import argparse

import polars as pl

from io_utils import ROOT

W = ROOT / "research/gpu_exp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--france", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fr = (pl.read_parquet(ROOT / "cache/norm_trn/test_source1.parquet", columns=["entity_id", "country"])
            .filter(pl.col("country") == "France").select(s1="entity_id"))
    m = pl.read_parquet(W / a.main).join(fr, on="s1", how="anti")
    f = pl.read_parquet(W / a.france).join(fr, on="s1", how="semi")
    # keep every column of the main table; French rows get nulls where their stack has no such column
    f = f.with_columns(pl.lit(None, m.schema[c]).alias(c) for c in m.columns if c not in f.columns)
    pl.concat([m, f.select(m.columns)]).write_parquet(W / a.out)
    print(f"wrote {a.out}: {m.height:,} non-French + {f.height:,} French pairs")


if __name__ == "__main__":
    main()
