"""Join reranker score files (shards, or inner-band + wide-band parts) into one s1, cand, score table.
--swapped: the inputs were scored candidate-first; flip the ids back so the table is keyed (s1, cand).
Usage: python concat_scores.py --out results/rr_bge_local_all.parquet results/rr_bge_local_rr_all_{0..3}.parquet
"""
import argparse

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--swapped", action="store_true")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    d = pl.concat([pl.read_parquet(f, columns=["s1", "cand", "score"]) for f in a.files])
    if a.swapped:
        d = d.select(s1="cand", cand="s1", score="score")
    d = d.unique(["s1", "cand"], keep="first")
    d.write_parquet(a.out)
    print(f"wrote {a.out}: {d.height:,} pairs")


if __name__ == "__main__":
    main()
