"""Cross-fitted bge scores (experiment A): every S1 gets a score from a bge that never trained on its half.
  H_A S1s + test: the local bge (trained on H_B)            results/rr_bge_local_allE.parquet (+ _swap_allE)
  H_B S1s:        the second bge (trained on H_A, bundle_B)  results/rr_bge_B_rr_hb.parquet (+ _rr_hb_swap)
Swapped-order files from score_reranker.py hold the candidate in `s1`; they are flipped back here.
Writes results/rr_L_ABE.parquet (normal order) and results/rr_S_ABE.parquet (swapped order).
Usage: python crossfit_scores.py
"""
import argparse
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(RES))
    O = Path(ap.parse_args().out_dir)
    hb = pl.read_parquet(RES / "rr_bge_B_rr_hb.parquet").select("s1", "cand", "score")
    hbs = hb.select("s1").unique()
    loc = pl.read_parquet(RES / "rr_bge_local_allE.parquet").select("s1", "cand", "score")
    pl.concat([loc.join(hbs, on="s1", how="anti"), hb]).write_parquet(O / "rr_L_ABE.parquet")
    hsw = pl.read_parquet(RES / "rr_bge_B_rr_hb_swap.parquet").select(s1="cand", cand="s1", score="score")
    lsw = pl.read_parquet(RES / "rr_bge_local_swap_allE.parquet").select("s1", "cand", "score")
    pl.concat([lsw.join(hbs, on="s1", how="anti"), hsw]).write_parquet(O / "rr_S_ABE.parquet")
    print(f"rr_L_ABE / rr_S_ABE: {loc.join(hbs, on='s1', how='anti').height:,} local + {hb.height:,} bge_B pairs")


if __name__ == "__main__":
    main()
