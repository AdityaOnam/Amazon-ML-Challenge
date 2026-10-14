"""R10, HOME server: turn the four returned select-score files into two --rr style inputs for merge_feature.py.

  results/rr_sel_<tag>_oof.parquet   s1, cand, score   honest cross-fit: H_A options from model_B (sel_B_oof_A,
                                                       ha=True rows), H_B options from model_A (sel_A_oof_B, ha=False)
  results/rr_sel_<tag>_test.parquet  s1, cand, score   mean of both models' sel_prob -> logit
score = logit(sel_prob) (float32), the same "one logit per pair" contract as the reranker files. Pairs outside the
contests have no row, so merge_feature's left join leaves them null (NaN for LightGBM).
python assemble_select.py --tag ABS [--results results]
"""
import argparse
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
EPS = 1e-6


def lg(c):
    x = pl.col(c).clip(EPS, 1 - EPS)
    return (x / (1 - x)).log().cast(pl.Float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ABS")
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--in-a", default="sel_A", help="tag used by score_select.py for model_A")
    ap.add_argument("--in-b", default="sel_B", help="tag used by score_select.py for model_B")
    ap.add_argument("--test-mode", choices=["mean", "A", "B"], default="mean",
                    help="test score: mean of both models (default) or one model only (OOF rows are single-model)")
    a = ap.parse_args()
    RES = Path(a.results)
    ra = pl.read_parquet(RES / f"sel_{a.in_b}_oof_A.parquet").filter(pl.col("ha"))    # model_B on H_A
    rb = pl.read_parquet(RES / f"sel_{a.in_a}_oof_B.parquet").filter(~pl.col("ha"))   # model_A on H_B
    oof = pl.concat([ra, rb]).select("s1", "cand", "score")
    assert oof.select("s1", "cand").is_duplicated().sum() == 0
    oof.write_parquet(RES / f"rr_sel_{a.tag}_oof.parquet")
    ta = pl.read_parquet(RES / f"sel_{a.in_a}_test.parquet").select("s1", "cand", pa="sel_prob")
    tb = pl.read_parquet(RES / f"sel_{a.in_b}_test.parquet").select("s1", "cand", pb="sel_prob")
    t = ta.join(tb, on=["s1", "cand"], how="full", coalesce=True)
    sel = {"mean": pl.mean_horizontal("pa", "pb"), "A": pl.col("pa"), "B": pl.col("pb")}[a.test_mode]
    t = t.with_columns(sel=sel).select("s1", "cand", score=lg("sel"))
    t.write_parquet(RES / f"rr_sel_{a.tag}_test.parquet")
    c = ta.join(tb, on=["s1", "cand"]).select(pl.corr("pa", "pb")).item()
    print(f"oof {oof.height:,} (H_A {ra.height:,} / H_B {rb.height:,}); test {t.height:,} "
          f"(model A/B sel_prob corr {c:.3f})")


if __name__ == "__main__":
    main()
