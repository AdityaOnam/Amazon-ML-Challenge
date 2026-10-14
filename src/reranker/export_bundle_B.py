"""Mirror bundle for the second bge (experiment A): train on H_A, score H_B, so every S1 gets an honest bge score.

H_A = S1s of the full_v3 OOF (the half the first bge never trained on); H_B = the other S1s of full_v3b.
  bundle_B/rr_hb.parquet       s1, cand          H_B pairs with full_v3b OOF p in [0.003, 0.997]   -> scored
  bundle_B/rr_hb_swap.parquet  s1, cand          the same pairs in swapped order (candidate first)  -> scored
  bundle_B/rr_train.parquet    s1, cand, y, p, kind   H_A: every pair with p in [0.02, 0.98] ("hard") plus
                               sampled easy negatives (0.5 x hard, p < 0.02) and easy positives (0.15 x hard,
                               p > 0.98); candidates that appear in rr_hb are excluded (never trained on a
                               record the model later scores)
Usage: python export_bundle_B.py   (paths relative to LLM_based_work; writes research/gpu_exp/bundle_B/)
"""
import argparse
import shutil
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "bundle_B"))
    out = Path(ap.parse_args().out)
    out.mkdir(exist_ok=True)
    if not (out / "records.parquet").exists():  # record texts for scoring/training (same as the first bundle)
        shutil.copy(HERE / "bundle/records.parquet", out / "records.parquet")
    ha = pl.read_parquet(R / "cache/oof/full_v3.parquet", columns=["s1"]).unique()
    ob = pl.read_parquet(R / "cache/oof/full_v3b.parquet", columns=["s1", "cand", "y", "p"])
    hb = ob.join(ha, on="s1", how="anti").filter(pl.col("p").is_between(0.003, 0.997)).select("s1", "cand")
    hb.write_parquet(out / "rr_hb.parquet")
    hb.select(s1="cand", cand="s1").write_parquet(out / "rr_hb_swap.parquet")
    tr = ob.join(ha, on="s1", how="semi").join(hb.select("cand").unique(), on="cand", how="anti")
    hard = tr.filter(pl.col("p").is_between(0.02, 0.98)).with_columns(kind=pl.lit("hard"))
    eneg = tr.filter((pl.col("p") < 0.02) & ~pl.col("y"))
    epos = tr.filter((pl.col("p") > 0.98) & pl.col("y"))
    eneg = eneg.sample(min(int(0.5 * hard.height), eneg.height), seed=1).with_columns(kind=pl.lit("easy_neg"))
    epos = epos.sample(min(int(0.15 * hard.height), epos.height), seed=2).with_columns(kind=pl.lit("easy_pos"))
    rr = pl.concat([hard, eneg, epos]).sample(fraction=1.0, shuffle=True, seed=3)
    rr.write_parquet(out / "rr_train.parquet")
    print(f"rr_hb {hb.height:,}  rr_train {rr.height:,} (hard {hard.height:,}, easy_neg {eneg.height:,}, "
          f"easy_pos {epos.height:,})")


if __name__ == "__main__":
    main()
