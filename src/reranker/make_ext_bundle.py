"""Band extension for the local bge (experiment E): pairs with full_v3b p in [0.001, 0.999] that no earlier
reranker table (rr_stack, rr_test, rr_stack_wide, rr_test_wide) holds.
  OOF side: S1s of the full_v3 OOF (H_A, the half the local bge never trained on).
  Test side: US / India S1s only (France keeps the inner/wide band of the French-renormalised base).
Written as n shards bundle/rr_e_<k>.parquet (s1, cand) and bundle/rr_esw_<k>.parquet (swapped order), split by
S1 hash like make_score_shards.py, so several GPUs can score them in parallel.
Usage: python make_ext_bundle.py [--n 4]
"""
import argparse
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--band", type=float, nargs=2, default=[0.001, 0.999])
    ap.add_argument("--out", default=str(HERE / "bundle"))
    a = ap.parse_args()
    B = HERE / "bundle"
    old = pl.concat([pl.read_parquet(B / f"{t}.parquet").select("s1", "cand")
                     for t in ("rr_stack", "rr_test", "rr_stack_wide", "rr_test_wide")]).unique()
    ha = pl.read_parquet(R / "cache/oof/full_v3.parquet", columns=["s1"]).unique()
    o = (pl.read_parquet(R / "cache/oof/full_v3b.parquet", columns=["s1", "cand", "p"]).join(ha, on="s1", how="semi")
           .filter(pl.col("p").is_between(*a.band)).select("s1", "cand"))
    fr = (pl.read_parquet(R / "cache/norm_trn/test_source1.parquet", columns=["entity_id", "country"])
            .filter(pl.col("country") == "France").select(s1="entity_id"))
    t = (pl.read_parquet(HERE / "work/test_p_full_v3b.parquet").filter(pl.col("p").is_between(*a.band))
           .select("s1", "cand").join(fr, on="s1", how="anti"))
    e = pl.concat([o, t]).unique(maintain_order=True).join(old, on=["s1", "cand"], how="anti")
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    k = e.with_columns(k=(pl.col("s1").hash(seed=0) % a.n))
    for i in range(a.n):
        s = k.filter(pl.col("k") == i).select("s1", "cand")
        s.write_parquet(out / f"rr_e_{i}.parquet")
        s.select(s1="cand", cand="s1").write_parquet(out / f"rr_esw_{i}.parquet")
    print(f"band extension: {o.join(old, on=['s1', 'cand'], how='anti').height:,} OOF + "
          f"{t.join(old, on=['s1', 'cand'], how='anti').height:,} test pairs in {a.n} shards")


if __name__ == "__main__":
    main()
