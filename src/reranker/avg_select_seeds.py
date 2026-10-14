"""Average select-model seeds per half: sel_prob mean over sel_<a> and sel_<b> files -> sel_<out>_* (same schema).
python avg_select_seeds.py  (seed0 = sel_A/sel_B, seed1 = sel_A1/sel_B1 -> sel_A2/sel_B2)"""
import polars as pl
from pathlib import Path
R = Path(__file__).resolve().parent / "results"
for m, h in (("A", "oof_B"), ("A", "test"), ("B", "oof_A"), ("B", "test")):
    x = pl.read_parquet(R / f"sel_sel_{m}_{h}.parquet")
    y = pl.read_parquet(R / f"sel_sel_{m}1_{h}.parquet").select("s1", "cand", p1="sel_prob")
    z = x.join(y, on=["s1", "cand"], how="left")
    assert z["p1"].null_count() == 0 and z.height == y.height, (m, h)
    z = z.with_columns(sel_prob=((pl.col("sel_prob") + pl.col("p1")) / 2).cast(pl.Float32)).drop("p1").with_columns(score=(pl.col("sel_prob").clip(1e-6, 1 - 1e-6) / (1 - pl.col("sel_prob").clip(1e-6, 1 - 1e-6))).log().cast(pl.Float32))
    z.write_parquet(R / f"sel_sel_{m}2_{h}.parquet")
    print(m, h, z.height)
