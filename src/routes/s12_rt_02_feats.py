"""research/17 step 2a: recompute the s12_14 route features for ALL train route pairs (routes_train.parquet, same
pair set so the pair-set-dependent features n_cands / cand_n_s1 / mix_rank_src are identical) and cache them with
y and the cross-fitted route p: s12_cache/rt_feats_train.parquet."""
import os
os.environ["POLARS_MAX_THREADS"] = "24"
import polars as _pl  # noqa: F401,E402  (initialise polars with 24 threads before s12_00_common)
from s12_lib import *  # noqa: F403,E402

fin = pl.read_parquet(SC / "oof_final.parquet", columns=["s1", "cand", "p"])
R = pl.read_parquet(SC / "routes_train.parquet")
s1r, s23r = features.records("full")
d = route_features(R.drop("y"), s1r, s23r, fin).join(R.select("s1", "cand", "y"), on=["s1", "cand"])
e = pl.read_parquet(SC / "e2e_final.parquet", columns=["s1", "cand", "p"]).rename({"p": "p_route"})
d = d.join(e, on=["s1", "cand"], how="left")
assert d["p_route"].null_count() == 0 and d.height == e.height, (d.height, e.height)
d.write_parquet(SC / "rt_feats_train.parquet")
print(f"rt_feats_train: {d.height:,} x {d.width} ({T()})")
