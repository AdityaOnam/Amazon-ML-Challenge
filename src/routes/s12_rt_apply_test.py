"""research/17: test extra-pairs file with the gated route pairs re-scored by the bge-augmented route model.
Rebuilds the US/India test routes + route features exactly as s12_apply_test.py (same cached production p
s12_cache/test_final_usin.parquet, same exclusions), checks that the current route p is reproduced
(s12_cache/test_route_pairs.parquet), then for pairs with route p >= 0.01 joins the bge scores of ckpt/rr_bge_local
(src/reranker/bundle_rt/results, normal + swapped order; trained on train H_B only, test is never trained on)
and predicts with the mean of the 5 train-only models s12_cache/route_rr_model_fold{k}.txt.
Writes src/s12_cache/test_p_full_v3fr2w_routes_rr.parquet = test_p_full_v3fr2w (unchanged, incl. France)
+ route pairs (US/India) with the new p. Drop-in replacement for --extra in stage2.py apply."""
import os
os.environ["POLARS_MAX_THREADS"] = "24"
import polars as _pl  # noqa: F401,E402
import json  # noqa: E402
from s12_lib import *  # noqa: F403,E402
import lightgbm as lgb  # noqa: E402

W = GX / "work"
RB = GX / "bundle_rt/results"
C1 = pl.read_parquet(NORM / "test_source1.parquet", columns=["entity_id", "country"])
usin = C1.filter(pl.col("country").is_in(["US", "India"])).select(s1="entity_id")
fin = pl.read_parquet(SC / "test_final_usin.parquet")
ex = pl.read_parquet(W / "test_p_full_v3fr2w.parquet")
pool = pl.concat([pl.scan_parquet(C / "cand/test_cands_k10.parquet").select("s1", "cand"),
                  pl.scan_parquet(C / "cand/test_emb_Qwen3-Embedding-0.6B_k20.parquet").select("s1", "cand")]
                 ).join(usin.lazy(), on="s1", how="semi").collect()
exclude = pl.concat([pool, ex.join(usin, on="s1", how="semi").select("s1", "cand")]).unique()
S1 = pl.read_parquet(NORM / "test_source1.parquet", columns=RCOLS).filter(pl.col("country").is_in(["US", "India"]))
R23 = pl.concat([pl.read_parquet(NORM / f"test_source{i}.parquet", columns=RCOLS) for i in (2, 3)]) \
        .filter(pl.col("country").is_in(["US", "India"]))
R = build_routes(S1, R23, fin, exclude, log=lambda s: print(s, flush=True))
s1r, s23r = features.records("test")
s1r, s23r = s1r.filter(pl.col("country").is_in(["US", "India"])), s23r.filter(pl.col("country").is_in(["US", "India"]))
d = route_features(R, s1r, s23r, fin)
FE0 = json.load(open(SC / "route_model_features.json"))
X = X_of(d, FE0)
p0 = np.mean([lgb.Booster(model_file=str(SC / f"route_model_fold{f}.txt")).predict(X, num_threads=24)
              for f in range(5)], axis=0).astype(np.float32)
d = d.with_columns(p_route=pl.Series(p0))
old = pl.read_parquet(SC / "test_route_pairs.parquet")
chk = d.select("s1", "cand", "p_route").join(old, on=["s1", "cand"], how="full", coalesce=True)
dev = (chk["p_route"] - chk["p"]).abs().max()
print(f"route pairs {d.height:,} vs s12_apply_test {old.height:,}; nulls {chk['p_route'].null_count()}/{chk['p'].null_count()}; "
      f"max |p diff| {dev} ({T()})", flush=True)
assert d.height == old.height and chk["p"].null_count() == 0 and dev < 1e-5
meta = json.load(open(SC / "route_rr_model_features.json"))
GATE, FE = meta["gate"], meta["features"]
L = pl.read_parquet(RB / "rr_bge_local_rt_all.parquet").select("s1", "cand", bge_L="score")
Sw = pl.read_parquet(RB / "rr_bge_local_rt_all_swap.parquet").select(s1="cand", cand="s1", bge_S="score")
BG = L.join(Sw, on=["s1", "cand"]).with_columns(bge_m=(pl.col("bge_L") + pl.col("bge_S")) / 2)
g = d.filter(pl.col("p_route") >= GATE).join(BG, on=["s1", "cand"], how="left")
assert g["bge_L"].null_count() == 0 and g["bge_S"].null_count() == 0, "gated test pair without bge score"
Xg = g.select(pl.col(x).cast(pl.Float32) for x in FE).to_numpy()
pn = np.mean([lgb.Booster(model_file=str(SC / f"route_rr_model_fold{f}.txt")).predict(Xg, num_threads=24)
              for f in range(5)], axis=0).astype(np.float32)
new = g.select("s1", "cand").with_columns(pn=pl.Series(pn))
rt = (d.select("s1", "cand", "p_route").join(new, on=["s1", "cand"], how="left")
        .select("s1", "cand", p=pl.coalesce("pn", "p_route").cast(ex.schema["p"])))
cnt = (rt.join(old.rename({"p": "p_old"}), on=["s1", "cand"]).join(C1.rename({"entity_id": "s1"}), on="s1")
         .group_by("country").agg(pairs=pl.len(), gated=(pl.col("p_old") >= GATE).sum(),
                                  p75_old=(pl.col("p_old") >= 0.75).sum(), p75_new=(pl.col("p") >= 0.75).sum(),
                                  p90_new=(pl.col("p") >= 0.9).sum()).sort("country"))
ns1 = C1.group_by("country").agg(n_s1=pl.len())
print(cnt.join(ns1, on="country").with_columns(p75_old_per_s1=pl.col("p75_old") / pl.col("n_s1"),
                                               p75_new_per_s1=pl.col("p75_new") / pl.col("n_s1")))
out = pl.concat([ex, rt.join(ex.select("s1", "cand"), on=["s1", "cand"], how="anti")])
out.write_parquet(SC / "test_p_full_v3fr2w_routes_rr.parquet")
ref = pl.scan_parquet(SC / "test_p_full_v3fr2w_routes.parquet").select(pl.len()).collect().item()
print(f"wrote test_p_full_v3fr2w_routes_rr.parquet: {out.height:,} pairs (routes file {ref:,}) ({T()})")
