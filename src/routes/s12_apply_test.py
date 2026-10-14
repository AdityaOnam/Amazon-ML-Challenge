"""Build the two shortlist routes (s12_lib: taddr + revmix) for TEST S1 in US/India (France untouched), score
them with the 5 train-only route models (fold average), and write the extra-pairs file for stage2.py apply:
  src/s12_cache/test_p_full_v3fr2w_routes.parquet = test_p_full_v3fr2w (v3 p + v4 extras) + route pairs
  (route pairs anti-joined against everything already there).
Production p used for claims/anchors = stage 2 (s2_v3b_ABE_CM, fold average) on test_stacked_v13, i.e. the
same p v13 thresholds for US/India. Use:
  cd src && python stage2.py apply --stacked research/gpu_exp/work/test_stacked_v13H.parquet --tag s2_v3b_ABE_CM \
     --groups CM --norm fr --extra src/s12_cache/test_p_full_v3fr2w_routes.parquet --no-extra-country France \
     --fr-swap --out-tag v14"""
import json

from s12_lib import *  # noqa: F403
import lightgbm as lgb
import stage2
from rapidfuzz import process as _proc

W = GX / "work"
C1 = pl.read_parquet(NORM / "test_source1.parquet", columns=["entity_id", "country"])
usin = C1.filter(pl.col("country").is_in(["US", "India"])).select(s1="entity_id")
fp = SC / "test_final_usin.parquet"
if not fp.exists():
    class _P:
        @staticmethod
        def cpdist(*a, **k):
            k["workers"] = 32
            return _proc.cpdist(*a, **k)
    stage2.process = _P
    stage2.GROUPS = "CM"
    stage2.FEATS = stage2.FEATS + stage2.C + stage2.MULTI
    st = pl.read_parquet(W / "test_stacked_v3b_ABE.parquet").join(usin, on="s1", how="semi")
    d = stage2.build(st, "test", "fr", "cache/feat/test.parquet")
    ms = [lgb.Booster(model_file=str(ROOT / f"runs/models/s2_v3b_ABE_CM_fold{k}.txt")) for k in range(5)]
    X = stage2.X_of(d)
    p2 = np.mean([m.predict(X, num_threads=32) for m in ms], axis=0).astype(np.float32)
    fin = (st.select("s1", "cand", "p").join(d.select("s1", "cand", p2=pl.Series(p2)), on=["s1", "cand"], how="left")
             .with_columns(p=pl.coalesce("p2", "p")).drop("p2"))
    fin.write_parquet(fp)
    print(f"test production p (US/India): {fin.height:,} pairs, p>=0.75 {(fin['p'] >= 0.75).sum():,} ({T()})", flush=True)
fin = pl.read_parquet(fp)
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
FE = json.load(open(SC / "route_model_features.json"))
X = X_of(d, FE)
p = np.mean([lgb.Booster(model_file=str(SC / f"route_model_fold{f}.txt")).predict(X, num_threads=32)
             for f in range(5)], axis=0).astype(np.float32)
rt = d.select("s1", "cand").with_columns(p=pl.Series(p))
rt.write_parquet(SC / "test_route_pairs.parquet")
cnt = rt.join(C1.rename({"entity_id": "s1"}), on="s1").group_by("country").agg(
    pairs=pl.len(), p75=(pl.col("p") >= 0.75).sum(), p90=(pl.col("p") >= 0.9).sum()).sort("country")
ns1 = C1.group_by("country").agg(n_s1=pl.len())
print(cnt.join(ns1, on="country").with_columns(p75_per_s1=pl.col("p75") / pl.col("n_s1")))
out = pl.concat([ex, rt.join(ex.select("s1", "cand"), on=["s1", "cand"], how="anti")])
out.write_parquet(SC / "test_p_full_v3fr2w_routes.parquet")
print(f"wrote test_p_full_v3fr2w_routes.parquet: {out.height:,} pairs ({ex.height:,} base + {out.height - ex.height:,} route) "
      f"({T()})")
