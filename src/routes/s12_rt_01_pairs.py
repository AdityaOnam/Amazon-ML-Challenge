"""research/17 step 1: build the pair tables for re-scoring the NEW shortlist-route pairs with the fine-tuned
bge cross-encoders, in a NEW bundle dir src/reranker/bundle_rt/.
  rt_train.parquet        train route pairs, route p >= 0.01, S1 in H_A (= S1 of cache/oof/full_v3.parquet)
  rt_test.parquet         test route pairs (US/India), route p >= 0.01 (routes file anti-joined with v3fr2w)
  rt_dens.parquet         density-simulation route pairs (s12_13), S1 in the honest universe, p >= 0.01, not
                          already in rt_train
  rt_all.parquet          union of the three (s1, cand)      -> scored in normal order
  rt_all_swap.parquet     same with s1 <-> cand swapped       -> scored in swapped order
  records.parquet         bundle/records.parquet restricted to the ids needed (same columns); records missing
                          from it are rebuilt exactly as export_bundle.py does (cache/norm_trn).
All of these are scored with ckpt/rr_bge_local (trained on H_B): honest for H_A S1 and for test.
H_B train S1 are NOT scored: ckpt/rr_bge_B has no weights on this server (only rr_meta.json)."""
from s12_00_common import *  # noqa: F403

B = GX / "bundle_rt"
B.mkdir(exist_ok=True)
GATE = 0.01
HA = pl.scan_parquet(C / "oof/full_v3.parquet").select("s1").unique().collect()
tr = pl.read_parquet(SC / "e2e_final.parquet", columns=["s1", "cand", "p"]).filter(pl.col("p") >= GATE)
print(f"train gated {tr.height:,}; in H_A {tr.join(HA, on='s1', how='semi').height:,}")
tr = tr.join(HA, on="s1", how="semi").select("s1", "cand")
ex = pl.scan_parquet(GX / "work/test_p_full_v3fr2w.parquet").select("s1", "cand")
te = (pl.scan_parquet(SC / "test_p_full_v3fr2w_routes.parquet").filter(pl.col("p") >= GATE)
        .join(ex, on=["s1", "cand"], how="anti").select("s1", "cand").collect())
uni = honest_universe()
# density-simulation pairs exist only when the (research-only) density check was run; empty otherwise
de = (pl.read_parquet(SC / "e2e_density.parquet", columns=["s1", "cand", "p"]).filter(pl.col("p") >= GATE)
        .join(uni, on="s1", how="semi").select("s1", "cand")) if (SC / "e2e_density.parquet").exists() \
    else tr.clear()
print(f"test gated {te.height:,}; density gated (universe) {de.height:,}, new vs train {de.join(tr, on=['s1', 'cand'], how='anti').height:,}")
tr.write_parquet(B / "rt_train.parquet")
te.write_parquet(B / "rt_test.parquet")
de.write_parquet(B / "rt_dens.parquet")
al = pl.concat([tr, te, de]).unique().sort("s1", "cand")
al.write_parquet(B / "rt_all.parquet")
al.select(s1="cand", cand="s1").write_parquet(B / "rt_all_swap.parquet")
need = pl.concat([al["s1"], al["cand"]]).unique().to_frame("entity_id")
rec = pl.scan_parquet(GX / "bundle/records.parquet").join(need.lazy(), on="entity_id", how="semi").collect()
miss = need.join(rec, on="entity_id", how="anti")
print(f"pairs {al.height:,}; ids {need.height:,}; in bundle records {rec.height:,}; missing {miss.height:,}")
if miss.height:  # rebuild exactly as export_bundle.py
    recs = []
    for split in ("train", "test"):
        for i in (1, 2, 3):
            r = pl.read_parquet(C / f"norm_trn/{split}_source{i}.parquet",
                                columns=["entity_id", "country", "business_name", "business_address",
                                         "name_clean", "addr_clean"])
            recs.append(r.with_columns(split=pl.lit(split), src=pl.lit(i, pl.UInt8)))
    add = pl.concat(recs).join(miss, on="entity_id", how="semi").select(rec.columns)
    print(f"rebuilt {add.height:,} of {miss.height:,} missing records")
    rec = pl.concat([rec, add])
assert need.join(rec, on="entity_id", how="anti").height == 0, "record text still missing"
assert rec["entity_id"].n_unique() == rec.height
rec.write_parquet(B / "records.parquet", compression="zstd")
print(f"bundle_rt done: {rec.height:,} records ({T()})")
