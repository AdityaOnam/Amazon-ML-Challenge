"""Reproduce the honest 0.99079 baseline (stacker v3b_ABE + stage-2 s2_v3b_ABE_CM) as a saved OOF table.
Stage-2 fold models are applied to the fold they were NOT trained on (fold = hash(s1, seed=17) % 5), so the
result equals stage2.cmd_train's cross-fitted p2. Writes s12_cache/oof_final.parquet (s1, cand, y, p, p_st, p_lgb)."""
from s12_00_common import *  # noqa: F403
import lightgbm as lgb
import stage2
from rapidfuzz import process as _proc


class _P:  # cap rapidfuzz workers at 32 (stage2.build hard-codes 48)
    @staticmethod
    def cpdist(*a, **k):
        k["workers"] = 32
        return _proc.cpdist(*a, **k)


stage2.process = _P
stage2.GROUPS = "CM"
stage2.FEATS = stage2.FEATS + stage2.C + stage2.MULTI
st = pl.read_parquet(STACKED)
print(f"stacked {st.height:,} ({T()})", flush=True)
d = stage2.build(st, "train", "trn", str(FEAT.relative_to(ROOT)))
X = stage2.X_of(d)
fold = (d["s1"].hash(seed=17) % 5).to_numpy()
p2 = np.zeros(len(d), dtype=np.float32)
for k in range(5):
    m = lgb.Booster(model_file=str(ROOT / f"runs/models/s2_v3b_ABE_CM_fold{k}.txt"))
    assert m.feature_name() == stage2.FEATS, "feature order mismatch"
    va = fold == k
    p2[va] = m.predict(X[va], num_threads=32)
    print(f"  fold {k} predicted ({T()})", flush=True)
out = (st.select("s1", "cand", "y", p_st="p", p_lgb="p_lgb")
         .join(d.select("s1", "cand", p2=pl.Series(p2)), on=["s1", "cand"], how="left")
         .with_columns(p=pl.coalesce("p2", "p_st")).drop("p2"))
out.write_parquet(SC / "oof_final.parquet")
uni, gt = honest_universe(), gt_all()
gth = gt.join(uni, on="s1", how="semi")
print("stacked", score(out, gth, uni, col="p_st"))
print("stage2 ", score(out, gth, uni))
print(f"done ({T()})")
