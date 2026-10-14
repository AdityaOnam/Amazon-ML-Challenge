"""Production baseline for the honest universe = v3-selection pairs (final stage-2 p) + v4-only extra pairs
(add_v4_pairs.py; in the upload for US/India). v4 p: OOF where the S1 is in the v4 training sample, else the mean
of the 3 full_v4 fold models (never trained on that S1). Writes s12_cache/v4extra_uni.parquet (s1, cand, y, p)."""
from s12_00_common import *  # noqa: F403
import lightgbm as lgb

uni = honest_universe()
fin = pl.read_parquet(SC / "oof_final.parquet", columns=["s1", "cand"]).join(uni, on="s1", how="semi")
ms = [lgb.Booster(model_file=str(ROOT / f"runs/models/full_v4_fold{k}.txt")) for k in range(3)]
cols = ms[0].feature_name()
d = (pl.scan_parquet(C / "feat/full_v4.parquet").select(["s1", "cand", "y"] + cols)
       .join(uni.lazy().select("s1"), on="s1", how="semi").join(fin.lazy(), on=["s1", "cand"], how="anti").collect())
X = d.select(pl.col(c).cast(pl.Float32) for c in cols).to_numpy()
p = np.mean([m.predict(X, num_threads=32) for m in ms], axis=0).astype(np.float32)
d = d.select("s1", "cand", "y").with_columns(p_avg=pl.Series(p))
oof = pl.read_parquet(C / "oof/full_v4.parquet", columns=["s1", "cand", "p"]).join(uni, on="s1", how="semi")
d = d.join(oof.rename({"p": "p_oof"}), on=["s1", "cand"], how="left").with_columns(p=pl.coalesce("p_oof", "p_avg"))
print(f"v4-only pairs {d.height:,} ({d.height / uni.height:.1f}/S1), true {d['y'].sum():,}, from OOF {d['p_oof'].is_not_null().sum():,}")
d.select("s1", "cand", "y", "p").write_parquet(SC / "v4extra_uni.parquet")
gth = gt_all().join(uni, on="s1", how="semi")
F = pl.read_parquet(SC / "oof_final.parquet", columns=["s1", "cand", "p"]).join(uni, on="s1", how="semi")
print("v3 selection only        ", score(F, gth, uni))
print("+ v4 extras (production) ", score(pl.concat([F, d.select("s1", "cand", "p")]), gth, uni))
print(T())
