"""Train the route-pair LightGBM (5 folds by hash(s1, seed=17) % 5, inner 10% S1 early stopping) on the train
route pairs from s12_lib.build_routes, save the fold models to s12_cache/route_model_fold{k}.txt (+ feature list),
and re-measure the honest end-to-end gain with the cross-fitted p. Also saves s12_cache/routes_train.parquet."""
import json

from s12_lib import *  # noqa: F403
import lightgbm as lgb

fin = pl.read_parquet(SC / "oof_final.parquet", columns=["s1", "cand", "p"])
S1 = pl.read_parquet(NORM / "train_source1.parquet", columns=RCOLS)
R23 = pl.concat([pl.read_parquet(NORM / f"train_source{i}.parquet", columns=RCOLS) for i in (2, 3)])
R = build_routes(S1, R23, fin, pool_pairs().collect(), log=lambda s: print(s, flush=True))
gt = gt_all().with_columns(y=pl.lit(True))
R = R.join(gt, on=["s1", "cand"], how="left").with_columns(pl.col("y").fill_null(False))
R.write_parquet(SC / "routes_train.parquet")
print(f"route pairs {R.height:,} ({R.height / S1.height:.2f}/S1), true {R['y'].sum():,} ({T()})", flush=True)
s1r, s23r = features.records("full")
d = route_features(R.drop("y"), s1r, s23r, fin).join(R.select("s1", "cand", "y"), on=["s1", "cand"])
FE = [x for x in d.columns if x not in ("s1", "cand", "y")]
json.dump(FE, open(SC / "route_model_features.json", "w"))
X, y = X_of(d, FE), d["y"].to_numpy().astype(np.float32)
fold = (d["s1"].hash(seed=17) % 5).to_numpy()
inner = ((d["s1"].hash(seed=99) % 10) == 0).to_numpy()
P = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=32, seed=0)
p = np.zeros(len(y), dtype=np.float32)
for f in range(5):
    tr, va, es = (fold != f) & ~inner, fold == f, (fold != f) & inner
    m = lgb.train(P, lgb.Dataset(X[tr], y[tr], feature_name=FE), 2000, valid_sets=[lgb.Dataset(X[es], y[es])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    p[va] = m.predict(X[va], num_iteration=m.best_iteration, num_threads=32)
    m.save_model(str(SC / f"route_model_fold{f}.txt"), num_iteration=m.best_iteration)
    print(f"  fold {f}: {m.best_iteration} iters ({T()})", flush=True)
d = d.with_columns(p=pl.Series(p))
d.select("s1", "cand", "y", "p", "f_taddr_word", "f_revmix").write_parquet(SC / "e2e_final.parquet")
print(f"OOF pairs p>=0.75: {(p >= .75).sum():,} (true {y[p >= .75].mean() * 100:.1f}%), per S1 {(p >= .75).sum() / S1.height:.5f}")
uni = honest_universe()
gth = gt_all().join(uni, on="s1", how="semi")
F = fin.join(uni, on="s1", how="semi")
V4 = pl.read_parquet(SC / "v4extra_uni.parquet", columns=["s1", "cand", "p"])
base = pl.concat([F, V4])
du = d.join(uni, on="s1", how="semi").select("s1", "cand", "p")
f0, f1 = score(base, gth, uni), score(pl.concat([base, du]), gth, uni)
print(f"production {f0}\n+ routes   {f1}  delta {f1[0] - f0[0]:+.5f}\nv3-only base {score(F, gth, uni)} "
      f"+ routes (no v4 extras) {score(pl.concat([F, du]), gth, uni)}\ndone ({T()})")
