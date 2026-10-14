"""research/17 step 3: re-score the gated route pairs (route p >= 0.01) with the bge cross-encoder scores as extra
features, cross-fitted exactly like s12_14 (folds hash(s1, seed=17) % 5, inner 10% S1 split hash(seed=99) % 10 == 0
for early stopping, same LightGBM params), and re-measure the honest end-to-end gain with the s12_14 protocol
(production = v3 selection + stage 2 + v4 extras 0.99109; + current routes 0.99154).

Training rows = train route pairs with p_route >= 0.01 whose S1 is in H_A (bge scores from ckpt/rr_bge_local, which
was trained on H_B, so every score is out-of-sample). H_B rows cannot be scored honestly here (ckpt/rr_bge_B weights
are not on this server). The honest universe is inside H_A, so the evaluation is complete.
Pairs with p_route < 0.01 keep their current route p.

Variants (each: 5 fold models, OOF p for the gated universe pairs):
  stk      FE + p_route + bge_L + bge_S + bge_m     (stacked on the current OOF route p)
  stk0     FE + p_route                              (ablation: same rows, no bge)
  full     FE + bge_L + bge_S + bge_m, trained on ALL H_A route pairs (bge NaN where p_route < 0.01), p for gated
  full0    FE, trained on all H_A route pairs        (ablation)
Usage: s12_rt_04_train.py [--save stk]   (--save writes s12_cache/route_rr_model_fold{k}.txt + features json)"""
import os
os.environ["POLARS_MAX_THREADS"] = "24"
import polars as _pl  # noqa: F401,E402
import argparse  # noqa: E402
import json  # noqa: E402
from s12_00_common import *  # noqa: F403,E402
import lightgbm as lgb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--save", default="")
ap.add_argument("--variants", default="stk,stk0,full,full0")
a = ap.parse_args()
GATE = 0.01
RB = GX / "bundle_rt/results"
FE0 = json.load(open(SC / "route_model_features.json"))
BGE = ["bge_L", "bge_S", "bge_m"]


def bge():
    L = pl.read_parquet(RB / "rr_bge_local_rt_all.parquet").select("s1", "cand", bge_L="score")
    S = pl.read_parquet(RB / "rr_bge_local_rt_all_swap.parquet").select(s1="cand", cand="s1", bge_S="score")
    return L.join(S, on=["s1", "cand"]).with_columns(bge_m=(pl.col("bge_L") + pl.col("bge_S")) / 2)


BG = bge()
HA = pl.scan_parquet(C / "oof/full_v3.parquet").select("s1").unique().collect()
d = pl.read_parquet(SC / "rt_feats_train.parquet").join(HA, on="s1", how="semi").join(BG, on=["s1", "cand"], how="left")
d = d.with_columns(gated=pl.col("p_route") >= GATE)
assert d.filter("gated")["bge_L"].null_count() == 0
d = d.with_columns([pl.when("gated").then(pl.col(c)).otherwise(None).alias(c) for c in BGE])
print(f"H_A route pairs {d.height:,}; gated {d['gated'].sum():,} (true {d.filter('gated')['y'].sum():,}) ({T()})", flush=True)
P = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=24, seed=0)
VAR = {"stk": (FE0 + ["p_route"] + BGE, True), "stk0": (FE0 + ["p_route"], True),
       "full": (FE0 + BGE, False), "full0": (FE0, False)}


def X_of(t, FE):
    return t.select(pl.col(x).cast(pl.Float32) for x in FE).to_numpy()


def fit(name):
    FE, gated_only = VAR[name]
    t = d.filter("gated") if gated_only else d
    X, y = X_of(t, FE), t["y"].to_numpy().astype(np.float32)
    fold = (t["s1"].hash(seed=17) % 5).to_numpy()
    inner = ((t["s1"].hash(seed=99) % 10) == 0).to_numpy()
    p = np.zeros(len(y), dtype=np.float32)
    ms = []
    for f in range(5):
        tr, va, es = (fold != f) & ~inner, fold == f, (fold != f) & inner
        m = lgb.train(P, lgb.Dataset(X[tr], y[tr], feature_name=FE), 2000, valid_sets=[lgb.Dataset(X[es], y[es])],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        p[va] = m.predict(X[va], num_iteration=m.best_iteration, num_threads=24)
        ms.append(m)
    it = [m.best_iteration for m in ms]
    o = t.select("s1", "cand", "y", "gated", "p_route").with_columns(p=pl.Series(p)).filter("gated")
    return o, ms, it


uni = honest_universe()
gth = gt_all().join(uni, on="s1", how="semi")
F = pl.read_parquet(SC / "oof_final.parquet", columns=["s1", "cand", "p"]).join(uni, on="s1", how="semi")
V4 = pl.read_parquet(SC / "v4extra_uni.parquet", columns=["s1", "cand", "p"])
base = pl.concat([F, V4])
E = pl.read_parquet(SC / "e2e_final.parquet", columns=["s1", "cand", "y", "p"]).join(uni, on="s1", how="semi")


def per_s1(s):
    s11 = one_to_one(s.select("s1", "cand", "p"))
    return macro_f05(s11.filter(pl.col("p") >= THR).select("s1", "cand"), gth, uni).select("s1", "country", "f")


def routes_with(newp):
    """current route pairs (universe) with p replaced for the re-scored gated pairs"""
    return (E.join(newp.select("s1", "cand", pn="p"), on=["s1", "cand"], how="left")
              .with_columns(p=pl.coalesce("pn", "p")).select("s1", "cand", "p"))


f_base = per_s1(base)
f_cur = per_s1(pl.concat([base, E.select("s1", "cand", "p")]))
print(f"production {f_base['f'].mean():.5f}; + current routes {f_cur['f'].mean():.5f} "
      f"(delta {f_cur['f'].mean() - f_base['f'].mean():+.5f})", flush=True)
orc = E.filter(pl.col("p") >= GATE).with_columns(p=pl.col("y").cast(pl.Float32)).select("s1", "cand", "p")
f_or = per_s1(pl.concat([base, routes_with(orc)]))
print(f"oracle on gated pairs: {f_or['f'].mean():.5f} (delta {f_or['f'].mean() - f_base['f'].mean():+.5f})")


def report(tag, fn, ref=f_cur):
    dd = ref.join(fn.select("s1", f1="f"), on="s1").with_columns(dl=pl.col("f1") - pl.col("f"))
    dl = dd["dl"].to_numpy()
    rng = np.random.default_rng(0)
    se = np.std([dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(1000)])
    by = {c: round(v, 5) for c, v in fn.group_by("country").agg(pl.col("f").mean()).sort("country").iter_rows()}
    print(f"  {tag:10s} {fn['f'].mean():.5f}  delta vs production {fn['f'].mean() - f_base['f'].mean():+.5f}  "
          f"vs current routes {dl.mean():+.6f} (SE {se:.6f}; S1 +{(dl > 0).sum()} / -{(dl < 0).sum()})  {by}", flush=True)
    return fn['f'].mean()


DENS = (SC / "rt_feats_density.parquet").exists()  # research-only check (s12_rt_03); skipped in the package run
# density simulation inputs (universe only): baseline = s12_13 route p; gated pairs re-scored by the held-out fold model
if DENS:
    DN = pl.read_parquet(SC / "rt_feats_density.parquet").join(BG, on=["s1", "cand"], how="left")
    DN = DN.with_columns(gated=pl.col("p_route") >= GATE)
    assert DN.filter("gated")["bge_L"].null_count() == 0
    DN = DN.with_columns([pl.when("gated").then(pl.col(c)).otherwise(None).alias(c) for c in BGE])
    FD = pl.read_parquet(SC / "oof_final.parquet", columns=["s1", "cand", "p"]).join(uni, on="s1", how="semi")
    based = pl.concat([FD, V4])  # universe S1 are never dropped, so the universe production p is unchanged
    ED = DN.select("s1", "cand", "y", p="p_route")
    fd_cur = per_s1(pl.concat([based, ED.select("s1", "cand", "p")]))
    print(f"density: production {f_base['f'].mean():.5f}; + current routes {fd_cur['f'].mean():.5f} "
          f"(delta {fd_cur['f'].mean() - f_base['f'].mean():+.5f})", flush=True)

res = {}
for name in a.variants.split(","):
    o, ms, it = fit(name)
    ou = o.join(uni, on="s1", how="semi")
    hi = o.filter(pl.col("p") >= THR)
    print(f"[{name}] iters {it}; gated OOF AUC-ish: p>=.75 {hi.height:,} true {hi['y'].mean() * 100:.1f}% "
          f"(route p>=.75 {o.filter(pl.col('p_route') >= THR).height:,} true "
          f"{o.filter(pl.col('p_route') >= THR)['y'].mean() * 100:.1f}%)", flush=True)
    res[name] = report(name, per_s1(pl.concat([base, routes_with(ou)])))
    FE = VAR[name][0]
    if DENS:
        g = DN.filter("gated")
        X = X_of(g, FE)
        fold = (g["s1"].hash(seed=17) % 5).to_numpy()
        p = np.zeros(len(g), dtype=np.float32)
        for f in range(5):
            p[fold == f] = ms[f].predict(X[fold == f], num_iteration=ms[f].best_iteration, num_threads=24)
        nd = g.select("s1", "cand").with_columns(p=pl.Series(p))
        edn = (ED.join(nd.select("s1", "cand", pn="p"), on=["s1", "cand"], how="left")
                 .with_columns(p=pl.coalesce("pn", "p")).select("s1", "cand", "p"))
        report(name + "@dens", per_s1(pl.concat([based, edn])), ref=fd_cur)
    if name == a.save:
        for f, m in enumerate(ms):
            m.save_model(str(SC / f"route_rr_model_fold{f}.txt"), num_iteration=m.best_iteration)
        json.dump({"features": FE, "gate": GATE, "variant": name}, open(SC / "route_rr_model_features.json", "w"))
        o.write_parquet(SC / "e2e_rt_gated.parquet")
        print(f"saved route_rr_model_fold*.txt ({name})")
print(f"done ({T()})")
