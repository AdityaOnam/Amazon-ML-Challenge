"""Back on the HOME server: turn returned reranker scores into a better p for the uncertain pairs, measure it, apply it.

eval : stack on H_A band pairs.  Features = logit(p_lgb), rr score, within-S1 / within-candidate rr context.
       Small LightGBM, 5-fold cross-fit by S1 -> p_new (OOF) for band pairs; p outside the band is untouched.
       Scores macro F0.5 on the honest universe (H_A and val fold, same as src/compare_oof.py) before/after.
       python merge_feature.py eval --rr results/rr_qwen06_full_rr_stack.parquet --tag qwen06_full
apply: same features on test band pairs, average of the fold stackers, override p in work/test_p_<models>.parquet,
       one-to-one + threshold, write research/gpu_exp/submission_<tag>/{matching_results,candidate_pairs}.tsv
       and run the official validator. Never touches output/.
       python merge_feature.py apply --rr results/rr_qwen06_full_rr_test.parquet --tag qwen06_full --thr 0.75
"""
import argparse
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import Timer, write_json  # noqa: E402

BASE_FEATS = ["rr", "rr_minus_lp", "rr_rank_s1", "rr_max_s1", "rr_gap_s1", "rr_max_cand", "rr_gap_cand"]
FEATS = ["lp", "n_band_s1", "p_rank_s1", "n_band_cand"] + BASE_FEATS  # updated by stack_features for >1 reranker
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=32)


def load_rr(paths) -> pl.DataFrame:
    """One or more reranker score files -> s1, cand, score[, score_2, ...]. The FIRST file defines the pair set
    (left join): later rerankers may cover only part of it (e.g. the inner band); their gaps stay null (NaN for
    LightGBM)."""
    out = None
    for i, pth in enumerate(paths):
        r = pl.read_parquet(pth).select("s1", "cand", pl.col("score").alias("score" if i == 0 else f"score_{i + 1}"))
        out = r if out is None else out.join(r, on=["s1", "cand"], how="left")
    return out


def _rr_block(d: pl.DataFrame, src: str, sfx: str) -> pl.DataFrame:
    rr = f"rr{sfx}"
    d = d.with_columns(pl.col(src).cast(pl.Float32).alias(rr))
    d = d.with_columns(
        (pl.col(rr) - pl.col("lp")).alias(f"rr_minus_lp{sfx}"),
        pl.col(rr).rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias(f"rr_rank_s1{sfx}"),
        pl.col(rr).max().over("s1").alias(f"rr_max_s1{sfx}"),
        pl.col(rr).max().over("cand").alias(f"rr_max_cand{sfx}"))
    # gap to the best OTHER pair of the same S1 / same candidate (one-to-one competition)
    second_s1 = pl.col(rr).sort(descending=True).slice(1, 1).first().over("s1")
    second_c = pl.col(rr).sort(descending=True).slice(1, 1).first().over("cand")
    return d.with_columns(
        pl.when(pl.col(rr) >= pl.col(f"rr_max_s1{sfx}")).then(pl.col(rr) - second_s1)
          .otherwise(pl.col(rr) - pl.col(f"rr_max_s1{sfx}")).alias(f"rr_gap_s1{sfx}"),
        pl.when(pl.col(rr) >= pl.col(f"rr_max_cand{sfx}")).then(pl.col(rr) - second_c)
          .otherwise(pl.col(rr) - pl.col(f"rr_max_cand{sfx}")).alias(f"rr_gap_cand{sfx}"))


def stack_features(band: pl.DataFrame) -> pl.DataFrame:
    """band: s1, cand, p, score[, score_2..]. Context features only use pairs inside the band (all we have rr for)."""
    global FEATS
    eps = 1e-6
    d = band.with_columns(lp=(pl.col("p").clip(eps, 1 - eps) / (1 - pl.col("p").clip(eps, 1 - eps))).log())
    d = d.with_columns(p_rank_s1=pl.col("p").rank("ordinal", descending=True).over("s1").cast(pl.Float32),
                       n_band_s1=pl.len().over("s1").cast(pl.Float32),
                       n_band_cand=pl.len().over("cand").cast(pl.Float32))
    srcs = ["score"] + sorted(c for c in band.columns if c.startswith("score_"))
    feats = ["lp", "n_band_s1", "p_rank_s1", "n_band_cand"]
    for i, src in enumerate(srcs):
        sfx = "" if i == 0 else f"_{i + 1}"
        d = _rr_block(d, src, sfx)
        feats += [f + sfx for f in BASE_FEATS]
    if len(srcs) > 1:  # agreement between rerankers (both are logits)
        d = d.with_columns(rr_mean=pl.mean_horizontal([f"rr{'' if i == 0 else f'_{i + 1}'}" for i in range(len(srcs))]),
                           rr_diff=pl.col("rr") - pl.col("rr_2"))
        feats += ["rr_mean", "rr_diff"]
    FEATS = feats
    return d


def X_of(d):
    return d.select(pl.col(c).cast(pl.Float32) for c in FEATS).to_numpy()


def cmd_eval(a, R, T):
    sys.path.insert(0, str(R / "src"))
    from decide import choose, macro_f05, one_to_one  # read-only import of the project scorer

    rr = load_rr(a.rr)
    oof = pl.read_parquet(R / a.oof)
    band = oof.filter(pl.col("p").is_between(*a.band)).join(rr, on=["s1", "cand"], how="inner")
    n_band_all = oof.filter(pl.col("p").is_between(*a.band)).height
    print(f"band pairs {n_band_all:,}, with rr score {band.height:,} ({T()})", flush=True)
    d = stack_features(band)
    X, y = X_of(d), d["y"].to_numpy().astype(np.float32)
    fold = (d["s1"].hash(seed=17) % a.folds).to_numpy()
    p_new = np.zeros(len(y), dtype=np.float32)
    mdir = HERE / "work"
    mdir.mkdir(exist_ok=True)
    for k in range(a.folds):
        tr, va = fold != k, fold == k
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=FEATS), a.rounds,
                      valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        p_new[va] = m.predict(X[va], num_iteration=m.best_iteration)
        m.save_model(str(mdir / f"stacker_{a.tag}_fold{k}.txt"), num_iteration=m.best_iteration)
    from sklearn.metrics import log_loss, roc_auc_score
    rep = {"band": a.band, "n_band": band.height,
           "band_auc_lgb": roc_auc_score(y, d["p"]), "band_auc_rr": roc_auc_score(y, d["rr"]),
           "band_auc_stack": roc_auc_score(y, p_new),
           "band_ll_lgb": log_loss(y, d["p"].clip(1e-6, 1 - 1e-6)), "band_ll_stack": log_loss(y, p_new)}
    print({k: round(v, 5) if isinstance(v, float) else v for k, v in rep.items()}, flush=True)

    new = oof.join(d.select("s1", "cand", p_new=pl.Series(p_new)), on=["s1", "cand"], how="left") \
             .with_columns(p=pl.coalesce("p_new", "p")).drop("p_new")
    rrc = [c for c in d.columns if c in ("rr", "rr_mean", "rr_diff") or (c.startswith("rr_") and c[3:].isdigit())]
    (new.join(d.select(["s1", "cand"] + rrc), on=["s1", "cand"], how="left")
        .join(oof.select("s1", "cand", p_lgb="p"), on=["s1", "cand"]).with_columns(in_band=pl.col("rr").is_not_null())
        .write_parquet(HERE / "work" / f"oof_stacked_{a.tag}.parquet"))  # input for src/stage2.py
    uni = (pl.read_parquet(R / a.oof.replace(".parquet", "_s1.parquet"))
             .join(pl.read_parquet(R / "cache/split/val_source1.parquet", columns=["entity_id", "country"])
                   .rename({"entity_id": "s1"}), on="s1", how="inner"))
    if a.smoke:  # tiny universe: only S1s that have rr scores
        uni = uni.join(band.select("s1").unique(), on="s1", how="semi")
    gt = pl.read_csv(R / "cache/split/full_ground_truth.tsv", separator="\t", infer_schema=False).fill_null("")
    gtp = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids").str.len_chars() > 0)
             .select(s1="source1_entity_id", cand="matched_entity_ids").join(uni.select("s1"), on="s1", how="semi"))
    print(f"honest universe {uni.height:,} S1 ({T()})", flush=True)
    for name, s in (("baseline", oof), ("stacked", new)):
        s11 = one_to_one(s.join(uni.select("s1"), on="s1", how="semi").select("s1", "cand", "p"))
        for t in a.thr_grid:
            r = macro_f05(s11.filter(pl.col("p") >= t).select("s1", "cand"), gtp, uni)
            by = {c: round(v, 5) for c, v in r.group_by("country").agg(pl.col("f").mean()).iter_rows()}
            rep[f"{name}_thr{t}"] = {"all": round(r["f"].mean(), 5), **by}
            print(f"  {name:8s} thr {t:.2f}: {r['f'].mean():.5f}  {by}", flush=True)
        r = macro_f05(choose(s11, 0.0), gtp, uni)
        rep[f"{name}_expF"] = round(r["f"].mean(), 5)
        print(f"  {name:8s} expected-F0.5 sets: {r['f'].mean():.5f}", flush=True)
    best = {n: max((v["all"], k) for k, v in rep.items() if k.startswith(n + "_thr")) for n in ("baseline", "stacked")}
    rep["best"] = best
    print(f"BEST baseline {best['baseline']}  stacked {best['stacked']}  "
          f"delta {best['stacked'][0] - best['baseline'][0]:+.5f} ({T()})")
    write_json(HERE / "work" / f"merge_eval_{a.tag}.json", rep)


def cmd_apply(a, R, T):
    sys.path.insert(0, str(R / "src"))
    from decide import one_to_one

    tp = pl.read_parquet(HERE / "work" / f"test_p_{a.models}.parquet")
    rr = load_rr(a.rr)
    band = tp.filter(pl.col("p").is_between(*a.band)).join(rr, on=["s1", "cand"], how="inner")
    if a.exclude_country:  # e.g. keep LightGBM p for France (unseen in training) if the stacker looks shaky there
        c = pl.read_parquet(R / "cache/norm_trn/test_source1.parquet", columns=["entity_id", "country"])
        band = band.join(c.rename({"entity_id": "s1"}), on="s1").filter(~pl.col("country").is_in(a.exclude_country)) \
                   .drop("country")
    d = stack_features(band)
    files = sorted((HERE / "work").glob(f"stacker_{a.stacker_tag or a.tag}_fold*.txt"))
    assert files, "run `eval` first (it saves the fold stackers)"
    boosters = [lgb.Booster(model_file=str(f)) for f in files]
    X = d.select(pl.col(c).cast(pl.Float32) for c in boosters[0].feature_name()).to_numpy()  # the stacker's own order
    p_new = np.mean([b.predict(X) for b in boosters], axis=0).astype(np.float32)
    new = tp.join(d.select("s1", "cand", p_new=pl.Series(p_new)), on=["s1", "cand"], how="left") \
            .with_columns(p=pl.coalesce("p_new", "p")).drop("p_new")
    rrc = [c for c in d.columns if c in ("rr", "rr_mean", "rr_diff") or (c.startswith("rr_") and c[3:].isdigit())]
    (new.join(d.select(["s1", "cand"] + rrc), on=["s1", "cand"], how="left")
        .join(tp.select("s1", "cand", p_lgb="p"), on=["s1", "cand"]).with_columns(in_band=pl.col("rr").is_not_null())
        .write_parquet(HERE / "work" / f"test_stacked_{a.tag}.parquet"))  # input for src/stage2.py
    print(f"test pairs {tp.height:,}; overrode {band.height:,} band pairs ({T()})", flush=True)
    pred = one_to_one(new).filter(pl.col("p") >= a.thr)
    s1_ids = pl.read_parquet(R / "cache/raw/test_source1.parquet", columns=["entity_id"])["entity_id"]
    out = HERE / f"submission_{a.tag}"
    out.mkdir(exist_ok=True)
    order = pl.DataFrame({"s1": s1_ids})
    cl = order.join(new.group_by("s1").agg(pl.col("cand").unique(maintain_order=True)), on="s1", how="left")
    ml = order.join(pred.sort("p", descending=True).group_by("s1", maintain_order=True).agg(pl.col("cand")),
                    on="s1", how="left")
    for df, col, fn in ((cl, "candidate_entity_ids", "candidate_pairs.tsv"),
                        (ml, "matched_entity_ids", "matching_results.tsv")):
        pl.DataFrame({"source1_entity_id": df["s1"], col: [",".join(x or []) for x in df["cand"].to_list()]}) \
          .write_csv(out / fn, separator="\t", quote_style="never")
    print(f"wrote {out}: {pred.height:,} matches ({T()})", flush=True)
    r = subprocess.run(["python3", "utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"],
                       cwd=R / "data", capture_output=True, text=True)
    print(r.stdout[-800:], r.stderr[-300:])
    print("VALIDATOR", "PASS" if r.returncode == 0 else "FAIL")
    if r.returncode != 0 and (R / "data/utils/validate_submission.py").exists():
        raise SystemExit("validator failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "apply"])
    ap.add_argument("--root", default=str(HERE.parents[1]))
    ap.add_argument("--rr", nargs="+", required=True, help="one or more reranker score files")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--oof", default="cache/oof/full_v3.parquet")
    ap.add_argument("--models", default="full_v3")
    ap.add_argument("--band", type=float, nargs=2, default=[0.02, 0.98])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--thr-grid", type=float, nargs="+", default=[0.65, 0.7, 0.75, 0.8, 0.85])
    ap.add_argument("--thr", type=float, default=0.75)
    ap.add_argument("--stacker-tag", default=None, help="apply: use the stackers saved by `eval --tag <this>`")
    ap.add_argument("--exclude-country", nargs="*", default=[])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    R, T = Path(a.root), Timer()
    (cmd_eval if a.cmd == "eval" else cmd_apply)(a, R, T)


if __name__ == "__main__":
    main()
