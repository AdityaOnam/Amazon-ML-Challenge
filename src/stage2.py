"""Stage 2: re-score pairs on top of the reranker-stacked p (research/10_residual_errors_v5.md fix #1).

Features per pair (all pairs with LightGBM p >= 1e-3):
  H  house-number edit type (digit offset, Hamming, prefix/suffix, transposition, number found elsewhere)
  K  collective agreement with the S1's other confident matches ("anchors", stacked p > 0.9):
     address / name / house-number agreement between this candidate and the anchors, and the name margin
     against the candidate's best rival S1
  R  raw pair features from the first-stage table
  (+ stacked p, logit LightGBM p, reranker logit, in-band flag, S3 flag, empty-address flag)
The competition group of the research prototype is left out: it depends on how many S1 compete, which differs
between the OOF sample and test (the report's safer variant, +0.00053).

train: cross-fitted honest evaluation on the OOF stacked table, then 5 fold models saved to runs/models/.
apply: average of the fold models on the test stacked table; --keep-country keeps stacked p there (France:
       stage 2 never saw it); optional --extra adds pairs scored outside the stacked table (v4 widening).
Usage:
  python stage2.py train --stacked research/gpu_exp/work/oof_stacked_bge_full.parquet --tag s2_bge
  python stage2.py apply --stacked research/gpu_exp/work/test_stacked_bge_fr2.parquet --tag s2_bge \
         --norm fr --feat cache/feat/test.parquet --keep-country France --out-tag v8
"""
import argparse
import subprocess
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from decide import macro_f05, one_to_one
from io_utils import ROOT, gt_long, write_id_lists

FLOOR = 1e-3
BASE = ["p", "lp_lgb", "rr", "in_band", "src3", "c_empty"]
H = ["L1", "L2", "hn_both", "hn_off", "hn_ham", "hn_dlen", "hn_prefix", "hn_suffix", "hn_firstdiff",
     "hn_lastdiff_from_end", "hn_transpose", "hn2_in_nums1", "hn1_in_nums2", "hn_rel_off", "hn_type"]
K = ["k_n", "k_ad", "k_hn", "k_hoff", "k_nm", "k_nm_mean", "k_ad_x", "k_hn_x", "k_nm_x", "k_nm_same",
     "k_nm_best_other", "k_nm_margin"]
RAW = ["n_tset", "n_tsort", "a_tset", "a_ratio", "hnum_lev", "city_ratio", "nf_s1", "nf_pool", "nf_pool_c",
       "u1_min_df", "n_cands", "cand_n_s1", "s_emb", "a_num_jac", "addr_len2", "legal"]
C = ["c_n", "c_rank", "c_other_max", "c_other_sum", "s_rank", "s_rank_src", "s_n90", "s_n90_src", "s_psum", "s_n"]
A = ["a_rr", "a_rr_minus_rr", "a_has"]
MULTI = ["rr_2", "rr_3", "rr_4", "rr_mean", "rr_diff"]
FEATS = BASE + H + K + RAW  # + C / A when requested (--groups)
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=48)
REC = ["entity_id", "name_core", "addr_clean", "addr_hnum", "addr_nums"]
T0 = time.time()
GROUPS = ""
ANCHOR = None  # (anchor map s1,cand,anchor ; anchor scores s1(=anchor),cand,score)


def T():
    return f"{time.time() - T0:.0f}s"


def records(split, norm, ids, which):
    out = [pl.scan_parquet(ROOT / f"cache/norm_{norm}/{split}_source{i}.parquet").select(REC)
             .join(ids.lazy(), on="entity_id", how="semi").collect() for i in which]
    return pl.concat(out)


def build(st: pl.DataFrame, split: str, norm: str, feat: str) -> pl.DataFrame:
    """st: s1, cand, p (stacked), p_lgb, rr, in_band [, y]."""
    st = st.filter(pl.col("p_lgb") >= FLOOR)
    R1 = records(split, norm, st.select(entity_id="s1").unique(), (1,))
    R2 = records(split, norm, st.select(entity_id="cand").unique(), (2, 3))
    d = (st.join(R1.rename({c: c + "_1" for c in REC[1:]}), left_on="s1", right_on="entity_id", how="left")
           .join(R2.rename({c: c + "_2" for c in REC[1:]}), left_on="cand", right_on="entity_id", how="left"))
    # ---- house-number edit type
    d = d.with_columns(h1=pl.col("addr_hnum_1").str.extract(r"(\d+)"), h2=pl.col("addr_hnum_2").str.extract(r"(\d+)"))
    d = d.with_columns(L1=pl.col("h1").str.len_chars().cast(pl.Int8), L2=pl.col("h2").str.len_chars().cast(pl.Int8),
                       i1=pl.col("h1").str.slice(0, 9).cast(pl.Int64, strict=False),
                       i2=pl.col("h2").str.slice(0, 9).cast(pl.Int64, strict=False))
    ham = sum(((pl.col("h1").str.slice(i, 1) != pl.col("h2").str.slice(i, 1)) & (pl.col("L1") > i)).cast(pl.Int8)
              for i in range(9))
    first_diff = pl.min_horizontal([pl.when(pl.col("h1").str.slice(i, 1) != pl.col("h2").str.slice(i, 1)).then(i)
                                    .otherwise(99) for i in range(9)])
    d = d.with_columns(
        hn_both=pl.col("h1").is_not_null() & pl.col("h2").is_not_null(),
        hn_off=(pl.col("i1") - pl.col("i2")).abs().cast(pl.Float32),
        hn_ham=pl.when(pl.col("L1") == pl.col("L2")).then(ham).otherwise(None).cast(pl.Int8),
        hn_dlen=(pl.col("L1") - pl.col("L2")).cast(pl.Int8),
        hn_prefix=(pl.col("h1").str.starts_with(pl.col("h2")) | pl.col("h2").str.starts_with(pl.col("h1"))),
        hn_suffix=(pl.col("h1").str.ends_with(pl.col("h2")) | pl.col("h2").str.ends_with(pl.col("h1"))),
        hn_firstdiff=first_diff.cast(pl.Int8),
        hn_lastdiff_from_end=(pl.min_horizontal(pl.col("L1"), pl.col("L2")) - 1 - first_diff).cast(pl.Int8),
        hn_transpose=(pl.col("L1") == pl.col("L2")) & (pl.col("h1") != pl.col("h2"))
                     & (pl.col("h1").str.split("").list.sort().list.join("")
                        == pl.col("h2").str.split("").list.sort().list.join("")),
        hn2_in_nums1=(pl.col("h2").is_not_null() & pl.col("addr_nums_1").str.split(" ").list.contains(pl.col("h2"))),
        hn1_in_nums2=(pl.col("h1").is_not_null() & pl.col("addr_nums_2").str.split(" ").list.contains(pl.col("h1"))),
        hn_rel_off=((pl.col("i1") - pl.col("i2")).abs()
                    / (pl.max_horizontal(pl.col("i1"), pl.col("i2")) + 1)).cast(pl.Float32))
    d = d.with_columns(hn_type=pl.when(~pl.col("hn_both")).then(0).when(pl.col("h1") == pl.col("h2")).then(1)
                       .when((pl.col("hn_ham") == 1) & (pl.col("hn_off") > 10)).then(2)
                       .when(pl.col("hn_off") <= 10).then(3).when(pl.col("hn_transpose")).then(4)
                       .when(pl.col("hn_prefix") | pl.col("hn_suffix")).then(5).otherwise(6).cast(pl.Int8))
    miss = lambda c: pl.col(c).fill_null("") == ""  # noqa: E731
    d = d.with_columns(c_empty=miss("addr_clean_2"), src3=pl.col("cand").str.starts_with("S3"))
    print(f"  house numbers done ({T()})", flush=True)
    # ---- collective: agreement with the S1's other anchors (stacked p > 0.9)
    anch = d.filter(pl.col("p") > 0.9).select("s1", cand_b="cand", nm_b="name_core_2", ad_b="addr_clean_2",
                                             hn_b="h2", ib="i2", src_b="src3")
    pr = (d.select("s1", cand_a="cand", nm_a="name_core_2", ad_a="addr_clean_2", hn_a="h2", ia="i2", src_a="src3")
           .join(anch, on="s1").filter(pl.col("cand_a") != pl.col("cand_b")))
    print(f"  anchor pairs {pr.height:,} ({T()})", flush=True)
    kw = dict(workers=48, dtype=np.float32)
    pr = pr.with_columns(
        ad=pl.Series(process.cpdist(pr["ad_a"].fill_null("").to_list(), pr["ad_b"].fill_null("").to_list(),
                                    scorer=fuzz.token_set_ratio, **kw)),
        nm=pl.Series(process.cpdist(pr["nm_a"].fill_null("").to_list(), pr["nm_b"].fill_null("").to_list(),
                                    scorer=fuzz.token_sort_ratio, **kw)),
        ae=miss("ad_a") | miss("ad_b"),
        hn=pl.col("hn_a").is_not_null() & (pl.col("hn_a") == pl.col("hn_b")),
        hoff=(pl.col("ia") - pl.col("ib")).abs().cast(pl.Float32),
        xs=pl.col("src_a") != pl.col("src_b"))
    cf = pr.group_by("s1", "cand_a").agg(
        k_n=pl.len().cast(pl.Int16), k_ad=pl.col("ad").filter(~pl.col("ae")).max(), k_hn=pl.col("hn").any(),
        k_hoff=pl.col("hoff").min(), k_nm=pl.col("nm").max(), k_nm_mean=pl.col("nm").mean(),
        k_ad_x=pl.col("ad").filter(~pl.col("ae") & pl.col("xs")).max(), k_hn_x=pl.col("hn").filter(pl.col("xs")).any(),
        k_nm_x=pl.col("nm").filter(pl.col("xs")).max(), k_nm_same=pl.col("nm").filter(~pl.col("xs")).max())
    d = d.join(cf.rename({"cand_a": "cand"}), on=["s1", "cand"], how="left").with_columns(pl.col("k_n").fill_null(0))
    top2 = pl.col("k_nm").fill_null(-1).sort(descending=True).slice(0, 2).implode().over("cand")
    d = d.with_columns(k_top2=top2)
    d = d.with_columns(k_nm_best_other=pl.when(pl.col("k_nm").fill_null(-1) >= pl.col("k_top2").list.first())
                       .then(pl.col("k_top2").list.get(1, null_on_oob=True))
                       .otherwise(pl.col("k_top2").list.first()).cast(pl.Float32))
    d = d.with_columns(k_nm_margin=(pl.col("k_nm").fill_null(-1) - pl.col("k_nm_best_other").fill_null(-1)))
    eps = 1e-6
    d = d.with_columns(lp_lgb=(pl.col("p_lgb").clip(eps, 1 - eps) / (1 - pl.col("p_lgb").clip(eps, 1 - eps))).log())
    raw = (pl.scan_parquet(ROOT / feat).select(["s1", "cand"] + RAW)
             .join(d.lazy().select("s1", "cand"), on=["s1", "cand"], how="semi").collect())
    d = d.join(raw, on=["s1", "cand"], how="left")
    if "M" in GROUPS:  # every reranker's own score (and their mean / disagreement), not just the stacked p
        for c in MULTI:
            if c not in d.columns:
                d = d.with_columns(pl.lit(None, pl.Float32).alias(c))
    if "C" in GROUPS:  # competition on stacked p: rival S1s for the same candidate, the S1's own list
        d = d.sort("cand", "p", descending=[False, True]).with_columns(
            c_n=pl.len().over("cand").cast(pl.Int16), c_rank=(pl.int_range(pl.len()).over("cand") + 1).cast(pl.Int16),
            c_p1=pl.col("p").first().over("cand"), c_p2=pl.col("p").slice(1, 1).first().over("cand"),
            c_psum=pl.col("p").sum().over("cand"))
        d = d.with_columns(c_other_max=pl.when(pl.col("c_rank") == 1).then(pl.col("c_p2").fill_null(0))
                           .otherwise(pl.col("c_p1")), c_other_sum=pl.col("c_psum") - pl.col("p"))
        d = d.sort("s1", "p", descending=[False, True]).with_columns(
            s_rank=(pl.int_range(pl.len()).over("s1") + 1).cast(pl.Int16),
            s_rank_src=(pl.int_range(pl.len()).over("s1", "src3") + 1).cast(pl.Int16),
            s_n90=(pl.col("p") > 0.9).sum().over("s1").cast(pl.Int16),
            s_n90_src=(pl.col("p") > 0.9).sum().over("s1", "src3").cast(pl.Int16),
            s_psum=pl.col("p").sum().over("s1"), s_n=pl.len().over("s1").cast(pl.Int16))
    if "A" in GROUPS:  # reranker score of the candidate against the S1's most confident OTHER match
        am = pl.read_parquet(ROOT / ANCHOR[0]).join(
            pl.read_parquet(ROOT / ANCHOR[1]).rename({"s1": "anchor", "score": "a_rr"}), on=["anchor", "cand"], how="left")
        d = d.join(am.select("s1", "cand", "a_rr"), on=["s1", "cand"], how="left").with_columns(
            a_rr_minus_rr=pl.col("a_rr") - pl.col("rr"), a_has=pl.col("a_rr").is_not_null())
    print(f"  features for {d.height:,} pairs ({T()})", flush=True)
    return d.select(["s1", "cand"] + (["y"] if "y" in d.columns else []) + FEATS)


def X_of(d):
    return d.select(pl.col(c).cast(pl.Float32) for c in FEATS).to_numpy()


def honest(p: pl.DataFrame, tag: str):
    uni = (pl.read_parquet(ROOT / "cache/oof/full_v3_s1.parquet")
             .join(pl.read_parquet(ROOT / "cache/split/val_source1.parquet", columns=["entity_id", "country"])
                   .rename({"entity_id": "s1"}), on="s1"))
    gtp = (gt_long(pl.read_csv(ROOT / "cache/split/full_ground_truth.tsv", separator="\t", infer_schema=False)
                   .fill_null("")).select(s1="source1_entity_id", cand="match_id").join(uni, on="s1", how="semi"))
    s11 = one_to_one(p.join(uni.select("s1"), on="s1", how="semi").select("s1", "cand", "p"))
    for t in (0.7, 0.75, 0.8):
        r = macro_f05(s11.filter(pl.col("p") >= t).select("s1", "cand"), gtp, uni)
        by = "  ".join(f"{c}={v:.5f}" for c, v in r.group_by("country").agg(pl.col("f").mean()).sort("country").iter_rows())
        print(f"  {tag:8s} thr {t:.2f}: {r['f'].mean():.5f}   {by}", flush=True)


def cmd_train(a):
    st = pl.read_parquet(ROOT / a.stacked)
    feat = a.feat or "cache/feat/full_v3.parquet"
    d = build(st, "train", "trn", feat)
    d_full = d
    if a.dropmix:  # density-robust training (research/16, 21): test has ~19% of its S1 missing (their copies stay in
        # S2/S3), so half of the training S1 (hash 77 odd) get features built on a table without 20% of the S1
        # (hash 321); the others keep full-density features. Only provided train S1 are subsampled, nothing is added.
        s1s = st.select("s1").unique()
        keep = s1s.filter((pl.col("s1").hash(seed=321) % 1000) >= 200)
        dtr = build(st.join(keep, on="s1", how="semi"), "train", "trn", feat)
        h = lambda df: (df["s1"].hash(seed=77) % 2).to_numpy()  # noqa: E731
        d = pl.concat([d.filter(pl.Series(h(d) == 0)), dtr.filter(pl.Series(h(dtr) == 1))])
        print(f"  dropmix training table {d.height:,} ({T()})", flush=True)
    X, y = X_of(d), d["y"].to_numpy().astype(np.float32)
    fold = (d["s1"].hash(seed=17) % 5).to_numpy()
    inner = ((d["s1"].hash(seed=99) % 10) == 0).to_numpy()
    p2 = np.zeros(len(y), dtype=np.float32)
    for k in range(5):
        tr, va, es = (fold != k) & ~inner, fold == k, (fold != k) & inner
        for j in range(a.seeds):  # bagging seeds; apply averages every saved model of the tag
            prm = dict(PARAMS, seed=j, bagging_seed=j, feature_fraction_seed=j)
            m = lgb.train(prm, lgb.Dataset(X[tr], y[tr], feature_name=FEATS), 1500,
                          valid_sets=[lgb.Dataset(X[es], y[es])],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            p2[va] += m.predict(X[va], num_iteration=m.best_iteration) / a.seeds
            sfx = f"_s{j}" if a.seeds > 1 else ""
            m.save_model(str(ROOT / f"runs/models/{a.tag}_fold{k}{sfx}.txt"), num_iteration=m.best_iteration)
            print(f"  fold {k} seed {j}: {m.best_iteration} iters ({T()})", flush=True)
    if a.dropmix:  # honest report at full density: cross-fitted prediction of the full-density table
        Xf, ff = X_of(d_full), (d_full["s1"].hash(seed=17) % 5).to_numpy()
        p2 = np.zeros(d_full.height, dtype=np.float32)
        for k in range(5):
            for f in sorted((ROOT / "runs/models").glob(f"{a.tag}_fold{k}*.txt")):
                p2[ff == k] += lgb.Booster(model_file=str(f)).predict(Xf[ff == k]) / a.seeds
        d = d_full
    new = st.join(d.select("s1", "cand", p2=pl.Series(p2)), on=["s1", "cand"], how="left") \
            .with_columns(p=pl.coalesce("p2", "p")).drop("p2")
    honest(st, "stacked")
    honest(new, "stage2")


def cmd_apply(a):
    st = pl.read_parquet(ROOT / a.stacked)
    if a.cand_min > 0:  # meta-blocking: only pairs the first-stage LightGBM keeps (p_lgb >= cand_min) are candidates
        n0, st_keys = st.height, st.select("s1", "cand")
        st = st.filter(pl.col("p_lgb") >= a.cand_min)
        print(f"meta-blocking p_lgb >= {a.cand_min}: {st.height:,} of {n0:,} pairs kept", flush=True)
    d = build(st, "test", a.norm, a.feat or "cache/feat/test.parquet")
    c = pl.read_parquet(ROOT / "cache/norm_trn/test_source1.parquet", columns=["entity_id", "country"])
    if a.keep_country:
        d = d.join(c.filter(pl.col("country").is_in(a.keep_country)).select(s1="entity_id"), on="s1", how="anti")
    ms = [lgb.Booster(model_file=str(f)) for f in sorted((ROOT / "runs/models").glob(f"{a.tag}_fold*.txt"))]
    p2 = np.mean([m.predict(X_of(d)) for m in ms], axis=0).astype(np.float32)
    new = st.select("s1", "cand", "p").join(d.select("s1", "cand", p2=pl.Series(p2)), on=["s1", "cand"], how="left") \
            .with_columns(p=pl.coalesce("p2", "p")).drop("p2")
    print(f"stage 2 re-scored {d.height:,} of {st.height:,} pairs ({T()})", flush=True)
    if a.extra:  # widening pairs: s1, cand, p from a table that also holds the base pairs
        ex = pl.read_parquet(ROOT / a.extra).join(new.select("s1", "cand"), on=["s1", "cand"], how="anti")
        if a.cand_min > 0:  # base pairs cut by meta-blocking must not come back through the extras table
            ex = ex.join(st_keys, on=["s1", "cand"], how="anti")
        if a.cand_min > 0:
            ex = ex.filter(pl.col("p") >= a.cand_min)
        skip = a.keep_country + a.no_extra_country  # kept countries stay exactly as stacked: no widening either
        if skip:
            ex = ex.join(c.filter(pl.col("country").is_in(skip)).select(s1="entity_id"), on="s1", how="anti")
        new = pl.concat([new, ex.select("s1", "cand", "p")])
        print(f"  + {ex.height:,} extra pairs", flush=True)
    if a.fr_swap:  # France-only descriptor-swap demotion (fr_swap.py), then one-to-one re-runs on the rest
        from fr_swap import flag_pairs
        frs = c.filter(pl.col("country") == "France").select(s1="entity_id")
        cand_fr = new.join(frs, on="s1", how="semi").filter(pl.col("p") >= 0.3)
        sw = flag_pairs(cand_fr, a.norm).filter(pl.col("dswap"))
        n_hi = sw.join(cand_fr.filter(pl.col("p") >= a.thr), on=["s1", "cand"], how="semi").height
        new = new.join(sw.select("s1", "cand", "dswap"), on=["s1", "cand"], how="left") \
                 .with_columns(p=pl.when(pl.col("dswap")).then(0.0).otherwise(pl.col("p")).cast(pl.Float32)).drop("dswap")
        print(f"  descriptor swaps demoted: {sw.height:,} (of which p >= thr before: {n_hi:,})", flush=True)
    pred = one_to_one(new).filter(pl.col("p") >= a.thr)
    out = ROOT / "output" / a.out_tag
    out.mkdir(parents=True, exist_ok=True)
    order = pl.DataFrame({"s1": pl.read_parquet(ROOT / "cache/raw/test_source1.parquet", columns=["entity_id"])["entity_id"]})
    cl = order.join(new.group_by("s1").agg(pl.col("cand").unique(maintain_order=True)), on="s1", how="left")
    ml = order.join(pred.sort("p", descending=True).group_by("s1", maintain_order=True).agg(pl.col("cand")),
                    on="s1", how="left")
    write_id_lists(cl["s1"], [x or [] for x in cl["cand"].to_list()], out / "candidate_pairs.tsv", "candidate_entity_ids")
    write_id_lists(ml["s1"], [x or [] for x in ml["cand"].to_list()], out / "matching_results.tsv", "matched_entity_ids")
    print(f"wrote {out}: {pred.height:,} matches ({T()})", flush=True)
    r = subprocess.run(["python3", "utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "dataset/test", "--check-ids"],
                       cwd=ROOT / "data", capture_output=True, text=True)
    print(r.stdout[-400:], r.stderr[-300:])
    print("VALIDATOR", "PASS" if r.returncode == 0 else "FAIL")
    if r.returncode != 0 and (ROOT / "data/utils/validate_submission.py").exists():
        raise SystemExit("validator failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "apply"])
    ap.add_argument("--stacked", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feat", default=None)
    ap.add_argument("--norm", default="trn")
    ap.add_argument("--keep-country", nargs="*", default=[])
    ap.add_argument("--extra", default=None)
    ap.add_argument("--no-extra-country", nargs="*", default=[], help="countries that get no widening pairs")
    ap.add_argument("--fr-swap", action="store_true", help="France-only descriptor-swap demotion")
    ap.add_argument("--thr", type=float, default=0.75)
    ap.add_argument("--cand-min", type=float, default=0.0, help="meta-blocking: drop pairs with first-stage p below this")
    ap.add_argument("--out-tag", default="stage2")
    ap.add_argument("--groups", default="", help="extra feature groups: C (competition), A (anchor reranker)")
    ap.add_argument("--seeds", type=int, default=1, help="average this many bagging seeds per fold")
    ap.add_argument("--dropmix", action="store_true", help="density-robust training (half the S1 at 20%% fewer S1)")
    ap.add_argument("--anchor-map", default=None)
    ap.add_argument("--anchor-rr", default=None)
    a = ap.parse_args()
    global GROUPS, ANCHOR, FEATS
    GROUPS = a.groups
    FEATS = FEATS + (C if "C" in GROUPS else []) + (A if "A" in GROUPS else []) + (MULTI if "M" in GROUPS else [])
    ANCHOR = (a.anchor_map, a.anchor_rr)
    cmd_train(a) if a.cmd == "train" else cmd_apply(a)


if __name__ == "__main__":
    main()
