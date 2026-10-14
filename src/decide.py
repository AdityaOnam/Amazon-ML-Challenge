"""Phase 4: per-entity decision layer on scored pairs, tuned for macro F0.5.

Steps:
 1. one-to-one: every candidate goes only to the S1 that scores it highest (GT is strictly one-to-one)
 2. per S1, choose the prediction set among {top-m by p} maximising expected F0.5 (m=0 allowed)
 3. optional hard floor `tau` on p
Usage: python decide.py --oof cache/oof/val.parquet --fold val     (grid-searches tau, reports macro F0.5)
"""
import argparse

import numpy as np
import polars as pl

from io_utils import ROOT, gt_long

B2 = 0.25


def one_to_one(s: pl.DataFrame) -> pl.DataFrame:
    """Each candidate kept only for its highest-p S1; exact ties broken deterministically by S1 id."""
    return (s.sort(["cand", "p", "s1"], descending=[False, True, False])
             .filter(pl.int_range(pl.len()).over("cand") == 0))


def choose(s: pl.DataFrame, tau: float, max_m: int = 12) -> pl.DataFrame:
    """Per S1: pick m maximising E[F0.5] ~ 1.25 TP / (1.25 TP + 0.25 FN + FP), with m=0 scored as P(no match)."""
    s = s.filter(pl.col("p") >= 0.01).sort(["s1", "p"], descending=[False, True])
    s = s.with_columns(rk=pl.int_range(pl.len()).over("s1") + 1,
                       T=pl.col("p").sum().over("s1"),
                       p_none=(1 - pl.col("p")).log().sum().over("s1").exp())
    s = s.with_columns(tp=pl.col("p").cum_sum().over("s1"))
    s = s.with_columns(ef=(1 + B2) * pl.col("tp") / ((1 + B2) * pl.col("tp") + B2 * (pl.col("T") - pl.col("tp"))
                                                     + (pl.col("rk") - pl.col("tp"))) * (1 - pl.col("p_none")))
    s = s.filter(pl.col("rk") <= max_m)
    best = (s.group_by("s1").agg(pl.col("ef").max().alias("ef_best"), pl.col("p_none").first(),
                                 pl.col("rk").sort_by("ef").last().alias("m")))
    best = best.with_columns(m=pl.when(pl.col("p_none") >= pl.col("ef_best")).then(0).otherwise(pl.col("m")))
    s = s.join(best.select("s1", "m"), on="s1").filter((pl.col("rk") <= pl.col("m")) & (pl.col("p") >= tau))
    return s.select("s1", "cand", "p")


def macro_f05(pred: pl.DataFrame, gt_pairs: pl.DataFrame, all_s1: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 F0.5 exactly as the official metric (singletons: 1 if empty prediction, else 0)."""
    tp = pred.join(gt_pairs, on=["s1", "cand"], how="inner").group_by("s1").len().rename({"len": "tp"})
    npred = pred.group_by("s1").len().rename({"len": "npred"})
    ntrue = gt_pairs.group_by("s1").len().rename({"len": "ntrue"})
    e = (all_s1.join(npred, on="s1", how="left").join(ntrue, on="s1", how="left").join(tp, on="s1", how="left")
               .fill_null(0))
    P = pl.col("tp") / pl.col("npred")
    R = pl.col("tp") / pl.col("ntrue")
    f = (pl.when((pl.col("ntrue") == 0) & (pl.col("npred") == 0)).then(1.0)
           .when((pl.col("ntrue") == 0) | (pl.col("npred") == 0) | (pl.col("tp") == 0)).then(0.0)
           .otherwise((1 + B2) * P * R / (B2 * P + R)))
    return e.with_columns(f=f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--fold", default="val")
    ap.add_argument("--val-only", action="store_true", help="score only S1s of the val fold (honest for full-fold OOF)")
    a = ap.parse_args()
    s = pl.read_parquet(ROOT / a.oof if not a.oof.startswith("/") else a.oof)
    gt = pl.read_csv(ROOT / "cache/split" / f"{a.fold}_ground_truth.tsv", separator="\t", infer_schema=False).fill_null("")
    gtp = gt_long(gt).select(s1="source1_entity_id", cand="match_id")
    s1c = pl.read_parquet(ROOT / "cache/split" / f"{a.fold}_source1.parquet", columns=["entity_id", "country"]) \
            .rename({"entity_id": "s1"})
    uni = ROOT / str(a.oof).replace(".parquet", "_s1.parquet")
    if uni.exists():
        s1c = s1c.join(pl.read_parquet(uni), on="s1", how="semi")
    if a.val_only:
        s1c = s1c.join(pl.read_parquet(ROOT / "cache/split/val_source1.parquet", columns=["entity_id"])
                       .rename({"entity_id": "s1"}), on="s1", how="semi")
    gtp = gtp.join(s1c.select("s1"), on="s1", how="semi")
    print(f"scoring universe: {s1c.height:,} S1")
    s11 = one_to_one(s)
    print(f"pairs {s.height:,} -> after one-to-one {s11.height:,}")
    for label, src in [("1:1", s11)]:
        for thr in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85]:
            pred = src.filter(pl.col("p") >= thr).select("s1", "cand")
            r = macro_f05(pred, gtp, s1c)
            print(f"  plain threshold {thr:.2f} [{label}]  macro F0.5 = {r['f'].mean():.5f}")
    best = None
    for tau in [0.0, 0.05, 0.1, 0.2, 0.3]:
        pred = choose(s11, tau)
        r = macro_f05(pred, gtp, s1c)
        by = r.group_by("country").agg(pl.col("f").mean()).sort("country")
        score = r["f"].mean()
        print(f"  expected-F0.5 sets, tau={tau:.2f}: macro F0.5 = {score:.5f}  "
              + "  ".join(f"{c}={v:.5f}" for c, v in by.iter_rows()), flush=True)
        if best is None or score > best[0]:
            best = (score, tau, r)
    score, tau, r = best
    sing = r.filter(pl.col("ntrue") == 0)
    print(f"BEST macro F0.5 = {score:.5f} (tau={tau}); singletons correct {sing['f'].mean():.4f}, "
          f"non-singleton mean {r.filter(pl.col('ntrue') > 0)['f'].mean():.5f}")


if __name__ == "__main__":
    main()
