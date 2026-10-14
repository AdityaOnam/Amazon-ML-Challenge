"""R10: data bundle for the candidate-side listwise "select" cross-encoder (runs on the HOME server, CPU only).

A CONTEST is one candidate record (S2/S3) with >= 2 S1s whose current stacked p > --p-contest (0.05); its
options are the top --k (6) of those S1s by p. The select model reads `cand + all options` in one pass and
returns a softmax over {options, none}; sel_prob(s1, cand) goes back into the stacker as an extra --rr input.
One-to-one holds exactly in the ground truth, so a candidate has 0 or 1 true option (checked, see manifest).

HONESTY (same cross-fitting as experiment A; H_A = cache/oof/full_v3_s1.parquet, H_B = the other train S1s):
  * model_B trains on H_B S1s only, model_A on H_A S1s only (S1-disjoint, like rr_train / bundle_B).
  * A contest's sel_prob for an H_A option is taken from model_B, for an H_B option from model_A.
    oof_A = every OOF contest with >= 1 H_A option (scored by model_B; keep only rows with ha=True)
    oof_B = every OOF contest with >= 1 H_B option (scored by model_A; keep only rows with ha=False)
    MIXED contests (options from both halves, ~70% of all) are in both sets, scored by both models, and each
    option keeps the score of the model that never saw its half. The other-half options are only context.
  * Training contests of model_X are built from candidates that model_X never scores (no H_other option at
    p > p_contest), with the options restricted to half-X S1s (pairs with p >= --floor, top k by p, >= 2
    options). If the true S1 is not among them the label is "none". So no candidate record is both trained
    on and scored by the same model, and no model sees labels of the S1s it scores (the rule of export_bundle.py:
    "any H_B training pair whose candidate appears in the H_A band is dropped", applied per contest).
  * Test contests (no labels) are scored by both models and averaged. US/India contests come from
    --test-main, France contests from --test-fr (the French re-normalised base). Test records are never trained on.

Outputs (--out, default research/gpu_exp/bundle_sel/):
  train_A.parquet / train_B.parquet   cand, s1[list], y[list], p[list], hard     (training contests)
  oof_A.parquet / oof_B.parquet       cand, s1[list], y[list], p[list], ha[list], country   (scoring)
  test.parquet                        cand, s1[list], p[list], country                       (scoring)
  records.parquet                     entity_id, country, business_name, business_address (only referenced ids)
  manifest.json                       counts + contest statistics
Usage:  python export_select_bundle.py            (~1 min)
"""
import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, Timer, write_json  # noqa: E402


def contests(pairs: pl.DataFrame, k: int, min_opt: int = 2, extra=()) -> pl.DataFrame:
    """pairs(s1, cand, p, ...) -> one row per candidate with list columns, options sorted by p desc, top k."""
    cols = ["s1", "p"] + [c for c in ("y", "ha") if c in pairs.columns]
    g = (pairs.sort(["cand", "p", "s1"], descending=[False, True, False])
              .group_by("cand", maintain_order=True)
              .agg([pl.col(c).head(k) for c in cols] + [pl.col(c).first() for c in extra]))
    return g.filter(pl.col("s1").list.len() >= min_opt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[1]))
    ap.add_argument("--out", default=str(HERE / "bundle_sel"))
    ap.add_argument("--oof", default="research/gpu_exp/work/oof_stacked_v3b_ABE.parquet")
    ap.add_argument("--ha", default="cache/oof/full_v3_s1.parquet", help="H_A S1 list (the rest of train = H_B)")
    ap.add_argument("--test-main", default="research/gpu_exp/work/test_stacked_v3b_ABE.parquet")
    ap.add_argument("--test-fr", default="research/gpu_exp/work/test_stacked_v3b_ABE_fr2.parquet")
    ap.add_argument("--fr-countries", nargs="*", default=["France"], help="test countries taken from --test-fr")
    ap.add_argument("--records", default="research/gpu_exp/bundle/records.parquet")
    ap.add_argument("--p-contest", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=6, help="max options per contest")
    ap.add_argument("--floor", type=float, default=0.001, help="min p of a training option")
    ap.add_argument("--sample", type=float, default=1.0, help="<1: keep this fraction of candidates (smoke)")
    a = ap.parse_args()
    R, out, T = Path(a.root), Path(a.out), Timer()
    out.mkdir(parents=True, exist_ok=True)
    man = {"args": vars(a)}
    samp = (lambda d: d.filter((pl.col("cand").hash(seed=7) % 1_000_000) < int(a.sample * 1_000_000))) \
        if a.sample < 1 else (lambda d: d)

    # ------------------------------------------------ OOF (train S1s, 100% S1 OOF base)
    ha = pl.read_parquet(R / a.ha).select("s1").with_columns(ha=pl.lit(True))
    o = samp(pl.scan_parquet(R / a.oof).select("s1", "cand", "y", "p").filter(pl.col("p") >= a.floor).collect())
    o = o.join(ha, on="s1", how="left").with_columns(pl.col("ha").fill_null(False))
    nt = o.filter(pl.col("y")).group_by("cand").len()
    man["gt_cands_with_gt_2plus_s1"] = int((nt["len"] > 1).sum())  # one-to-one check (expected 0)
    s1c = pl.read_parquet(R / a.records, columns=["entity_id", "country"]).rename({"entity_id": "s1"})

    hot = o.filter(pl.col("p") > a.p_contest)
    oof_all = contests(hot, a.k).join(
        hot.select("s1", "cand").join(s1c, on="s1").group_by("cand").agg(pl.col("country").first()), on="cand")
    oof_all = oof_all.with_columns(n_ha=pl.col("ha").list.sum(), n=pl.col("s1").list.len())
    oof_A = oof_all.filter(pl.col("n_ha") > 0)          # scored by model_B
    oof_B = oof_all.filter(pl.col("n_ha") < pl.col("n"))  # scored by model_A
    keep = ["cand", "s1", "y", "p", "ha", "country"]
    oof_A.select(keep).write_parquet(out / "oof_A.parquet")
    oof_B.select(keep).write_parquet(out / "oof_B.parquet")
    man["oof_contests"] = {"all": oof_all.height, "pure_A": int((oof_all["n_ha"] == oof_all["n"]).sum()),
                           "pure_B": int((oof_all["n_ha"] == 0).sum()),
                           "mixed": int(((oof_all["n_ha"] > 0) & (oof_all["n_ha"] < oof_all["n"])).sum()),
                           "true_labelled": int((oof_all["y"].list.sum() > 0).sum()),
                           "options": int(oof_all["n"].sum()),
                           "oof_A": oof_A.height, "oof_B": oof_B.height}
    # true S1 of a contest outside the top k (label becomes none): how much the k cap costs
    full_n = hot.group_by("cand").agg(ny=pl.col("y").sum(), n=pl.len()).filter(pl.col("n") >= 2)
    man["oof_contests"]["true_outside_topk"] = int(full_n["ny"].sum() - oof_all["y"].list.sum().sum())
    man["oof_contests"]["n_options_hist"] = {int(r[0]): int(r[1]) for r in
                                             oof_all["n"].value_counts().sort("n").iter_rows()}
    print(f"OOF contests {man['oof_contests']} ({T()})", flush=True)

    # ------------------------------------------------ training contests (S1-disjoint, candidate-disjoint)
    for half, flag, scored_by_me in (("A", True, oof_B), ("B", False, oof_A)):
        pool = o.filter(pl.col("ha") == flag).join(scored_by_me.select("cand"), on="cand", how="anti")
        tr = contests(pool.drop("ha"), a.k)
        tr = tr.with_columns(hard=pl.col("p").list.get(1) > a.p_contest).sample(fraction=1.0, shuffle=True, seed=11)
        tr.write_parquet(out / f"train_{half}.parquet")
        man[f"train_{half}"] = {"contests": tr.height, "hard": int(tr["hard"].sum()),
                                "p2_gt_0.005": int((tr["p"].list.get(1) > 0.005).sum()),
                                "true_labelled": int((tr["y"].list.sum() > 0).sum()),
                                "options": int(tr["s1"].list.len().sum())}
        print(f"train_{half} {man[f'train_{half}']} ({T()})", flush=True)

    # ------------------------------------------------ test contests (scored only)
    tc = pl.read_parquet(R / "cache/norm_trn/test_source1.parquet", columns=["entity_id", "country"]) \
           .rename({"entity_id": "s1"})
    fr = tc.filter(pl.col("country").is_in(a.fr_countries)).select("s1")
    tm = pl.scan_parquet(R / a.test_main).select("s1", "cand", "p").filter(pl.col("p") > a.p_contest).collect() \
           .join(fr, on="s1", how="anti")
    tf = pl.scan_parquet(R / a.test_fr).select("s1", "cand", "p").filter(pl.col("p") > a.p_contest).collect() \
           .join(fr, on="s1", how="semi")
    th = samp(pl.concat([tm, tf]).join(tc, on="s1"))
    test = contests(th, a.k, extra=("country",))
    test.select("cand", "s1", "p", "country").write_parquet(out / "test.parquet")
    man["test_contests"] = {"all": test.height, "options": int(test["s1"].list.len().sum()),
                            **{f"country_{c}": int(n) for c, n in test["country"].value_counts().iter_rows()}}
    print(f"test contests {man['test_contests']} ({T()})", flush=True)

    # ------------------------------------------------ slim records (only referenced ids)
    ids = [d.select(pl.col("s1").explode()) for d in (oof_all, test)] + \
          [pl.read_parquet(out / f"train_{h}.parquet").select(pl.col("s1").explode()) for h in "AB"]
    need = pl.concat([x.rename({"s1": "entity_id"}) for x in ids]
                     + [d.select(entity_id="cand") for d in (oof_all, test)]
                     + [pl.read_parquet(out / f"train_{h}.parquet").select(entity_id="cand") for h in "AB"]).unique()
    rec = pl.scan_parquet(R / a.records).select("entity_id", "split", "country", "business_name", "business_address") \
            .join(need.lazy(), on="entity_id", how="semi").collect()
    assert rec.height == need.height, f"records missing: {need.height - rec.height}"
    rec.write_parquet(out / "records.parquet", compression="zstd")
    man["records"] = rec.height
    write_json(out / "manifest.json", man)
    print(f"records {rec.height:,}; bundle_sel done ({T()})")
    for f in sorted(out.glob("*.parquet")):
        print(f"  {f.name:20s} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
