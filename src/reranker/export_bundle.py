"""Export the compact data bundle the GPU server needs (runs on the HOME server, CPU only, reads cache/ read-only).

Honesty scheme (see 06_gpu_experiments.md section 2):
  H_A = S1s in the LightGBM OOF sample (cache/oof/<oof>_s1.parquet, 50%): have honest OOF p.
  H_B = the other S1s in the feature file: scored here by the <models>_fold* LightGBM boosters,
        which were trained on H_A only, so p_B is out-of-sample too.
  Reranker / embedder TRAIN on H_B; they SCORE H_A (stacker training + honest eval) and test.

Outputs in --out (default research/gpu_exp/bundle/):
  records.parquet     entity_id, split, src, country, business_name, business_address, name_clean, addr_clean
  rr_train.parquet    s1, cand, y, p, kind      (H_B: hard band + sampled easy pairs)       -> exp A training
  rr_stack.parquet    s1, cand, y, p, is_val    (H_A: OOF p in stack band)                   -> exp A scoring
  rr_test.parquet     s1, cand, p               (test: p in test band)                       -> exp A scoring
  emb_train.parquet   s1, pos, negs(list)       (H_B: GT positives + hardest candidate negs) -> exp B training
  emb_eval_s1.parquet s1, country; emb_eval_gt.parquet s1, match; emb_eval_cands.parquet s1, cand, src
  manifest.json
Also (NOT in the bundle, stays home): work/test_p_<models>.parquet = s1, cand, p for ALL test pairs.

Smoke:  python export_bundle.py --sample 0.0005 --out research/gpu_exp/smoke/bundle
Full:   python export_bundle.py                           (about 30-60 min CPU, ~2-4 GB output)
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, Timer, write_json  # noqa: E402


def gt_pairs(root: Path) -> pl.DataFrame:
    gt = pl.read_csv(root / "cache/split/full_ground_truth.tsv", separator="\t", infer_schema=False).fill_null("")
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids").str.len_chars() > 0)
              .select(s1="source1_entity_id", match="matched_entity_ids"))


def sample_expr(frac: float, col="s1"):
    return (pl.col(col).hash(seed=123) % 1_000_000) < int(frac * 1_000_000)


def predict_lgb(lf: pl.LazyFrame, boosters, chunk=4_000_000, keep=("s1", "cand", "y")) -> pl.DataFrame:
    """Average fold-booster probabilities over a (lazy) feature table, chunked to bound memory."""
    cols = boosters[0].feature_name()
    d = lf.collect()
    keep = [k for k in keep if k in d.columns]
    out = []
    for s in range(0, d.height, chunk):
        part = d.slice(s, chunk)
        X = part.select(pl.col(c).cast(pl.Float32) for c in cols).to_numpy()
        p = np.mean([b.predict(X, num_threads=int(__import__("os").environ.get("LGB_THREADS", 32))) for b in boosters], axis=0).astype(np.float32)
        out.append(part.select(keep).with_columns(p=pl.Series(p)))
    return pl.concat(out) if out else d.select(keep).with_columns(p=pl.lit(0.0, pl.Float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[1]), help="LLM_based_work directory")
    ap.add_argument("--out", default=str(HERE / "bundle"))
    ap.add_argument("--feat", default="cache/feat/full_v3.parquet")
    ap.add_argument("--oof", default="cache/oof/full_v3.parquet")
    ap.add_argument("--models", default="full_v3", help="runs/models/<prefix>_fold*.txt (trained on H_A)")
    ap.add_argument("--test-feat", default="cache/feat/test.parquet")
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of S1s (train and test) to export; <1 = smoke")
    ap.add_argument("--train-band", type=float, nargs=2, default=[0.02, 0.98])
    ap.add_argument("--stack-band", type=float, nargs=2, default=[0.02, 0.98])
    ap.add_argument("--test-band", type=float, nargs=2, default=[0.02, 0.98])
    ap.add_argument("--easy-neg-ratio", type=float, default=0.5, help="easy negatives per hard pair")
    ap.add_argument("--easy-pos-ratio", type=float, default=0.15, help="easy positives per hard pair")
    ap.add_argument("--emb-anchors", type=int, default=1_500_000)
    ap.add_argument("--emb-hardneg", type=int, default=3)
    ap.add_argument("--eval-s1", type=int, default=50_000)
    ap.add_argument("--skip-test", action="store_true")
    a = ap.parse_args()
    R, out, T = Path(a.root), Path(a.out), Timer()
    out.mkdir(parents=True, exist_ok=True)
    work = HERE / "work"
    work.mkdir(exist_ok=True)
    smoke = a.sample < 1.0
    rng = np.random.default_rng(0)
    man = {"args": vars(a)}

    boosters = [lgb.Booster(model_file=str(f)) for f in sorted((R / "runs/models").glob(f"{a.models}_fold*.txt"))]
    assert boosters, f"no boosters {a.models}"
    HA = pl.read_parquet(R / a.oof.replace(".parquet", "_s1.parquet"))  # column s1
    val = pl.read_parquet(R / "cache/split/val_source1.parquet", columns=["entity_id"]).rename({"entity_id": "s1"})
    gtp = gt_pairs(R)

    # ---------------- H_A: OOF pairs in the stack band
    oof = pl.scan_parquet(R / a.oof)
    if smoke:
        oof = oof.filter(sample_expr(a.sample))
    lo, hi = a.stack_band
    oof = oof.collect()
    man["oof_pairs"] = oof.height
    for b in ((0.02, 0.98), (0.05, 0.95), (0.1, 0.9)):
        man[f"oof_frac_{b[0]}_{b[1]}"] = float(oof["p"].is_between(*b).mean())
    stack = oof.filter(pl.col("p").is_between(lo, hi)).join(val.with_columns(is_val=pl.lit(True)), on="s1", how="left") \
               .with_columns(pl.col("is_val").fill_null(False))
    stack.write_parquet(out / "rr_stack.parquet")
    man["rr_stack"] = stack.height
    print(f"rr_stack {stack.height:,} ({T()})", flush=True)

    # ---------------- H_B: score with the H_A-trained boosters, then select training pairs
    fb = pl.scan_parquet(R / a.feat).join(HA.lazy(), on="s1", how="anti")
    if smoke:
        fb = fb.filter(sample_expr(a.sample))
    pb = predict_lgb(fb, boosters)
    man["hb_pairs"] = pb.height
    print(f"H_B scored {pb.height:,} pairs ({T()})", flush=True)
    stack_cands = stack.select("cand").unique()
    pb_tr = pb.join(stack_cands, on="cand", how="anti")  # no record seen in the stack set is trained on
    lo, hi = a.train_band
    hard = pb_tr.filter(pl.col("p").is_between(lo, hi)).with_columns(kind=pl.lit("hard"))
    eneg = pb_tr.filter((pl.col("p") < lo) & ~pl.col("y"))
    epos = pb_tr.filter((pl.col("p") > hi) & pl.col("y"))
    n_en, n_ep = int(a.easy_neg_ratio * hard.height), int(a.easy_pos_ratio * hard.height)
    eneg = eneg.sample(min(n_en, eneg.height), seed=1).with_columns(kind=pl.lit("easy_neg"))
    epos = epos.sample(min(n_ep, epos.height), seed=2).with_columns(kind=pl.lit("easy_pos"))
    rr_train = pl.concat([hard, eneg, epos]).sample(fraction=1.0, shuffle=True, seed=3)
    rr_train.write_parquet(out / "rr_train.parquet")
    man["rr_train"] = {"total": rr_train.height, "hard": hard.height, "easy_neg": eneg.height,
                       "easy_pos": epos.height, "pos_rate": float(rr_train["y"].mean())}
    print(f"rr_train {man['rr_train']} ({T()})", flush=True)

    # ---------------- exp B training: GT positives of H_B S1s + hardest candidate negatives
    hb_s1 = pb.select("s1").unique()
    pos = gtp.join(hb_s1, on="s1", how="semi").join(stack_cands.rename({"cand": "match"}), on="match", how="anti")
    negs = (pb.filter(~pl.col("y")).sort(["s1", "p"], descending=[False, True])
              .group_by("s1", maintain_order=True).agg(pl.col("cand").head(a.emb_hardneg).alias("negs")))
    emb = pos.rename({"match": "pos"}).join(negs, on="s1", how="left")
    if emb.height > a.emb_anchors:
        emb = emb.sample(a.emb_anchors, seed=4)
    emb.write_parquet(out / "emb_train.parquet")
    man["emb_train"] = emb.height
    del pb, pb_tr

    # ---------------- exp B eval: honest S1s (H_A and val fold), their GT, and the current candidate sets
    ev = HA.join(val, on="s1", how="semi")
    if smoke:
        ev = ev.filter(sample_expr(a.sample * 20))
    ev = ev.sample(min(a.eval_s1, ev.height), seed=5)
    s1rec = pl.read_parquet(R / "cache/norm_trn/train_source1.parquet", columns=["entity_id", "country"])
    ev = ev.join(s1rec.rename({"entity_id": "s1"}), on="s1", how="left")
    ev.write_parquet(out / "emb_eval_s1.parquet")
    gtp.join(ev.select("s1"), on="s1", how="semi").write_parquet(out / "emb_eval_gt.parquet")
    cur = pl.scan_parquet(R / a.feat).select("s1", "cand").join(ev.lazy().select("s1"), on="s1", how="semi") \
            .collect().with_columns(src=pl.lit("final"))
    base = pl.scan_parquet(R / "cache/cand/full_emb_Qwen3-Embedding-0.6B_k20.parquet").select("s1", "cand") \
             .join(ev.lazy().select("s1"), on="s1", how="semi").collect().with_columns(src=pl.lit("emb_base_k20"))
    pl.concat([cur, base]).write_parquet(out / "emb_eval_cands.parquet")
    man["emb_eval_s1"] = ev.height
    print(f"emb_train {emb.height:,}, emb_eval {ev.height:,} ({T()})", flush=True)

    # ---------------- test: p for all pairs (kept home) and the band (bundle)
    if not a.skip_test:
        tf = pl.scan_parquet(R / a.test_feat)
        if smoke:
            tf = tf.filter(sample_expr(a.sample))
        tp = predict_lgb(tf, boosters, keep=("s1", "cand"))
        tp.write_parquet(work / f"test_p_{a.models}{'_smoke' if smoke else ''}.parquet")
        lo, hi = a.test_band
        tb = tp.filter(pl.col("p").is_between(lo, hi))
        tb.write_parquet(out / "rr_test.parquet")
        man["test_pairs"], man["rr_test"] = tp.height, tb.height
        for b in ((0.02, 0.98), (0.05, 0.95), (0.1, 0.9)):
            man[f"test_frac_{b[0]}_{b[1]}"] = float(tp["p"].is_between(*b).mean())
        print(f"rr_test {tb.height:,} of {tp.height:,} ({T()})", flush=True)

    # ---------------- records: all train/test records (full) or only the referenced ones + a pool sample (smoke)
    recs = []
    for split in ("train", "test"):
        for i in (1, 2, 3):
            r = pl.read_parquet(R / f"cache/norm_trn/{split}_source{i}.parquet",
                                columns=["entity_id", "country", "business_name", "business_address",
                                         "name_clean", "addr_clean"])
            recs.append(r.with_columns(split=pl.lit(split), src=pl.lit(i, pl.UInt8)))
    rec = pl.concat(recs)
    if smoke:
        ids = [pl.read_parquet(out / f, columns=c) for f, c in
               [("rr_train.parquet", ["s1", "cand"]), ("rr_stack.parquet", ["s1", "cand"]),
                ("emb_eval_cands.parquet", ["s1", "cand"])]]
        if (out / "rr_test.parquet").exists():
            ids.append(pl.read_parquet(out / "rr_test.parquet", columns=["s1", "cand"]))
        e = pl.read_parquet(out / "emb_train.parquet")
        need = pl.concat([pl.concat([d["s1"], d["cand"]]) for d in ids]
                         + [e["s1"], e["pos"], e["negs"].explode().drop_nulls(),
                            pl.read_parquet(out / "emb_eval_gt.parquet")["match"]]).unique()
        pool = rec.filter((pl.col("split") == "train") & (pl.col("src") > 1)).sample(40_000, seed=6)["entity_id"]
        rec = rec.filter(pl.col("entity_id").is_in(pl.concat([need, pool]).unique().implode()))
    rec.write_parquet(out / "records.parquet", compression="zstd")
    man["records"] = rec.height
    write_json(out / "manifest.json", man)
    print(f"records {rec.height:,}; bundle done ({T()})")
    for f in sorted(out.glob("*.parquet")):
        print(f"  {f.name:24s} {f.stat().st_size / 1e6:9.1f} MB")


if __name__ == "__main__":
    main()
