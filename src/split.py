"""Phase 0 validation split.

Every fold is a self-contained mini version of the test problem:
  - S1 split 80/20, stratified by country (seeded).
  - Matched S2/S3 records follow their S1 (one-to-one holds in GT).
  - Unmatched S2/S3 distractors are split 80/20 at random, so the distractor ratio matches train.
Also writes a 'dev' subset (~50k S1 of val) for fast iteration.

Outputs cache/split/{trn,val,dev}_{source1,source2,source3}.parquet and {trn,val,dev}_ground_truth.tsv
"""
from pathlib import Path

import polars as pl

from io_utils import ROOT, gt_long, load

SEED = 2026
VAL_FRAC = 0.20
DEV_N = 50_000
OUT = ROOT / "cache" / "split"
OUT.mkdir(parents=True, exist_ok=True)

s1 = load("train", "source1")
s2 = load("train", "source2")
s3 = load("train", "source3")
gt = load("train", "ground_truth")
L = gt_long(gt)

# S1 fold assignment, stratified by country
s1 = s1.with_columns(
    r=pl.int_range(pl.len()).shuffle(seed=SEED).over("country") / pl.len().over("country"))
s1 = s1.with_columns(fold=pl.when(pl.col("r") < VAL_FRAC).then(pl.lit("val")).otherwise(pl.lit("trn"))).drop("r")

val_ids = s1.filter(pl.col("fold") == "val")["entity_id"]
dev_ids = (s1.filter(pl.col("fold") == "val")
             .with_columns(r=pl.int_range(pl.len()).shuffle(seed=SEED + 1).over("country") / pl.len().over("country"))
             .filter(pl.col("r") < DEV_N / val_ids.len())["entity_id"])

owner = L.join(s1.select("entity_id", "fold"), left_on="source1_entity_id", right_on="entity_id").select("match_id", "fold", "source1_entity_id")


def split_pool(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    df = df.join(owner, left_on="entity_id", right_on="match_id", how="left")
    rnd = pl.int_range(pl.len()).shuffle(seed=seed) / pl.len()
    return df.with_columns(fold=pl.when(pl.col("fold").is_null())
                           .then(pl.when(rnd < VAL_FRAC).then(pl.lit("val")).otherwise(pl.lit("trn")))
                           .otherwise(pl.col("fold")))


s2 = split_pool(s2, SEED + 2)
s3 = split_pool(s3, SEED + 3)
cols = ["entity_id", "business_name", "business_address", "country"]

for fold in ["trn", "val"]:
    s1.filter(pl.col("fold") == fold).select(cols).write_parquet(OUT / f"{fold}_source1.parquet")
    s2.filter(pl.col("fold") == fold).select(cols).write_parquet(OUT / f"{fold}_source2.parquet")
    s3.filter(pl.col("fold") == fold).select(cols).write_parquet(OUT / f"{fold}_source3.parquet")
    gt.join(s1.filter(pl.col("fold") == fold).select("entity_id"), left_on="source1_entity_id",
            right_on="entity_id", how="semi").write_csv(OUT / f"{fold}_ground_truth.tsv", separator="\t")

# dev = subset of val: its S1, their matches, plus a proportional share of val distractors
dev_frac = dev_ids.len() / val_ids.len()
for name, df, seed in [("source2", s2, SEED + 4), ("source3", s3, SEED + 5)]:
    v = df.filter(pl.col("fold") == "val")
    rnd = pl.int_range(pl.len()).shuffle(seed=seed) / pl.len()
    keep = v.filter(pl.col("source1_entity_id").is_in(dev_ids.implode())
                    | (pl.col("source1_entity_id").is_null() & (rnd < dev_frac)))
    keep.select(cols).write_parquet(OUT / f"dev_{name}.parquet")
s1.filter(pl.col("entity_id").is_in(dev_ids.implode())).select(cols).write_parquet(OUT / "dev_source1.parquet")
gt.filter(pl.col("source1_entity_id").is_in(dev_ids.implode())).write_csv(OUT / "dev_ground_truth.tsv", separator="\t")

for fold in ["trn", "val", "dev"]:
    n = {s: pl.scan_parquet(OUT / f"{fold}_{s}.parquet").select(pl.len()).collect().item() for s in ["source1", "source2", "source3"]}
    print(fold, n, "pool/S1 = %.2f" % ((n["source2"] + n["source3"]) / n["source1"]))

# "full" = the whole training set (used for the final models): links to the raw train tables + ground truth
for i in (1, 2, 3):
    load("train", f"source{i}")  # makes sure cache/raw/train_source{i}.parquet exists
for name, target in [(f"full_source{i}.parquet", Path("../raw") / f"train_source{i}.parquet") for i in (1, 2, 3)] + \
                    [("full_ground_truth.tsv", Path("../../data/dataset/train/train_ground_truth.tsv"))]:
    link = OUT / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)
print("full split: links to the complete training data")
