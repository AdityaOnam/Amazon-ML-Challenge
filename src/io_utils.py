"""Loading helpers: read the challenge TSVs once, cache as parquet."""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset"
RAW_CACHE = ROOT / "cache" / "raw"

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]


def _read_tsv(path: Path, cols) -> pl.DataFrame:
    # quote_char=None: names/addresses contain stray quotes; tabs are the only delimiter.
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    assert df.columns == cols, (path, df.columns)
    return df.with_columns(pl.col(c).fill_null("") for c in cols)


def load(split: str, name: str) -> pl.DataFrame:
    """split in {train,test}; name in {source1,source2,source3,ground_truth}."""
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    pq = RAW_CACHE / f"{split}_{name}.parquet"
    if pq.exists():
        return pl.read_parquet(pq)
    cols = GT_COLS if name == "ground_truth" else SOURCE_COLS
    df = _read_tsv(DATA / split / f"{split}_{name}.tsv", cols)
    df.write_parquet(pq)
    return df


def gt_long(gt: pl.DataFrame) -> pl.DataFrame:
    """One row per (s1, matched id); singletons dropped."""
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids")
              .rename({"matched_entity_ids": "match_id"})
              .filter(pl.col("match_id").str.len_chars() > 0))


def write_id_lists(s1_ids, lists, path, col):
    """Write a submission-style TSV. lists: iterable of lists of ids (may be empty).
    quote_style='never' is required: polars would otherwise write empty lists as "" and fail validation."""
    pl.DataFrame({"source1_entity_id": list(s1_ids), col: [",".join(x) for x in lists]}) \
      .write_csv(path, separator="\t", quote_style="never")
