"""Shared helpers for research/12_shortlist_round3.md. Read-only on cache/, src/, runs/, research/gpu_exp/.
Writes only src/s12_cache/ (package layout: this folder is src/routes). Threads capped at 32."""
import os

from pathlib import Path as _P
os.environ.setdefault("POLARS_MAX_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("TMPDIR", str(_P(__file__).resolve().parents[2] / ".tmp"))

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402,F401
import polars as pl  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decide import macro_f05, one_to_one  # noqa: E402,F401
from io_utils import gt_long  # noqa: E402

SC = HERE.parent / "s12_cache"
SC.mkdir(parents=True, exist_ok=True)
C = ROOT / "cache"
GX = ROOT / "research/gpu_exp"
STACKED = GX / "work/oof_stacked_v3b_ABE.parquet"
FEAT = C / "feat/full_v3.parquet"
CANDS = C / "cand/full_cands_k10.parquet"
EMB = C / "cand/full_emb_Qwen3-Embedding-0.6B_k20.parquet"
NORM = C / "norm_trn"
THR = 0.75
T0 = time.time()


def T():
    return f"{time.time() - T0:.0f}s"


def honest_universe() -> pl.DataFrame:
    u = pl.read_parquet(C / "oof/full_v3_s1.parquet")
    v = pl.read_parquet(C / "split/val_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    return u.join(v, on="s1", how="inner")


def gt_all() -> pl.DataFrame:
    gt = pl.read_csv(C / "split/full_ground_truth.tsv", separator="\t", infer_schema=False).fill_null("")
    return gt_long(gt).select(s1="source1_entity_id", cand="match_id")


def score(s: pl.DataFrame, gth, uni, thr=THR, col="p"):
    """stage2.honest protocol: restrict to universe, one_to_one, threshold."""
    s11 = one_to_one(s.join(uni.select("s1"), on="s1", how="semi").select("s1", "cand", p=col))
    r = macro_f05(s11.filter(pl.col("p") >= thr).select("s1", "cand"), gth, uni)
    by = {c: round(v, 5) for c, v in r.group_by("country").agg(pl.col("f").mean()).sort("country").iter_rows()}
    return round(r["f"].mean(), 5), by


def pool_pairs() -> pl.LazyFrame:
    """Raw candidate pool = tf k10 (name/addr/mix) + exact keys + emb k20 (= v4 selection, in production)."""
    return pl.concat([pl.scan_parquet(CANDS).select("s1", "cand"), pl.scan_parquet(EMB).select("s1", "cand")]).unique()
