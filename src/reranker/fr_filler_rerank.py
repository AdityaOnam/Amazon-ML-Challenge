"""France filler fix for the cross-encoders (final version F0).

The rerankers read the RAW business name. 14% of French records carry generator-injected filler words
(text_tables.FR_INJECTED: "france" e.g. "(France)", "developpement", "international"; the same list whose removal
from the name cores was the confirmed v6 fix). The rerankers were trained on US/India only and read these words as
evidence of a different business: at stage-1 p >= 0.99 a pair with a filler on one side only gets a mean bge logit
of 0.31 vs 5.79, and the stack accepts 59% vs 93% of such pairs.

  bundle : strip the filler words from the raw names of the French records (never emptying a name) and write
           bundle_frfil/{fl_all, fl_all_swap, records}.parquet = every France band pair (stage-1 p in [0.001, 0.999])
           that already has a bge_local score and whose text changed.
  merge  : after score_reranker.py has scored fl_all / fl_all_swap with ckpt/rr_bge_local (normal + swapped order),
           replace those pairs' scores in results/rr_L_ABE.parquet / rr_S_ABE.parquet
           -> results/rr_L_ABEF0.parquet / rr_S_ABEF0.parquet (all other pairs unchanged).
Run from src/reranker:  python fr_filler_rerank.py bundle  |  python fr_filler_rerank.py merge
"""
import re
import sys
import unicodedata
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text_tables import FR_INJECTED  # noqa: E402

B = HERE / "bundle_frfil"


def fold(t: str) -> str:
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()


def strip(name):
    if not name:
        return name
    out = []
    for t in re.split(r"(\s+)", name):
        core = re.sub(r"^[\(\[\-,]+|[\)\]\-,]+$", "", t)
        if core and fold(core) in FR_INJECTED:
            continue
        out.append(t)
    s = re.sub(r"\(\s*\)", "", "".join(out))
    s = re.sub(r"\s+", " ", s).strip(" ,-")
    return s or name  # never empty a name


def cmd_bundle():
    B.mkdir(exist_ok=True)
    fr = (pl.read_parquet(ROOT / "cache/raw/test_source1.parquet", columns=["entity_id", "country"])
            .filter(pl.col("country") == "France").select(s1="entity_id"))
    st = pl.read_parquet(HERE / "work/test_stacked_v3b_ABES2_fr2.parquet", columns=["s1", "cand", "p_lgb"])
    band = (st.join(fr, on="s1", how="semi").filter((pl.col("p_lgb") >= 0.001) & (pl.col("p_lgb") <= 0.999))
              .select("s1", "cand")
              .join(pl.read_parquet(HERE / "results/rr_L_ABE.parquet", columns=["s1", "cand"]), on=["s1", "cand"],
                    how="semi"))
    need = pl.concat([band.select(entity_id="s1"), band.select(entity_id="cand")]).unique()
    rec = pl.scan_parquet(HERE / "bundle/records.parquet").join(need.lazy(), on="entity_id", how="semi").collect()
    miss = need.join(rec, on="entity_id", how="anti")
    if miss.height:  # rebuild exactly as export_bundle.py
        recs = [pl.read_parquet(ROOT / f"cache/norm_trn/test_source{i}.parquet",
                                columns=["entity_id", "country", "business_name", "business_address", "name_clean",
                                         "addr_clean"]).with_columns(split=pl.lit("test"), src=pl.lit(i, pl.UInt8))
                for i in (1, 2, 3)]
        rec = pl.concat([rec, pl.concat(recs).join(miss, on="entity_id", how="semi").select(rec.columns)])
    rec = rec.unique("entity_id")
    new = rec["business_name"].map_elements(strip, return_dtype=pl.String)
    rec = rec.with_columns(changed=(new != pl.col("business_name")), business_name=new)
    ch = rec.filter(pl.col("changed")).select("entity_id")
    pairs = pl.concat([band.join(ch.rename({"entity_id": "s1"}), on="s1", how="semi"),
                       band.join(ch.rename({"entity_id": "cand"}), on="cand", how="semi")]).unique()
    pairs.write_parquet(B / "fl_all.parquet")
    pairs.select(s1="cand", cand="s1").write_parquet(B / "fl_all_swap.parquet")
    rec.drop("changed").write_parquet(B / "records.parquet", compression="zstd")
    print(f"records with a filler removed: {ch.height:,}; France band pairs to re-score: {pairs.height:,} "
          f"of {band.height:,}")


def cmd_merge():
    R = B / "results"
    new = {"L": pl.read_parquet(R / "rr_bge_local_fl_all.parquet").select("s1", "cand", new="score"),
           "S": pl.read_parquet(R / "rr_bge_local_fl_all_swap.parquet").select(s1="cand", cand="s1", new="score")}
    for k, n in new.items():
        b = pl.read_parquet(HERE / f"results/rr_{k}_ABE.parquet")
        o = (b.join(n, on=["s1", "cand"], how="left")
              .with_columns(score=pl.coalesce("new", "score").cast(b["score"].dtype)).drop("new"))
        o.write_parquet(HERE / f"results/rr_{k}_ABEF0.parquet")
        print(f"rr_{k}_ABEF0: {n.join(b, on=['s1', 'cand'], how='semi').height:,} France pairs re-scored "
              f"of {o.height:,}")


if __name__ == "__main__":
    {"bundle": cmd_bundle, "merge": cmd_merge}[sys.argv[1]]()
