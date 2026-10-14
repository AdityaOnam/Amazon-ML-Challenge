"""Phase 3a: pair features for every (s1, cand) candidate.

Usage: python features.py --fold val --cands cache/cand/val_cands_k20.parquet cache/cand/val_emb_Qwen3-Embedding-0.6B_k20.parquet
       [--select mix10,addr5,name5,emb5]  (keep only pairs picked by these blocker/k limits + exact keys)
Writes cache/feat/{fold}.parquet (+ label y when the fold has ground truth).
"""
import argparse
import os
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from io_utils import ROOT, gt_long

W = int(os.environ.get("SLURM_CPUS_PER_TASK", 64))
NCOLS = ["entity_id", "country", "name_clean", "name_core", "name_sorted", "name_nospace", "name_phon", "name_alias",
         "legal_form", "name_is_domain", "addr_clean", "addr_nums", "addr_hnum", "addr_tokens", "addr_phon",
         "state", "city"]
BLOCK_ALIASES = {"mix": "tf_mix", "addr": "tf_addr", "name": "tf_name", "emb": "emb"}


def load_cands(files, select):
    c = pl.read_parquet(files[0])
    for f in files[1:]:
        c = c.join(pl.read_parquet(f), on=["s1", "cand"], how="full", coalesce=True)
    if select:
        keep = pl.col("x_hnumcity").fill_null(False) | pl.col("x_nospace").fill_null(False)
        for part in select.split(","):
            b, k = part.rstrip("0123456789"), int(part[len(part.rstrip("0123456789")):])
            col = f"r_{BLOCK_ALIASES[b]}"
            if col in c.columns:
                keep = keep | (pl.col(col).fill_null(10_000) <= k)
        c = c.filter(keep)
    return c


def records(fold, norm="trn"):
    nd = ROOT / "cache" / f"norm_{norm}"
    split = "test" if fold == "test" else "train"
    s1 = pl.read_parquet(nd / f"{split}_source1.parquet", columns=NCOLS)
    s23 = pl.concat([pl.read_parquet(nd / f"{split}_source{i}.parquet", columns=NCOLS) for i in (2, 3)])
    return s1, s23


def pairwise(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=W, dtype=np.float32, **kw)


def set_sim(df, a, b, name):
    """Jaccard / containment of space-separated token sets (vectorised in polars)."""
    sa = pl.col(a).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    sb = pl.col(b).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    inter = sa.list.set_intersection(sb).list.len()
    union = sa.list.set_union(sb).list.len()
    return df.with_columns(
        (inter / pl.when(union > 0).then(union).otherwise(None)).fill_null(0).cast(pl.Float32).alias(f"{name}_jac"),
        (inter / pl.when(sa.list.len() > 0).then(sa.list.len()).otherwise(None)).fill_null(0).cast(pl.Float32)
        .alias(f"{name}_cov1"),
        inter.cast(pl.Int16).alias(f"{name}_inter"),
    )


def initials(col):
    return pl.col(col).str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("")


def uniqueness(s1: pl.DataFrame, s23: pl.DataFrame):
    """How common a name is: same-name counts among S1 / pool records, and pool doc-freq of the rarest core token."""
    f_s1 = s1.group_by("country", "name_sorted").agg(nf_s1=pl.len().cast(pl.Int32))
    f_pool = s23.group_by("country", "name_sorted").agg(nf_pool=pl.len().cast(pl.Int32))
    toks = (s23.select("country", tok=pl.col("name_core").str.split(" ").list.unique()).explode("tok")
               .filter(pl.col("tok") != "").group_by("country", "tok").agg(df=pl.len()))
    n_pool = s23.group_by("country").agg(npool=pl.len())

    def token_stats(df, name_col, prefix):
        e = (df.select("entity_id", "country", tok=pl.col(name_col).str.split(" ").list.unique()).explode("tok")
               .join(toks, on=["country", "tok"], how="left").with_columns(pl.col("df").fill_null(0))
               .join(n_pool, on="country", how="left")
               .with_columns(idf=((pl.col("npool") + 1) / (pl.col("df") + 1)).log()))
        return e.group_by("entity_id").agg(pl.col("df").min().cast(pl.Int32).alias(f"{prefix}_min_df"),
                                           pl.col("idf").sum().cast(pl.Float32).alias(f"{prefix}_idf_sum"))
    s1 = (s1.join(f_s1, on=["country", "name_sorted"], how="left").join(f_pool, on=["country", "name_sorted"], how="left")
            .join(token_stats(s1, "name_core", "u1"), on="entity_id", how="left"))
    s23 = (s23.join(f_pool.rename({"nf_pool": "nf_pool_c"}), on=["country", "name_sorted"], how="left")
              .join(token_stats(s23, "name_core", "u2"), on="entity_id", how="left"))
    return s1, s23


def build(c: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame) -> pl.DataFrame:
    t = time.time()
    s1, s23 = uniqueness(s1, s23)
    print(f"  uniqueness stats ({time.time() - t:.0f}s)", flush=True)
    d = (c.join(s1, left_on="s1", right_on="entity_id", how="left")
          .join(s23, left_on="cand", right_on="entity_id", how="left", suffix="_m"))
    print(f"  joined {d.height:,} pairs ({time.time() - t:.0f}s)", flush=True)

    L = lambda col: d[col].to_list()  # noqa: E731
    f = {}
    # ---- name string similarities
    f["n_ratio"] = pairwise(L("name_core"), L("name_core_m"), fuzz.ratio)
    f["n_tsort"] = pairwise(L("name_core"), L("name_core_m"), fuzz.token_sort_ratio)
    f["n_tset"] = pairwise(L("name_core"), L("name_core_m"), fuzz.token_set_ratio)
    f["n_partial"] = pairwise(L("name_core"), L("name_core_m"), fuzz.partial_ratio)
    f["n_full_tset"] = pairwise(L("name_clean"), L("name_clean_m"), fuzz.token_set_ratio)
    f["n_nospace_jw"] = pairwise(L("name_nospace"), L("name_nospace_m"), JaroWinkler.normalized_similarity)
    f["n_nospace_lev"] = pairwise(L("name_nospace"), L("name_nospace_m"), Levenshtein.normalized_similarity)
    f["n_phon_tset"] = pairwise(L("name_phon"), L("name_phon_m"), fuzz.token_set_ratio)
    f["n_alias_tset"] = pairwise(L("name_core"), L("name_alias_m"), fuzz.token_set_ratio)
    # ---- address string similarities
    f["a_tset"] = pairwise(L("addr_tokens"), L("addr_tokens_m"), fuzz.token_set_ratio)
    f["a_ratio"] = pairwise(L("addr_clean"), L("addr_clean_m"), fuzz.token_sort_ratio)
    f["a_partial"] = pairwise(L("addr_clean"), L("addr_clean_m"), fuzz.partial_token_set_ratio)
    f["a_phon_tset"] = pairwise(L("addr_phon"), L("addr_phon_m"), fuzz.token_set_ratio)
    f["hnum_lev"] = pairwise(L("addr_hnum"), L("addr_hnum_m"), Levenshtein.normalized_similarity)
    f["city_ratio"] = pairwise(L("city"), L("city_m"), fuzz.ratio)
    f["city_in_addr"] = pairwise(L("city"), L("addr_clean_m"), fuzz.partial_ratio)
    d = d.with_columns(pl.Series(k, v) for k, v in f.items())
    print(f"  string sims ({time.time() - t:.0f}s)", flush=True)

    d = set_sim(d, "name_core", "name_core_m", "n_tok")
    d = set_sim(d, "addr_tokens", "addr_tokens_m", "a_tok")
    d = set_sim(d, "addr_nums", "addr_nums_m", "a_num")
    e = lambda a, b: ((pl.col(a) != "") & (pl.col(a) == pl.col(b)))  # noqa: E731
    miss = lambda a: (pl.col(a) == "")  # noqa: E731
    d = d.with_columns(
        n_core_eq=pl.col("name_core") == pl.col("name_core_m"),
        n_sorted_eq=pl.col("name_sorted") == pl.col("name_sorted_m"),
        n_nospace_eq=e("name_nospace", "name_nospace_m"),
        n_phon_eq=e("name_phon", "name_phon_m"),
        n_acronym=((initials("name_core") == pl.col("name_core_m")) & (pl.col("name_core_m").str.len_chars() >= 2))
                  | ((initials("name_core_m") == pl.col("name_core")) & (pl.col("name_core").str.len_chars() >= 2)),
        n_len1=pl.col("name_core").str.len_chars().cast(pl.Int16),
        n_len2=pl.col("name_core_m").str.len_chars().cast(pl.Int16),
        n_ntok2=pl.col("name_core_m").str.count_matches(" ").cast(pl.Int16) + 1,
        n_is_domain=pl.col("name_is_domain_m"),
        n_has_alias=pl.col("name_alias_m") != "",
        # legal form: 0 both empty, 1 equal, 2 one missing, 3 overlap, 4 conflict
        legal=pl.when(miss("legal_form") & miss("legal_form_m")).then(0)
               .when(pl.col("legal_form") == pl.col("legal_form_m")).then(1)
               .when(miss("legal_form") | miss("legal_form_m")).then(2)
               .when(pl.col("legal_form").str.split(" ").list.set_intersection(
                   pl.col("legal_form_m").str.split(" ")).list.len() > 0).then(3)
               .otherwise(4).cast(pl.Int8),
        # house number: 0 both missing, 1 equal, 2 one missing, 3 different; also digit-prefix agreement
        hnum=pl.when(miss("addr_hnum") & miss("addr_hnum_m")).then(0).when(e("addr_hnum", "addr_hnum_m")).then(1)
              .when(miss("addr_hnum") | miss("addr_hnum_m")).then(2).otherwise(3).cast(pl.Int8),
        hnum_digits_eq=(pl.col("addr_hnum").str.extract(r"(\d+)") == pl.col("addr_hnum_m").str.extract(r"(\d+)"))
                       .fill_null(False),
        hnum_contains=((pl.col("addr_hnum") != "") & (pl.col("addr_hnum_m") != "")
                       & (pl.col("addr_hnum").str.contains(pl.col("addr_hnum_m"), literal=True)
                          | pl.col("addr_hnum_m").str.contains(pl.col("addr_hnum"), literal=True))),
        hnum_in_nums=((pl.col("addr_hnum") != "")
                      & pl.col("addr_nums_m").str.split(" ").list.contains(pl.col("addr_hnum"))),
        state=pl.when(miss("state") | miss("state_m")).then(0).when(e("state", "state_m")).then(1)
               .otherwise(2).cast(pl.Int8),
        city_eq=e("city", "city_m"),
        addr_empty2=miss("addr_clean_m"),
        addr_len2=pl.col("addr_clean_m").str.len_chars().cast(pl.Int16),
        src3=pl.col("cand").str.starts_with("S3"),
    )
    # ---- blocker scores / ranks (missing = not retrieved by that blocker)
    for b in ["tf_name", "tf_addr", "tf_mix", "emb"]:
        if f"s_{b}" in d.columns:
            d = d.with_columns(pl.col(f"s_{b}").fill_null(0.0), pl.col(f"r_{b}").fill_null(99).cast(pl.Int16))
    for x in ["x_hnumcity", "x_nospace"]:
        if x in d.columns:
            d = d.with_columns(pl.col(x).fill_null(False))
    # ---- context: competition within the S1's list and for the candidate across S1s
    key = "s_tf_mix"
    d = d.with_columns(
        n_cands=pl.len().over("s1").cast(pl.Int32),
        n_cands_src=pl.len().over("s1", "src3").cast(pl.Int32),
        mix_rank_src=pl.col(key).rank("ordinal", descending=True).over("s1", "src3").cast(pl.Int16),
        mix_gap_top=(pl.col(key).max().over("s1", "src3") - pl.col(key)).cast(pl.Float32),
        cand_n_s1=pl.len().over("cand").cast(pl.Int16),
        cand_best_other=pl.col(key).max().over("cand").cast(pl.Float32),
    )
    d = d.with_columns(cand_is_best=(pl.col(key) >= pl.col("cand_best_other")),
                       cand_margin=(pl.col(key) - pl.col(key).filter(pl.col(key) < pl.col(key).max()).max()
                                    .over("cand")).fill_null(1.0).cast(pl.Float32))
    for col in ["nf_s1", "nf_pool", "nf_pool_c", "u1_min_df", "u2_min_df"]:
        d = d.with_columns(pl.col(col).fill_null(0))
    d = d.with_columns(pl.col("u1_idf_sum").fill_null(0.0), pl.col("u2_idf_sum").fill_null(0.0))
    drop = [c for c in d.columns if c in NCOLS or c.endswith("_m") and c[:-2] in NCOLS]
    print(f"  done ({time.time() - t:.0f}s)", flush=True)
    return d.drop(drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="val")
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--select", default="mix10,addr5,name5,emb5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--norm", default="trn", help="normalised-record set: cache/norm_<norm>/")
    ap.add_argument("--countries", nargs="*", default=None, help="keep only pairs whose S1 is in these countries")
    a = ap.parse_args()
    t = time.time()
    c = load_cands([ROOT / x if not x.startswith("/") else x for x in a.cands], a.select)
    print(f"{a.fold}: {c.height:,} candidate pairs after selection", flush=True)
    s1, s23 = records(a.fold, a.norm)
    if a.countries:
        s1, s23 = s1.filter(pl.col("country").is_in(a.countries)), s23.filter(pl.col("country").is_in(a.countries))
        c = c.join(s1.select(s1="entity_id"), on="s1", how="semi")
        print(f"  {a.countries}: {c.height:,} pairs", flush=True)
    d = build(c, s1, s23)
    if a.fold != "test":
        gt = pl.read_csv(ROOT / "cache/split" / f"{a.fold}_ground_truth.tsv", separator="\t", infer_schema=False).fill_null("")
        L = gt_long(gt).select(s1="source1_entity_id", cand="match_id").with_columns(y=pl.lit(True))
        d = d.join(L, on=["s1", "cand"], how="left").with_columns(pl.col("y").fill_null(False))
        print(f"  positives {d['y'].sum():,} of {L.height:,} true pairs "
              f"(blocking recall {d['y'].sum() / L.height * 100:.2f}%)")
    out = ROOT / "cache" / "feat"
    out.mkdir(parents=True, exist_ok=True)
    path = out / (a.out or f"{a.fold}.parquet")
    d.write_parquet(path)
    print(f"wrote {path}: {d.height:,} rows x {d.width} cols in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
