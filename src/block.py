"""Phase 2: candidate generation (blocking). Union of cheap blockers, all within the same country label.

Blockers (each run separately against the S2 pool and the S3 pool, top-k per S1):
  tf_name     char 3-gram TF-IDF on name_core
  tf_addr     char 3-gram TF-IDF on address tokens + numbers   (catches invented / replaced names)
  tf_mix      word TF-IDF on name tokens + address tokens + numbers
  ex_hnumcity exact house number + city            (blocks capped at MAX_BLOCK)
  ex_nospace  exact name_nospace                    (domain-style names)
Output: cache/cand/{fold}_cands.parquet with one row per (s1, cand) and each blocker's score/rank.

Usage: python block.py --fold dev|val|test [--k 30] [--norm trn]
"""
import argparse
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from io_utils import ROOT

MAX_BLOCK = 50
N_THREADS = int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", 64))
COLS = ["entity_id", "country", "name_core", "name_nospace", "addr_tokens", "addr_nums", "addr_hnum", "city"]


def load_fold(fold: str, norm: str):
    nd = ROOT / "cache" / f"norm_{norm}"
    if fold == "test":
        return [pl.read_parquet(nd / f"test_source{i}.parquet", columns=COLS) for i in (1, 2, 3)]
    out = []
    for i in (1, 2, 3):
        ids = pl.read_parquet(ROOT / "cache/split" / f"{fold}_source{i}.parquet", columns=["entity_id"])
        out.append(pl.read_parquet(nd / f"train_source{i}.parquet", columns=COLS).join(ids, on="entity_id", how="semi"))
    return out


def texts(df: pl.DataFrame):
    return {
        "tf_name": df["name_core"].to_list(),
        "tf_addr": (df["addr_tokens"] + " " + df["addr_nums"]).to_list(),
        "tf_mix": (df["name_core"] + " " + df["addr_tokens"] + " "
                   + df["addr_nums"].str.replace_all(r"(\S+)", "n$1")).to_list(),
    }


VECT = {
    "tf_name": dict(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True, dtype=np.float32),
    "tf_addr": dict(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True, dtype=np.float32),
    "tf_mix": dict(analyzer="word", token_pattern=r"\S+", min_df=2, sublinear_tf=True, dtype=np.float32),
}


def topk(A, B, k, q_ids, p_ids, name, chunk=200_000):
    """Top-k cosine neighbours of each row of A among rows of B (both L2-normalised)."""
    BT = B.T.tocsr()
    rows, cols, vals = [], [], []
    for s in range(0, A.shape[0], chunk):
        C = sp_matmul_topn(A[s:s + chunk], BT, top_n=k, threshold=0.05, n_threads=N_THREADS, sort=True)
        C = C.tocoo()
        rows.append(C.row + s)
        cols.append(C.col)
        vals.append(C.data)
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    df = pl.DataFrame({"s1": q_ids[r], "cand": p_ids[c], f"s_{name}": v.astype(np.float32)})
    return df.with_columns(pl.col(f"s_{name}").rank("ordinal", descending=True).over("s1").cast(pl.UInt16)
                           .alias(f"r_{name}"))


def exact(q: pl.DataFrame, p: pl.DataFrame, keys, name):
    qk = q.filter(pl.all_horizontal(pl.col(k) != "" for k in keys)).select("entity_id", *keys)
    pk = p.filter(pl.all_horizontal(pl.col(k) != "" for k in keys)).select("entity_id", *keys)
    pk = pk.filter(pl.len().over(keys) <= MAX_BLOCK)
    return (qk.join(pk, on=keys, suffix="_p").select(s1="entity_id", cand="entity_id_p")
              .with_columns(pl.lit(True).alias(f"x_{name}")))


def block_country(q: pl.DataFrame, pools, k):
    parts = []
    tq = texts(q)
    tp = [texts(p) for p in pools]
    for name, params in VECT.items():
        vec = TfidfVectorizer(**params)
        # fit IDF on queries + a sample of each pool (bounded memory, same vocabulary for all)
        fit_txt = tq[name] + [t for x in tp for t in x[name][: 1_000_000]]
        vec.fit(fit_txt)
        A = vec.transform(tq[name])
        for p, t in zip(pools, tp):
            B = vec.transform(t[name])
            parts.append(topk(A, B, k, q["entity_id"].to_numpy(), p["entity_id"].to_numpy(), name))
    for p in pools:
        parts.append(exact(q, p, ["addr_hnum", "city"], "hnumcity"))
        parts.append(exact(q, p, ["name_nospace"], "nospace"))
    return parts


def union(parts):
    """Outer-merge all blocker outputs on (s1, cand), keeping each blocker's score/rank columns."""
    long = pl.concat(parts, how="diagonal_relaxed")
    agg = [pl.col(c).max() for c in long.columns if c.startswith("s_") or c.startswith("x_")]
    agg += [pl.col(c).min() for c in long.columns if c.startswith("r_")]
    return long.group_by("s1", "cand").agg(agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="dev")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--norm", default="trn")
    ap.add_argument("--countries", default="", help="comma list; run only these (parallel jobs), output gets a suffix")
    a = ap.parse_args()
    t0 = time.time()
    s1, s2, s3 = load_fold(a.fold, a.norm)
    print(f"{a.fold}: S1 {s1.height:,}  S2 {s2.height:,}  S3 {s3.height:,}", flush=True)
    parts = []
    only = [c for c in a.countries.split(",") if c]
    for country in sorted(s1["country"].unique().to_list()):
        if only and country not in only:
            continue
        q = s1.filter(pl.col("country") == country)
        pools = [s.filter(pl.col("country") == country) for s in (s2, s3)]
        t = time.time()
        parts += block_country(q, pools, a.k)
        print(f"  {country}: {q.height:,} queries, pools {pools[0].height:,}/{pools[1].height:,} "
              f"in {time.time() - t:.0f}s", flush=True)
    cands = union(parts)
    out = ROOT / "cache" / "cand"
    out.mkdir(parents=True, exist_ok=True)
    suffix = ("_" + "-".join(only)) if only else ""
    cands.write_parquet(out / f"{a.fold}_cands_k{a.k}{suffix}.parquet")
    print(f"{cands.height:,} unique pairs ({cands.height / s1.height:.1f} per S1) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
