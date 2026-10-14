"""Shared route builder + route-pair features for the two accepted shortlist routes (research/12):
  taddr   transitive address route: each confident anchor record x (production p > 0.9 for S1 s) queries the
          UNCLAIMED records (no S1 has production p >= 0.5), non-empty address, by address word TF-IDF
          (addr_tokens + n<numbers>), top-3, cosine >= 0.3  ->  (s, y)
  revmix  reverse kNN: each UNCLAIMED record y queries the S1 index by name+address word TF-IDF
          (block.py tf_mix text), top-3, cosine >= 0.05  ->  (s, y)
Only record text and the production p are used (no labels). Pairs already in `exclude` are dropped.
Used by s12_14 (train, cross-fitted), s12_13 (test-density simulation) and s12_apply_test.py (test)."""
import os

os.environ.setdefault("SLURM_CPUS_PER_TASK", "32")
from s12_00_common import *  # noqa: F403,E402
from rapidfuzz import fuzz, process  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sparse_dot_topn import sp_matmul_topn  # noqa: E402

import features  # noqa: E402  (src/features.py, read-only import)

WORD = dict(analyzer="word", token_pattern=r"\S+", min_df=2, sublinear_tf=True, dtype=np.float32)
RCOLS = ["entity_id", "country", "name_core", "addr_tokens", "addr_nums"]
K_T, K_R = 3, 3


def _addr(df):
    return (df["addr_tokens"] + " " + df["addr_nums"].str.replace_all(r"(\S+)", "n$1")).to_list()


def _mix(df):
    return (df["name_core"] + " " + df["addr_tokens"] + " "
            + df["addr_nums"].str.replace_all(r"(\S+)", "n$1")).to_list()


def _topk(A, B, k, thr):
    return sp_matmul_topn(A, B.T.tocsr(), top_n=k, threshold=thr, n_threads=32, sort=True).tocoo()


def build_routes(S1, R23, fin, exclude, log=print):
    """S1, R23: records (RCOLS). fin: production p (s1, cand, p) over all S1. exclude: (s1, cand) already scored."""
    claimed = fin.filter(pl.col("p") >= 0.5).select(entity_id="cand").unique()
    anch = fin.filter(pl.col("p") > 0.9).select("s1", x="cand")
    U = R23.join(claimed, on="entity_id", how="anti")
    UA = U.filter(pl.col("addr_tokens") + pl.col("addr_nums") != "")
    QA = R23.filter(pl.col("addr_tokens") + pl.col("addr_nums") != "").join(
        anch.select(entity_id="x").unique(), on="entity_id", how="semi")
    log(f"  routes: anchors {QA.height:,}, unclaimed {U.height:,} ({UA.height:,} with address)")
    tp, rp = [], []
    for country in sorted(S1["country"].unique().to_list()):
        q, ix = QA.filter(pl.col("country") == country), UA.filter(pl.col("country") == country)
        if q.height and ix.height:
            vec = TfidfVectorizer(**WORD).fit(_addr(ix) + _addr(q)[:2_000_000])
            A, B = vec.transform(_addr(q)), vec.transform(_addr(ix))
            qi, ii = q["entity_id"].to_numpy(), ix["entity_id"].to_numpy()
            for s in range(0, A.shape[0], 200_000):
                Cm = _topk(A[s:s + 200_000], B, K_T, 0.3)
                tp.append(pl.DataFrame({"x": qi[Cm.row + s], "cand": ii[Cm.col], "sc": Cm.data.astype(np.float32)}))
        q, ix = U.filter(pl.col("country") == country), S1.filter(pl.col("country") == country)
        if q.height and ix.height:
            vec = TfidfVectorizer(**WORD).fit(_mix(ix) + _mix(q)[:1_000_000])
            A, B = vec.transform(_mix(q)), vec.transform(_mix(ix))
            qi, ii = q["entity_id"].to_numpy(), ix["entity_id"].to_numpy()
            for s in range(0, A.shape[0], 200_000):
                Cm = _topk(A[s:s + 200_000], B, K_R, 0.05)
                rp.append(pl.DataFrame({"s1": ii[Cm.col], "cand": qi[Cm.row + s], "sc": Cm.data.astype(np.float32)}))
        log(f"  routes {country} done ({T()})")
    t = (pl.concat(tp).with_columns(rk=pl.col("sc").rank("ordinal", descending=True).over("x").cast(pl.Int16))
           .join(anch, on="x").group_by("s1", "cand").agg(sc_taddr_word=pl.col("sc").max(),
                                                           rk_taddr_word=pl.col("rk").min()))
    r = (pl.concat(rp).with_columns(rk=pl.col("sc").rank("ordinal", descending=True).over("cand").cast(pl.Int16))
           .select("s1", "cand", sc_revmix="sc", rk_revmix="rk"))
    R = (t.join(r, on=["s1", "cand"], how="full", coalesce=True)
          .with_columns(f_taddr_word=pl.col("sc_taddr_word").is_not_null(), f_revmix=pl.col("sc_revmix").is_not_null())
          .join(exclude, on=["s1", "cand"], how="anti"))
    return R


def route_features(R, s1r, s23r, fin):
    """R: (s1, cand, route cols). s1r/s23r: features.records(...) tables. fin: production p over all S1."""
    c = R.select("s1", "cand").with_columns(
        [pl.lit(None, pl.Float32).alias(f"s_{b}") for b in ("tf_name", "tf_addr", "tf_mix", "emb")]
        + [pl.lit(None, pl.UInt16).alias(f"r_{b}") for b in ("tf_name", "tf_addr", "tf_mix", "emb")]
        + [pl.lit(False).alias("x_hnumcity"), pl.lit(False).alias("x_nospace")])
    d = features.build(c, s1r, s23r)
    drop = [x for x in d.columns if x.startswith(("s_tf", "r_tf", "x_", "s_emb", "r_emb", "mix_", "cand_best",
                                                  "cand_is", "cand_margin"))]
    d = d.drop(drop).join(R, on=["s1", "cand"])
    cl = fin.group_by("cand").agg(cmax=pl.col("p").max(), csum=pl.col("p").sum(), cn=pl.len().cast(pl.Int16))
    sx = fin.group_by("s1").agg(s_pmax=pl.col("p").max(), s_n90=(pl.col("p") > 0.9).sum().cast(pl.Int16),
                                s_n75=(pl.col("p") >= 0.75).sum().cast(pl.Int16), s_psum=pl.col("p").sum())
    d = d.join(cl, on="cand", how="left").join(sx, on="s1", how="left").with_columns(
        pl.col("cmax").fill_null(0), pl.col("csum").fill_null(0), pl.col("cn").fill_null(0))
    cols = ["entity_id", "name_core", "addr_clean", "addr_hnum"]
    R23 = s23r.select(cols)
    an = fin.filter(pl.col("p") > 0.9).select("s1", x="cand").join(d.select("s1").unique(), on="s1", how="semi")
    pr = (d.select("s1", "cand").join(an, on="s1")
            .join(R23.rename({c_: c_ + "_y" for c_ in cols[1:]}), left_on="cand", right_on="entity_id")
            .join(R23.rename({c_: c_ + "_x" for c_ in cols[1:]}), left_on="x", right_on="entity_id"))
    kw = dict(workers=32, dtype=np.float32)
    pr = pr.with_columns(
        ad=pl.Series(process.cpdist(pr["addr_clean_y"].to_list(), pr["addr_clean_x"].to_list(),
                                    scorer=fuzz.token_set_ratio, **kw)),
        nm=pl.Series(process.cpdist(pr["name_core_y"].to_list(), pr["name_core_x"].to_list(),
                                    scorer=fuzz.token_sort_ratio, **kw)),
        hn=(pl.col("addr_hnum_y") != "") & (pl.col("addr_hnum_y") == pl.col("addr_hnum_x")),
        ae=(pl.col("addr_clean_y") == "") | (pl.col("addr_clean_x") == ""))
    k = pr.group_by("s1", "cand").agg(k_ad=pl.col("ad").filter(~pl.col("ae")).max(), k_nm=pl.col("nm").max(),
                                      k_hn=pl.col("hn").any(), k_n=pl.len().cast(pl.Int16))
    return d.join(k, on=["s1", "cand"], how="left").with_columns(pl.col("k_n").fill_null(0))


def X_of(d, FE):
    return d.select(pl.col(x).cast(pl.Float32) for x in FE).to_numpy()
