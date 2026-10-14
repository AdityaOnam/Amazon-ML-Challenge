"""France-only descriptor-swap flag (research/09_france_round2.md, fix #2).

French S1 names are built from template descriptors (club, ecole, comite, amicale...). A candidate whose name
swaps one descriptor for ANOTHER (S1 has a descriptor the candidate lacks and vice versa, with no typo/prefix
link between them) is a sibling business, not a noisy copy. In US/India such swaps are character scrambles and
99% true, so the model (trained there) accepts them; in France the count test estimates ~0% are true.
D = name-core words in >= 0.2% of French test S1 cores, minus words of common city names (test inputs only).
"""
import polars as pl
from rapidfuzz.distance import Levenshtein

from io_utils import ROOT


def vocab(norm="fr", country="France", frac=0.002):
    s1 = (pl.read_parquet(ROOT / f"cache/norm_{norm}/test_source1.parquet", columns=["country", "name_core", "city"])
            .filter(pl.col("country") == country))
    cc = s1.group_by("city").len().filter(pl.col("len") >= 100)["city"].to_list()
    cities = {t for c in cc for t in c.split()} - {"saint", "st", "la", "le", "de", "sur", "du"}
    v = (s1.select(t=pl.col("name_core").str.split(" ").list.unique()).explode("t").group_by("t").len()
           .filter(pl.col("len") >= frac * s1.height))
    return set(v["t"].to_list()) - cities


def _swap(a: str, b: str, D) -> bool:
    A, B = set(a.split()), set(b.split())
    oa, ob = (A - B) & D, (B - A) & D
    return bool(oa and ob) and all(not (Levenshtein.distance(x, y) <= 2 or x.startswith(y) or y.startswith(x))
                                   for x in oa for y in ob)


def flag_pairs(pairs: pl.DataFrame, norm="fr") -> pl.DataFrame:
    """pairs: s1, cand (French). Returns s1, cand, dswap."""
    D = vocab(norm)
    n1 = pl.read_parquet(ROOT / f"cache/norm_{norm}/test_source1.parquet", columns=["entity_id", "name_core"])
    n23 = pl.concat([pl.read_parquet(ROOT / f"cache/norm_{norm}/test_source{i}.parquet", columns=["entity_id", "name_core"])
                     for i in (2, 3)])
    d = (pairs.select("s1", "cand").join(n1.rename({"entity_id": "s1"}), on="s1")
              .join(n23.rename({"entity_id": "cand", "name_core": "name_core_m"}), on="cand"))
    fl = [_swap(a, b, D) for a, b in zip(d["name_core"].to_list(), d["name_core_m"].to_list())]
    print(f"descriptor vocabulary {len(D)} words; swaps {sum(fl):,} of {d.height:,} pairs", flush=True)
    return d.select("s1", "cand", dswap=pl.Series(fl))
