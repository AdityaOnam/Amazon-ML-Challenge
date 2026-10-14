"""Learn native-script -> Latin maps from TRAINING pairs only (no external data).

tok_map: native name token -> Latin token, by position alignment of same-length names
         (95% of fully native names have the same token count as their S1 name; audit §3).
seg_map: native address segment (e.g. 'महाराष्ट्र') -> the S1 address segment it co-occurs with most.

Usage: python translit.py [--fold trn|all]   (trn = learn from the 80% train fold only; use for validation)
Writes cache/translit_{fold}.json
"""
import argparse
import json
import unicodedata
from collections import Counter, defaultdict

import polars as pl
import regex as re
from anyascii import anyascii

from io_utils import ROOT, gt_long, load

_NONPUNCT = re.compile(r"[^\p{L}\p{N}\p{M}]+")
_NONLATIN = re.compile(r"[^\p{Latin}\p{Common}\p{Inherited}]")


def is_native(s: str) -> bool:
    return bool(_NONLATIN.search(s))


_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"))  # zero-width joiners split Indic words otherwise


def native_tokens(s: str) -> list[str]:
    return _NONPUNCT.sub(" ", unicodedata.normalize("NFKC", s).translate(_ZW).lower()).split()


def latin_tokens(s: str) -> list[str]:
    s = re.sub(r"\b((?:[A-Za-z]\.){2,})", lambda m: m.group(1).replace(".", ""), s)
    return _NONPUNCT.sub(" ", anyascii(unicodedata.normalize("NFKC", s)).lower()).split()


def native_seg_key(seg: str) -> str:
    return " ".join(native_tokens(seg))


def latin_seg_key(seg: str) -> str:
    return " ".join(latin_tokens(seg))


def learn(pairs: pl.DataFrame, min_count=3, min_share=0.5):
    tok_cnt = defaultdict(Counter)
    seg_cnt = defaultdict(Counter)
    seg_tot = Counter()
    for n1, a1, n2, a2 in pairs.select("name1", "addr1", "name2", "addr2").iter_rows():
        if is_native(n2):
            lt, nt = latin_tokens(n1), native_tokens(n2)
            if len(lt) == len(nt):
                for x, y in zip(nt, lt):
                    if is_native(x):
                        tok_cnt[x][y] += 1
        if is_native(a2):
            s1_segs = {latin_seg_key(s) for s in a1.split(",") if s.strip()}
            for seg in a2.split(","):
                if is_native(seg):
                    k = native_seg_key(seg)
                    seg_tot[k] += 1
                    for s in s1_segs:
                        seg_cnt[k][s] += 1

    def pick(cnt, tot=None):
        out = {}
        for k, c in cnt.items():
            (best, n), total = c.most_common(1)[0], (tot[k] if tot else sum(c.values()))
            if n >= min_count and n / total >= min_share:
                out[k] = best
        return out

    return pick(tok_cnt), pick(seg_cnt, seg_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="trn", choices=["trn", "all"])
    a = ap.parse_args()
    if a.fold == "trn":
        gt = pl.read_csv(ROOT / "cache/split/trn_ground_truth.tsv", separator="\t", infer_schema=False).with_columns(pl.col("matched_entity_ids").fill_null(""))
    else:
        gt = load("train", "ground_truth")
    L = gt_long(gt)
    s1 = load("train", "source1").filter(pl.col("country").str.to_lowercase().is_in(["india", "in"]))
    s23 = pl.concat([load("train", "source2"), load("train", "source3")])
    pairs = (L.join(s1.select("entity_id", name1="business_name", addr1="business_address"),
                    left_on="source1_entity_id", right_on="entity_id")
              .join(s23.select("entity_id", name2="business_name", addr2="business_address"),
                    left_on="match_id", right_on="entity_id")
              .filter(pl.col("name2").str.contains(r"[^\x00-\x7F]") | pl.col("addr2").str.contains(r"[^\x00-\x7F]")))
    print("pairs with non-ASCII match text:", pairs.height)
    tok_map, seg_map = learn(pairs)
    out = ROOT / "cache" / f"translit_{a.fold}.json"
    json.dump({"tok_map": tok_map, "seg_map": seg_map}, open(out, "w"), ensure_ascii=False)
    print(f"tok_map {len(tok_map):,} tokens, seg_map {len(seg_map):,} segments -> {out}")
    for k in list(seg_map)[:15]:
        print("  seg", k, "->", seg_map[k])
    for k in list(tok_map)[:15]:
        print("  tok", k, "->", tok_map[k])


if __name__ == "__main__":
    main()
