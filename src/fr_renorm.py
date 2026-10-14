"""France-only re-normalisation of the test records (research/08_france_audit.md fixes #1, #3, #4).

 1. injected fillers: FR_INJECTED tokens are dropped from French name cores (as in fr_filler.py)
 2. domain / handle names: the run-together string is split with a FRENCH word model built from the
    test set's own French names (no external data) instead of wordninja's English one; accent-dropped
    spellings in domains ('collges' for 'collèges') map back to the folded word; a trailing legal form
    ('...sarl.com') is split off first, and the nospace key is kept without it (like non-domain names)
 3. addresses: 'N° 12', '12 - rue' and '21bis' no longer push the house number onto the postcode
Writes cache/norm_fr/test_source{1,2,3}.parquet = cache/norm_trn copies with the French rows replaced.
Usage: python fr_renorm.py [--no-addr]   (--no-addr writes cache/norm_frn/)
"""
import argparse
import gzip
import json
import time
import unicodedata
from collections import Counter
from multiprocessing import Pool

import polars as pl
import regex as re
import wordninja

import normalize as N
from io_utils import ROOT
from text_tables import FILLER, FR_INJECTED, LEGAL_TOKENS

OUT = ROOT / "cache/norm_fr"
WORDS = ROOT / "cache/fr_words.txt.gz"
FR_LEGAL = ("selarl", "sasu", "sarl", "eurl", "eirl", "scop", "sas", "sci", "snc", "scm", "gie", "sa")
_LEGAL_TAIL = re.compile(r"^(.{4,}?)(" + "|".join(FR_LEGAL) + r")$")
_NDEG = re.compile(r"(?i)\bn\s*[°º]\s*")
_BIS = re.compile(r"(?i)\b(\d+)\s*(?:bis|ter|quater)\b")
_DASH = re.compile(r"(\d+)\s+-\s+")

LM = None
DROPPED = {}  # accent-dropped spelling -> folded spelling


def _fold(tok: str) -> str:
    return unicodedata.normalize("NFKD", tok).encode("ascii", "ignore").decode()


def build_words():
    """Word list (most frequent first) + dropped-accent map from the French test names that are NOT domains."""
    cnt, drop = Counter(), Counter()
    for i in (1, 2, 3):
        names = (pl.scan_parquet(ROOT / f"cache/raw/test_source{i}.parquet").filter(pl.col("country") == "France")
                   .select("business_name").collect()["business_name"].to_list())
        for nm in names:
            s = (nm or "").lower()
            if "." in s or "@" in s or " " not in s.strip():
                continue
            for t in re.findall(r"[^\W\d_]+", s):
                f = _fold(t)
                if len(f) >= 2 or f in ("a", "y"):
                    cnt[f] += 1
                    d = re.sub(r"[^a-z]", "", t)  # accented letters deleted, as the domain writer does
                    if d != f and len(d) >= 3:
                        drop[(d, f)] += 1
    words = [w for w, c in cnt.most_common() if c >= 3]
    with gzip.open(WORDS, "wt") as fh:
        fh.write("\n".join(words))
    dmap = {}
    for (d, f), c in drop.most_common():
        if c >= 3 and d not in dmap and cnt.get(d, 0) < c:
            dmap[d] = f
    json.dump(dmap, open(ROOT / "cache/fr_dropped.json", "w"))
    print(f"French word model: {len(words):,} words; {len(dmap):,} accent-dropped spellings", flush=True)


class FrenchSplitter:
    """Drop-in for the `wordninja` module inside normalize.norm_name (only .split is used)."""

    @staticmethod
    def split(s: str):
        s = s.lower()
        m = _LEGAL_TAIL.match(s)
        legal = []
        if m and m.group(1) and LM is not None:
            s, legal = m.group(1), [m.group(2)]
        toks = LM.split(s) if s else []
        return [DROPPED.get(t, t) for t in toks] + legal


ADDR_FIX = True


def _init(maps, addr_fix=True):
    global LM, DROPPED, ADDR_FIX
    ADDR_FIX = addr_fix
    N._init(maps)
    LM = wordninja.LanguageModel(str(WORDS))
    DROPPED = json.load(open(ROOT / "cache/fr_dropped.json"))
    N.wordninja = FrenchSplitter
    N.FILLER = FILLER | FR_INJECTED


def _addr_pre(s: str) -> str:
    s = _NDEG.sub(" ", s or "")
    s = _BIS.sub(r"\1 ", s)
    return _DASH.sub(r"\1 ", s)


def _work(rows):
    out = []
    for name, addr in rows:
        r = N.norm_name(name)
        if r["name_is_domain"]:  # nospace key without the legal tail, like non-domain names
            m = _LEGAL_TAIL.match(r["name_nospace"])
            if m:
                r["name_nospace"] = m.group(1)
        r.update(N.norm_addr(_addr_pre(addr) if ADDR_FIX else addr, "France"))
        out.append(r)
    return out


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-addr", action="store_true", help="names only (skip the address pre-clean, fix 3)")
    a = ap.parse_args()
    if a.no_addr:
        OUT = ROOT / "cache/norm_frn"
    t = time.time()
    build_words()
    maps = json.load(open(ROOT / "cache/translit_trn.json"))
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(96, initializer=_init, initargs=(maps, not a.no_addr)) as pool:
        for i in (1, 2, 3):
            old = pl.read_parquet(ROOT / f"cache/norm_trn/test_source{i}.parquet")
            fr = old.filter(pl.col("country") == "France")
            rows = list(zip(fr["business_name"].to_list(), fr["business_address"].to_list()))
            parts = pool.map(_work, [rows[k:k + 20_000] for k in range(0, len(rows), 20_000)])
            new = pl.DataFrame([r for p in parts for r in p])
            base = [c for c in fr.columns if c not in new.columns]
            fr_new = pl.concat([fr.select(base), new], how="horizontal").select(old.columns)
            ch = {c: (fr[c] != fr_new[c]).sum() for c in ("name_core", "name_nospace", "addr_hnum", "addr_clean")}
            pl.concat([old.filter(pl.col("country") != "France"), fr_new]).write_parquet(OUT / f"test_source{i}.parquet")
            print(f"source{i}: {fr.height:,} French rows; changed {ch} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
