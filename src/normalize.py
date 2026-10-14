"""Phase 1: normalize names and addresses for every record (recipe: research/00_cleaning_plan.md).

Usage:
  python normalize.py --translit trn            # all train+test files, maps learned from the train fold
  python normalize.py --selftest                # run on hand examples
Writes cache/norm_{translit}/{split}_{source}.parquet with the original columns plus normalized ones.
"""
import argparse
import json
import os
import time
import unicodedata
from functools import lru_cache
from multiprocessing import Pool

import jellyfish
import polars as pl
import regex as re
import wordninja
from anyascii import anyascii

from io_utils import ROOT, load
from text_tables import (CITY_AFFIX, CITY_SUFFIX, CREDENTIALS, FILLER, FR_REPEAT, IN_CITY, IN_LIKE, LEGAL_PHRASES,
                         LEGAL_TOKENS, NULL_SEGS, ORDINALS, STATE_TABLES, STOP, STREET, UNIT_WORDS)

TOK_MAP: dict = {}
SEG_MAP: dict = {}

# ------------------------------------------------------------------ shared text helpers
_NONLATIN = re.compile(r"[^\p{Latin}\p{Common}\p{Inherited}]")
_NONPUNCT = re.compile(r"[^\p{L}\p{N}\p{M}]+")
_ANUSVARA = {c: "n" for c in (0x0902, 0x0982, 0x0A02, 0x0A82, 0x0B02, 0x0B82, 0x0C02, 0x0C82, 0x0D02)}
_DEDOT = re.compile(r"\b((?:[a-z]\.){2,})")
_WS = re.compile(r"\s+")
_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"))
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"})
_KEEP_ALNUM = re.compile(r"\d+(st|nd|rd|th|[a-z])?|[a-z]\d+")


def translit_segment(seg: str) -> str:
    """Native-script text -> Latin: learned segment map, then learned token map, then anyascii."""
    if not _NONLATIN.search(seg):
        return seg
    key = " ".join(_NONPUNCT.sub(" ", seg.lower()).split())
    if key in SEG_MAP:
        return SEG_MAP[key]
    out = []
    for tok in seg.split():
        if _NONLATIN.search(tok):
            k = _NONPUNCT.sub("", tok.lower())
            out.append(TOK_MAP.get(k) or tok.translate(_ANUSVARA))
        else:
            out.append(tok)
    return " ".join(out)


def to_ascii(s: str) -> str:
    return anyascii(s).lower()


def deleet_token(t: str) -> str:
    if any(c.isdigit() for c in t) and any(c.isalpha() for c in t) and not _KEEP_ALNUM.fullmatch(t):
        return t.translate(_LEET)
    return t


@lru_cache(maxsize=1 << 20)
def phon_token(t: str) -> str:
    if not t or t[0].isdigit():
        return t
    t = t.replace("tion", "sn")
    t = re.sub(r"(.)\1+", r"\1", t)
    t = t.replace("sh", "s").replace("ph", "f").replace("w", "v")
    t = re.sub(r"([bcdfghjklmnpqrstvxz])y", r"\1", t)
    return jellyfish.metaphone(t) or t


def phon(tokens) -> str:
    return " ".join(phon_token(t) for t in tokens)


# ------------------------------------------------------------------ names
_ALIAS = re.compile(r"\s(?:a\s?/\s?k\s?/\s?a|aka|d\s?/\s?b\s?/\s?a|dba|t\s?/\s?a|trading as|formerly known as|formerly"
                    r"|f\s?/\s?k\s?/\s?a|fka|also known as|doing business as|nee)\s")
_DOMAIN = re.compile(r"(?:https?://)?(?:www\.)?\b([a-z0-9][a-z0-9\-]*?)\.(?:co\.in|com|net|org|in|co|io|biz|us|info|fr)\b")
_HANDLE = re.compile(r"^\W*@([a-z0-9_\.]+)\s*$")
_ID_TAG = re.compile(r"\(?\s*id:\s*\d+\s*\)?")
_PIPE_TAIL = re.compile(r"\s*\|\s.*$")
_PHONE = re.compile(r"(?:\s-\s|\s)\+?\d[\d\-\s]{8,}\d\s*$")
_HASHNUM = re.compile(r"\s#\d+\s*$")
_PHRASES = sorted(LEGAL_PHRASES, key=len, reverse=True)


def name_tokens(s: str):
    s = s.replace("&", " and ")
    s = _NONPUNCT.sub(" ", s)
    return [deleet_token(t) for t in s.split()]


def split_legal(tokens):
    """Remove legal forms anywhere in the token list; return (remaining tokens, sorted canonical forms)."""
    text = " " + " ".join(tokens) + " "
    forms = []
    for ph in _PHRASES:
        if f" {ph} " in text:
            forms.extend(LEGAL_PHRASES[ph].split())
            text = text.replace(f" {ph} ", " ")
    rest = []
    for t in text.split():
        if t in LEGAL_TOKENS:
            forms.append(LEGAL_TOKENS[t])
        else:
            rest.append(t)
    return rest, " ".join(sorted(set(forms)))


def core_of(tokens):
    toks = list(tokens)
    while toks and toks[-1] in CREDENTIALS and len(toks) > 1:
        toks.pop()
    core = [t for t in toks if t not in FILLER and t not in STOP]
    return core or [t for t in toks if t not in STOP] or toks


def norm_name(raw: str) -> dict:
    s = unicodedata.normalize("NFKC", raw or "").translate(_ZW)
    s = translit_segment(s)
    s = to_ascii(s)
    s = _DEDOT.sub(lambda m: m.group(1).replace(".", ""), s)
    s = _ID_TAG.sub(" ", s)
    s = _PIPE_TAIL.sub("", s)
    s = _PHONE.sub("", s)
    s = _HASHNUM.sub("", s)
    alias = ""
    m = _ALIAS.search(f" {s} ")
    if m:  # audit: the S1 name is the RIGHT-hand side; the left side is the alias
        left, right = f" {s} "[:m.start()], f" {s} "[m.end():]
        s, alias = (right, left) if right.strip() else (left, "")
    nospace_dom = ""
    s = re.sub(r"\.c0m\b", ".com", s)
    s = re.sub(r"^(\W*(?:sri|shri|mr|mrs|ms|dr|smt)?\s*[a-z]{6,})(com|in)$", r"\1.\2", s.strip())  # 'xyzcom'
    dm = _DOMAIN.search(s) or _HANDLE.match(s.strip())  # anywhere: 'Sri rmztechnologies.com'
    if dm:
        nospace_dom = dm.group(1).replace("-", "").replace("_", "").replace(".", "")
        s = s[:dm.start()] + " " + " ".join(wordninja.split(nospace_dom)) + " " + s[dm.end():]
    toks = name_tokens(s)
    rest, legal = split_legal(toks)
    core = core_of(rest)
    out = {
        "name_clean": " ".join(toks),
        "name_core": " ".join(core),
        "name_sorted": " ".join(sorted(core)),
        # filler kept here: domains are the full name run together ('easternenterprises.com')
        "name_nospace": nospace_dom or "".join(t for t in rest if t not in STOP),
        "name_phon": phon(core),
        "legal_form": legal,
        "name_is_domain": bool(dm),
        "name_alias": "",
    }
    if alias.strip():
        a_rest, _ = split_legal(name_tokens(alias))
        out["name_alias"] = " ".join(core_of(a_rest))
    return out


# ------------------------------------------------------------------ addresses
_COMPOUND = re.compile(r"\b([a-z]{0,2}\d+[a-z]?(?:\s?[-/]\s?[a-z]?\d*[a-z]?)+)\b")
_PREFIX = re.compile(r"^(?:#+|(?:h|plot|door|d|sf|kh|khasra|shop|flat|survey|sy)?\s*n[o°](?![a-z])\.?|hno(?![a-z])\.?)"
                     r"\s*[-.:]?\s*")
_LETTER_NUM = re.compile(r"\b([a-z]{1,2})\s?-\s?(\d+[a-z]?)\b")
_POBOX = re.compile(r"\b(?:p\s?o\s?box|post box|pob|pmb)\s*#?\s*\d*\b")
_NUMTOK = re.compile(r"\d+[a-z]?|[a-z]\d+[a-z]?|\d+[a-z]\d+")
_ORD_NUM = re.compile(r"\d+(?:st|nd|rd|th)")


def _state_table(country: str):
    return STATE_TABLES.get((country or "").strip().lower(), {})


def _clean_seg(seg: str) -> str:
    seg = _POBOX.sub(" ", seg)
    seg = seg.replace("°", "o ")
    seg = _PREFIX.sub("", seg.strip())
    seg = seg.replace("#", " ")
    # join compound numbers: 1-79/2 -> 1792, d-5 -> d5, 101-c -> 101c; keep 1/2 as 'half'
    seg = re.sub(r"\b1\s?/\s?2\b", " half ", seg)
    seg = _LETTER_NUM.sub(r"\1\2", seg)
    seg = _COMPOUND.sub(lambda m: re.sub(r"[\s\-/]+", "", m.group(1)), seg)
    seg = re.sub(r"[^a-z0-9 ]+", " ", seg)
    return _WS.sub(" ", seg).strip()


def norm_addr(raw: str, country: str) -> dict:
    s = unicodedata.normalize("NFKC", raw or "").translate(_ZW)
    segs = [to_ascii(translit_segment(x)) for x in s.split(",")]
    segs = [_clean_seg(x) for x in segs]
    segs = [x for x in segs if x and x not in NULL_SEGS]
    table = _state_table(country)
    state, rest = "", []
    for x in segs:  # order-free: the LAST segment that is a state/region wins, others stay as tokens
        if x in table:
            if state:
                rest.append(state_seg)
            state, state_seg = table[x], x
        else:
            rest.append(x)
    # city = last remaining segment with no digits
    city = ""
    for x in reversed(rest):
        if not any(c.isdigit() for c in x):
            city = x
            break
    for p in CITY_AFFIX:
        if city.startswith(p):
            city = city[len(p):]
    for p in CITY_SUFFIX:
        if city.endswith(p) and len(city) > len(p) + 2:
            city = city[: -len(p)]
    if country.strip().lower() in IN_LIKE:
        city = IN_CITY.get(city, city)
    nums, words, hnum = [], [], ""
    for tok_i, t in enumerate(" ".join(rest).split()):
        t = ORDINALS.get(t, t)
        if t in FR_REPEAT and nums:
            nums[-1] += FR_REPEAT[t]
            continue
        if _ORD_NUM.fullmatch(t):
            words.append(t)  # ordinals are street names, not house numbers
            continue
        if _NUMTOK.fullmatch(t):
            n = t.lstrip("0") or t
            nums.append(n)
            if not hnum:
                hnum = n
            continue
        t = STREET.get(t, t)
        if country.strip().lower() in IN_LIKE:
            t = IN_CITY.get(t, t)
        if t not in UNIT_WORDS and t not in STOP:
            words.append(t)
    wset = sorted(set(words))
    return {
        "addr_clean": " ".join(rest),
        "addr_nums": " ".join(sorted(set(nums))),
        "addr_hnum": hnum,
        "addr_tokens": " ".join(wset),
        "addr_phon": phon(wset),
        "state": state,
        "city": city,
    }


# ------------------------------------------------------------------ driver
def _init(maps):
    global TOK_MAP, SEG_MAP
    TOK_MAP, SEG_MAP = maps["tok_map"], maps["seg_map"]


def _work(rows):
    out = []
    for name, addr, country in rows:
        r = norm_name(name)
        r.update(norm_addr(addr, country))
        out.append(r)
    return out


def normalize_df(df: pl.DataFrame, maps, pool) -> pl.DataFrame:
    rows = list(zip(df["business_name"].to_list(), df["business_address"].to_list(), df["country"].to_list()))
    chunk = 20_000
    parts = pool.map(_work, [rows[i:i + chunk] for i in range(0, len(rows), chunk)])
    recs = [r for p in parts for r in p]
    return pl.concat([df, pl.DataFrame(recs)], how="horizontal")


def selftest():
    names = ['शिवा सॉल्यूशंस एलएलपी', 'Shiva Solutions LLP', 'ಗೋಲ್ಡನ್ ಫುಡ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್', 'G0lden Foods Private Ltd',
             'Golden Foods Private Limited', 'Adr Good Prívate Limited', 'JBF RÉSOURCES PRIVATE LIMITED',
             'giudittahildebrantharbor.com', 'Vantagerizariza a/k/a Global Committee', '-- Holloway Peak Inc Seafood',
             'SIOUX CITY-HERITAGE PRIME ALLIANCE (LLC)', 'Sioux City Heritage Prime Alliance L.L.C.',
             'inc beacon freight alliance', 'QHC Culture [EURL]', 'Thermal & Fils SASU', 'SCI Ptit Àmicale',
             'Carter, Kellia, DPM', 'Shri Jsk Law  Chambers', 'Jsk Law Chambers Company', 'Syno Cmoagtal Owl, Inc']
    for n in names:
        r = norm_name(n)
        print(f"{n[:44]:44} | core={r['name_core']!r} legal={r['legal_form']!r} phon={r['name_phon']!r}"
              + (f" alias={r['name_alias']!r}" if r["name_alias"] else ""))
    addrs = [('2310 FIFTH STREET, SIOUX CITY, IA', 'US'), ('2310 5th Street, Sioux City, Iowa', 'US'),
             ('Sioux City, Iowa, 2310 Fifth Saint', 'US'), ('#3512 Stellhorn Rd, NULL, Fort Wayne, Indiana', 'US'),
             ('1058 1/2 Sharon Ridge Rd, Fairviw, North Carolina', 'US'), ('101-C CENTRAL AVE, BRAINTREE, MA', 'US'),
             ('2310 5th Street, PO Box 5219, Sioux City, Iowa', 'US'),
             ('D-5, GROUND FLOOR, ANAND VIHAR, LAXMI NAGAR, EAST DELHI, Delhi', 'India'),
             ('H NO-1-79/2, RAGHAVENDRA COLONY UPPAL, HYDERABAD, Andhra Pradesh', 'India'),
             ('H No-1-79/2, Raghavendra Colony Uppal, Hyderabad, Telangana', 'India'),
             ('CHOUDHARY CAMPUS, MAHARSHI DAYANAND COLONY, BHAWANPURA THIKRIA, BANSWARA, राजस्थान', 'India'),
             ('Sf No 172 Tiny Shop No 22, Sidco Indutrial Estate, Coimbatore South, தமிழ்நாடு', 'India'),
             ('63 R. DE DIEPPE, LILLE, Hauts-de-France', 'France'), ('63 Rue de Dieppe, Lille, Nord', 'France'),
             ('NO. 5 ALLÉE DES HÊTRES, Pornic, Loire-Atlantique', 'France'),
             ('Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac', 'France')]
    for a, c in addrs:
        r = norm_addr(a, c)
        print(f"{a[:52]:52} | nums={r['addr_nums']!r} hnum={r['addr_hnum']!r} state={r['state']!r} "
              f"city={r['city']!r} toks={r['addr_tokens']!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translit", default="trn", choices=["trn", "all"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 8))
    a = ap.parse_args()
    maps = json.load(open(ROOT / "cache" / f"translit_{a.translit}.json"))
    _init(maps)
    if a.selftest:
        selftest()
        return
    out = ROOT / "cache" / f"norm_{a.translit}"
    out.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers, initializer=_init, initargs=(maps,)) as pool:
        for split in ["train", "test"]:
            for src in ["source1", "source2", "source3"]:
                t = time.time()
                df = normalize_df(load(split, src), maps, pool)
                df.write_parquet(out / f"{split}_{src}.parquet")
                print(f"{split}_{src}: {df.height:,} rows in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
