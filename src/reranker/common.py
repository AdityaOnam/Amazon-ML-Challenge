"""Shared helpers for the GPU experiments (run on any server: everything is relative to --root / the bundle).

Nothing here imports from src/ except the optional merge/apply scripts, which run back on the home server.
"""
import json
import math
import os
import time
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent


def bundle_dir(a) -> Path:
    return Path(a.bundle) if getattr(a, "bundle", None) else HERE / "bundle"


def results_dir(a) -> Path:
    d = Path(a.results) if getattr(a, "results", None) else HERE / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_env_caches(root: Path):
    """Keep every cache inside the project tree (HF, torch, tmp)."""
    os.environ.setdefault("HF_HOME", str(root / ".hf_cache"))
    os.environ.setdefault("TORCH_HOME", str(root / ".torch_cache"))
    os.environ.setdefault("TMPDIR", str(root / ".tmp"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- serialisation
NAME_CH, ADDR_CH = 90, 140  # char caps so both fields survive max_len truncation


def serialise(name: str, addr: str, country: str, fmt: str = "lite") -> str:
    """Ditto-style attribute serialisation (Li et al., VLDB 2021).
    fmt=ditto : "[COL] name [VAL] .. [COL] address [VAL] .. [COL] country [VAL] .."  (~43 tokens/record, p50)
    fmt=lite  : "name: .. | address: .."  (~26 tokens/record; same attribute tagging, half the tokens; the
                country is dropped because both sides of a pair always share it)."""
    name = (name or "")[:NAME_CH]
    addr = (addr or "")[:ADDR_CH] or "(empty)"
    if fmt == "ditto":
        return f"[COL] name [VAL] {name} [COL] address [VAL] {addr} [COL] country [VAL] {country}"
    return f"name: {name} | address: {addr}"


def attach_text(pairs: pl.DataFrame, rec: pl.DataFrame, text: str = "raw", fmt: str = "lite") -> pl.DataFrame:
    """pairs(s1, cand, ...) + records -> adds columns ta (S1 text) and tb (candidate text)."""
    nc, ac = ("business_name", "business_address") if text == "raw" else ("name_clean", "addr_clean")
    r = rec.select("entity_id", pl.col(nc).alias("n"), pl.col(ac).alias("a"), "country")
    out = pairs.join(r.rename({"entity_id": "s1", "n": "n1", "a": "a1", "country": "c1"}), on="s1", how="left") \
               .join(r.rename({"entity_id": "cand", "n": "n2", "a": "a2", "country": "c2"}), on="cand", how="left")
    miss = out.filter(pl.col("n1").is_null() | pl.col("n2").is_null()).height
    assert miss == 0, f"{miss} pairs have no record text (bundle records incomplete)"
    ta = [serialise(n, a, c, fmt) for n, a, c in zip(out["n1"].to_list(), out["a1"].to_list(), out["c1"].to_list())]
    tb = [serialise(n, a, c, fmt) for n, a, c in zip(out["n2"].to_list(), out["a2"].to_list(), out["c2"].to_list())]
    return out.drop("n1", "a1", "c1", "n2", "a2", "c2").with_columns(ta=pl.Series(ta), tb=pl.Series(tb))


def load_records(bundle: Path) -> pl.DataFrame:
    return pl.read_parquet(bundle / "records.parquet")


# ---------------------------------------------------------------- Qwen3-Reranker prompt (official system prompt; ~72 tokens overhead)
QWEN_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
               "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN_INSTR = "Do the two business records refer to the same real-world business?"


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return f"{time.time() - self.t0:,.0f}s"


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))
