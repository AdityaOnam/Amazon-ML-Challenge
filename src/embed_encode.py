"""Sharded embedding worker. Any number of workers (on any MIG slice size) can run at once:
each claims the next unencoded 1M-row unit via an atomic mkdir lock, encodes it, and moves on.
Test units come first. Train records already encoded for the val fold are reused, not re-encoded.

Usage (inside SLURM GPU jobs): python embed_encode.py --bs 512
Writes cache/emb/units/{unit}.npy (fp16, L2-normalised, first --dim dims) + {unit}_ids.parquet
"""
import argparse
import os
import time

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

from embed_block import encode
from io_utils import ROOT

MODEL = "models/Qwen3-Embedding-0.6B"
UNIT_ROWS = 1_000_000
UDIR = ROOT / "cache" / "emb" / "units"


def units(norm="trn"):
    nd = ROOT / "cache" / f"norm_{norm}"
    val_ids = pl.concat([pl.read_parquet(ROOT / "cache/split" / f"val_source{i}.parquet", columns=["entity_id"])
                         for i in (1, 2, 3)])
    out = []
    for split in ("test", "train"):
        for i in (1, 2, 3):
            df = pl.read_parquet(nd / f"{split}_source{i}.parquet", columns=["entity_id", "name_clean", "addr_clean"])
            if split == "train":  # val records already have embeddings from the val run
                df = df.join(val_ids, on="entity_id", how="anti")
            for j, s in enumerate(range(0, df.height, UNIT_ROWS)):
                out.append((f"{split}_s{i}_{j:02d}", df.slice(s, UNIT_ROWS)))
    return out


def claim(name):
    if (UDIR / f"{name}.npy").exists():
        return False
    try:
        os.mkdir(UDIR / f"{name}.lock")
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--dim", type=int, default=512)
    a = ap.parse_args()
    UDIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(str(ROOT / MODEL), device="cuda", model_kwargs={"torch_dtype": torch.bfloat16},
                                tokenizer_kwargs={"padding_side": "left"})
    model.max_seq_length = 48
    todo = units()
    print(f"{len(todo)} units total", flush=True)
    for name, df in todo:
        if not claim(name):
            continue
        t = time.time()
        e = encode(model, df, a.dim, a.bs, prompt=False)
        np.save(UDIR / f"{name}.npy", e.cpu().numpy())
        df.select("entity_id").write_parquet(UDIR / f"{name}_ids.parquet")
        try:
            os.rmdir(UDIR / f"{name}.lock")
        except FileNotFoundError:
            pass
        print(f"{name}: {df.height:,} in {time.time() - t:.0f}s ({df.height / (time.time() - t):,.0f}/s)", flush=True)
    print("no units left", flush=True)


if __name__ == "__main__":
    main()
