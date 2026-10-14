"""Dense kNN over the sharded embeddings from embed_encode.py (+ the val-fold files from embed_block.py).

Usage (SLURM GPU job): python embed_knn.py --fold full|test --k 20
Writes cache/cand/{fold}_emb_Qwen3-Embedding-0.6B_k{k}.parquet with columns s1, cand, s_emb, r_emb.
"""
import argparse
import time

import numpy as np
import polars as pl
import torch

from embed_block import load_fold as load_fold_emb
from io_utils import ROOT

MNAME = "Qwen3-Embedding-0.6B"
EDIR = ROOT / "cache" / "emb"


def vectors(split: str, src: int):
    """(ids, fp16 matrix) for every record of split/source, from unit files (+ val files for train)."""
    ids, vecs = [], []
    for f in sorted((EDIR / "units").glob(f"{split}_s{src}_*.npy")):
        ids.append(pl.read_parquet(str(f).replace(".npy", "_ids.parquet"))["entity_id"])
        vecs.append(np.load(f))
    if split == "train":
        vf = EDIR / f"val_{MNAME}_d512_np_source{src}.npy"
        vids = load_fold_emb("val", "trn")[src - 1]["entity_id"]  # same row order as when val was encoded
        ids.append(vids)
        vecs.append(np.load(vf))
    return pl.concat(ids), np.concatenate(vecs)


def knn(Q, P, k, qchunk=2048, pchunk=500_000):
    """Exact top-k inner product; pool processed in chunks with a running top-k merge."""
    k = min(k, P.shape[0])
    out_v, out_i = [], []
    for s in range(0, Q.shape[0], qchunk):
        q = Q[s:s + qchunk]
        best_v = best_i = None
        for ps in range(0, P.shape[0], pchunk):
            sim = q @ P[ps:ps + pchunk].T
            v, i = torch.topk(sim, min(k, sim.shape[1]), dim=1)
            i = i + ps
            if best_v is None:
                best_v, best_i = v, i
            else:
                v = torch.cat([best_v, v], 1)
                i = torch.cat([best_i, i], 1)
                best_v, j = torch.topk(v, k, dim=1)
                best_i = torch.gather(i, 1, j)
        out_v.append(best_v.float().cpu())
        out_i.append(best_i.cpu())
    return torch.cat(out_v).numpy(), torch.cat(out_i).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="test", choices=["full", "test"])
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    split = "test" if a.fold == "test" else "train"
    norm = ROOT / "cache" / "norm_trn"
    parts = []
    data = []
    for src in (1, 2, 3):
        ids, V = vectors(split, src)
        ctry = pl.read_parquet(norm / f"{split}_source{src}.parquet", columns=["entity_id", "country"])
        c = pl.DataFrame({"entity_id": ids}).join(ctry, on="entity_id", how="left")["country"]
        assert c.null_count() == 0 and len(ids) == ctry.height, f"missing embeddings for {split} source{src}"
        data.append((ids.to_numpy(), c.to_numpy(), torch.from_numpy(V)))  # stays on CPU; per-country slices go to GPU
        print(f"loaded {split} source{src}: {len(ids):,}", flush=True)
    q_ids, q_c, Qall = data[0]
    for country in sorted(set(q_c)):
        qm = q_c == country
        Q = Qall[torch.from_numpy(qm)].cuda()
        for p_ids, p_c, Pall in data[1:]:
            pm = p_c == country
            if not pm.any():
                continue
            P = Pall[torch.from_numpy(pm)].cuda()
            v, idx = knn(Q, P, a.k)
            kk = v.shape[1]
            parts.append(pl.DataFrame({
                "s1": np.repeat(q_ids[qm], kk), "cand": p_ids[pm][idx.ravel()],
                "s_emb": v.ravel().astype(np.float32),
                "r_emb": np.tile(np.arange(1, kk + 1, dtype=np.uint16), int(qm.sum()))}))
            del P
            torch.cuda.empty_cache()
        print(f"  knn {country}: {qm.sum():,} queries ({time.time() - t0:.0f}s)", flush=True)
    out = ROOT / "cache" / "cand" / f"{a.fold}_emb_{MNAME}_k{a.k}.parquet"
    pl.concat(parts).write_parquet(out)
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
