"""Dense blocker: Qwen3-Embedding vectors of 'name | address', exact top-k cosine per country and per pool (GPU).

Usage (inside a SLURM GPU job):
  python embed_block.py --fold val --model models/Qwen3-Embedding-0.6B --k 20
Writes cache/emb/{fold}_{model}_source{i}.npy (fp16, L2-normalised, first --dim dims) and
       cache/cand/{fold}_emb_{model}_k{k}.parquet with columns s1, cand, s_emb, r_emb.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

from io_utils import ROOT

COLS = ["entity_id", "country", "name_clean", "addr_clean"]
PROMPT = "Instruct: Given a business name and address, retrieve records of the same business\nQuery: "


def load_fold(fold, norm):
    nd = ROOT / "cache" / f"norm_{norm}"
    if fold == "test":
        return [pl.read_parquet(nd / f"test_source{i}.parquet", columns=COLS) for i in (1, 2, 3)]
    out = []
    for i in (1, 2, 3):
        ids = pl.read_parquet(ROOT / "cache/split" / f"{fold}_source{i}.parquet", columns=["entity_id"])
        out.append(pl.read_parquet(nd / f"train_source{i}.parquet", columns=COLS).join(ids, on="entity_id", how="semi"))
    return out


def encode(model, df, dim, bs, prompt):
    txt = (df["name_clean"] + " | " + df["addr_clean"]).to_list()
    # sort by length so batches pad little; restore order afterwards
    order = np.argsort([len(t) for t in txt])
    emb = model.encode([txt[i] for i in order], prompt=PROMPT if prompt else None, batch_size=bs, convert_to_tensor=True,
                       normalize_embeddings=False, show_progress_bar=False)
    emb = torch.nn.functional.normalize(emb[:, :dim].float(), dim=1).half()
    out = torch.empty_like(emb)
    out[torch.as_tensor(order, device=emb.device)] = emb
    return out


def knn(Q, P, k, qchunk=2048):
    k = min(k, P.shape[0])
    vals, idx = [], []
    for s in range(0, Q.shape[0], qchunk):
        sim = Q[s:s + qchunk] @ P.T
        v, i = torch.topk(sim, k, dim=1)
        vals.append(v.float().cpu())
        idx.append(i.cpu())
    return torch.cat(vals).numpy(), torch.cat(idx).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="val")
    ap.add_argument("--model", default="models/Qwen3-Embedding-0.6B")
    ap.add_argument("--norm", default="trn")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--prompt", action="store_true", help="prepend the instruction (about 2x slower)")
    a = ap.parse_args()
    mname = Path(a.model).name
    t0 = time.time()
    model = SentenceTransformer(str(ROOT / a.model), device="cuda",
                                model_kwargs={"torch_dtype": torch.bfloat16},
                                tokenizer_kwargs={"padding_side": "left"})
    model.max_seq_length = 48 if not a.prompt else 64
    dfs = load_fold(a.fold, a.norm)
    edir = ROOT / "cache" / "emb"
    edir.mkdir(parents=True, exist_ok=True)
    embs = []
    for i, df in enumerate(dfs, 1):
        f = edir / f"{a.fold}_{mname}_d{a.dim}_{'p' if a.prompt else 'np'}_source{i}.npy"
        if f.exists():
            e = torch.from_numpy(np.load(f)).cuda()
        else:
            t = time.time()
            e = encode(model, df, a.dim, a.bs, a.prompt)
            np.save(f, e.cpu().numpy())
            print(f"encoded source{i}: {df.height:,} in {time.time() - t:.0f}s "
                  f"({df.height / (time.time() - t):,.0f}/s)", flush=True)
        embs.append(e)
    del model
    torch.cuda.empty_cache()
    parts = []
    s1 = dfs[0]
    for country in sorted(s1["country"].unique().to_list()):
        qm = (s1["country"] == country).to_numpy()
        Q = embs[0][torch.as_tensor(qm, device="cuda")]
        q_ids = s1["entity_id"].to_numpy()[qm]
        for df, E in zip(dfs[1:], embs[1:]):
            pm = (df["country"] == country).to_numpy()
            if pm.sum() == 0:
                continue
            P = E[torch.as_tensor(pm, device="cuda")]
            p_ids = df["entity_id"].to_numpy()[pm]
            v, idx = knn(Q, P, a.k)
            kk = v.shape[1]
            parts.append(pl.DataFrame({
                "s1": np.repeat(q_ids, kk), "cand": p_ids[idx.ravel()],
                "s_emb": v.ravel().astype(np.float32),
                "r_emb": np.tile(np.arange(1, kk + 1, dtype=np.uint16), len(q_ids))}))
        print(f"  knn {country}: {qm.sum():,} queries", flush=True)
    out = ROOT / "cache" / "cand" / f"{a.fold}_emb_{mname}{'_p' if a.prompt else ''}_k{a.k}.parquet"
    pl.concat(parts).write_parquet(out)
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
