"""Experiment A inference: score only the uncertain pairs with a fine-tuned (or zero-shot) cross-encoder.

python score_reranker.py --ckpt ckpt/rr_qwen06_full --pairs rr_stack --tag qwen06_full
python score_reranker.py --ckpt ckpt/rr_qwen06_full --pairs rr_test  --tag qwen06_full
Writes results/rr_<tag>_<pairs>.parquet with columns s1, cand, score (float32 logit). Resumable: finished
chunks are kept in results/parts/ and skipped on restart (safe on pre-emptible nodes).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, Timer, attach_text, bundle_dir, load_records, results_dir, set_env_caches  # noqa: E402
from rr_model import CrossEncoder, length_batches  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[1]))
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--ckpt", required=True, help="fine-tuned dir (has rr_meta.json) or a base model dir")
    ap.add_argument("--arch", default=None, help="needed only for a base model without rr_meta.json")
    ap.add_argument("--pairs", default="rr_test", help="bundle table: rr_stack | rr_test | rr_train")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny-random", action="store_true")
    a = ap.parse_args()
    R = Path(a.root)
    set_env_caches(R)
    T = Timer()
    B, RES = bundle_dir(a), results_dir(a)
    ck = Path(a.ckpt) if Path(a.ckpt).is_absolute() else (HERE / a.ckpt if (HERE / a.ckpt).exists() else R / a.ckpt)
    meta = json.loads((ck / "rr_meta.json").read_text()) if (ck / "rr_meta.json").exists() else {}
    arch = meta.get("arch", a.arch)
    text, fmt = meta.get("text", "raw"), meta.get("fmt", "lite")
    max_len = a.max_len or meta.get("max_len", 160 if arch == "qwen" else 128)
    dt = torch.bfloat16 if a.device.startswith("cuda") else torch.float32
    ce = CrossEncoder(arch, str(ck), max_len, a.device, dt, tiny_random=a.tiny_random)
    ce.model.eval()

    pairs = pl.read_parquet(B / f"{a.pairs}.parquet", columns=["s1", "cand"])
    if a.limit:
        pairs = pairs.head(a.limit)
    rec = load_records(B)
    parts_dir = RES / "parts" / f"{a.tag}_{a.pairs}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for c0 in range(0, pairs.height, a.chunk):
        pf = parts_dir / f"{c0:010d}.parquet"
        if pf.exists():
            continue
        d = attach_text(pairs.slice(c0, a.chunk), rec, text, fmt)
        ta, tb = d["ta"].to_list(), d["tb"].to_list()
        sc = np.zeros(len(ta), dtype=np.float32)
        with torch.inference_mode():
            for b in length_batches([len(x) + len(z) for x, z in zip(ta, tb)], a.bs, shuffle=False):
                sc[b] = ce.score(ce.encode([ta[i] for i in b], [tb[i] for i in b])).cpu().numpy()
        d.select("s1", "cand").with_columns(score=pl.Series(sc)).write_parquet(pf)
        n_done += len(ta)
        el = time.time() - T.t0
        print(f"chunk {c0:,}+{len(ta):,} done ({T()}, {n_done / el:,.0f} pairs/s)", flush=True)
    res = pl.concat([pl.read_parquet(f) for f in sorted(parts_dir.glob("*.parquet"))])
    outp = RES / f"rr_{a.tag}_{a.pairs}.parquet"
    res.write_parquet(outp)
    print(f"wrote {outp}: {res.height:,} pairs ({T()})")


if __name__ == "__main__":
    main()
