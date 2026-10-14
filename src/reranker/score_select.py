"""R10 inference: score contests with a trained select model.

python score_select.py --ckpt ckpt/sel_B --set oof_A --tag sel_B      (model_B -> H_A options, stacker/honest eval)
python score_select.py --ckpt ckpt/sel_B --set test  --tag sel_B
python score_select.py --ckpt ckpt/sel_A --set oof_B --tag sel_A
python score_select.py --ckpt ckpt/sel_A --set test  --tag sel_A
Writes results/sel_<tag>_<set>.parquet, one row per (s1, cand) option:
  s1, cand, sel_prob (mean over --tta option permutations), score (= logit(sel_prob), float32), p_none, n_opt[, ha]
Resumable: finished chunks stay in results/parts/sel_<tag>_<set>/ and are skipped on restart.
The home server keeps only the honest half of each OOF file (assemble_select.py): ha=True rows from model_B,
ha=False rows from model_A.
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
from common import HERE, Timer, set_env_caches  # noqa: E402
from sel_model import SelectCE  # noqa: E402
from train_select import record_text, run_contests  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[1]))
    ap.add_argument("--bundle", default=None, help="default research/gpu_exp/bundle_sel")
    ap.add_argument("--results", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", required=True, help="oof_A | oof_B | test (bundle_sel/<set>.parquet)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--tta", type=int, default=2, help="option permutations averaged (1 = the export order shuffled once)")
    ap.add_argument("--chunk", type=int, default=20_000, help="contests per resumable part")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    R = Path(a.root)
    set_env_caches(R)
    T = Timer()
    B = Path(a.bundle) if a.bundle else HERE / "bundle_sel"
    RES = Path(a.results) if a.results else HERE / "results"
    RES.mkdir(parents=True, exist_ok=True)
    ck = Path(a.ckpt) if Path(a.ckpt).is_absolute() else (HERE / a.ckpt if (HERE / a.ckpt).exists() else R / a.ckpt)
    meta = json.loads((ck / "sel_meta.json").read_text())
    k = meta.get("k", 6)
    dt = torch.bfloat16 if a.device.startswith("cuda") else torch.float32
    model = SelectCE(str(ck), a.max_len or meta.get("max_len", 384), a.device, dt)
    model.eval()
    dev_type = "cuda" if a.device.startswith("cuda") else "cpu"
    autocast = lambda: torch.autocast(dev_type, dtype=torch.bfloat16, enabled=dev_type == "cuda")  # noqa: E731

    df = pl.read_parquet(B / f"{a.set}.parquet")
    if a.limit:
        df = df.head(a.limit)
    lists = [c for c in ("s1", "p", "y", "ha") if c in df.columns]
    df = df.with_columns([pl.col(c).list.head(k) for c in lists])
    txt = record_text(B)
    parts = RES / "parts" / f"sel_{a.tag}_{a.set}"
    parts.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for c0 in range(0, df.height, a.chunk):
        pf = parts / f"{c0:010d}.parquet"
        if pf.exists():
            continue
        d = df.slice(c0, a.chunk)
        acc = None
        with torch.inference_mode():
            for t in range(a.tta):
                pr = run_contests(model, d, txt, a.bs, autocast, rng=np.random.default_rng(1000 + t))
                acc = pr if acc is None else [x + y for x, y in zip(acc, pr)]
        acc = [x / a.tta for x in acc]
        sel = pl.Series("sel_prob", [x[1:].tolist() for x in acc], dtype=pl.List(pl.Float32))
        pn = pl.Series("p_none", [float(x[0]) for x in acc], dtype=pl.Float32)
        long = (d.select(["cand"] + [c for c in ("s1", "ha") if c in d.columns])
                 .with_columns(sel, pn, n_opt=pl.col("s1").list.len().cast(pl.Int8))
                 .explode([c for c in ("s1", "ha") if c in d.columns] + ["sel_prob"]))
        eps = 1e-6
        long = long.with_columns(score=((pl.col("sel_prob").clip(eps, 1 - eps)) / (1 - pl.col("sel_prob").clip(eps, 1 - eps)))
                                 .log().cast(pl.Float32))
        long.select(["s1", "cand", "sel_prob", "score", "p_none", "n_opt"] + (["ha"] if "ha" in long.columns else [])) \
            .write_parquet(pf)
        n_done += d.height
        print(f"chunk {c0:,}+{d.height:,} contests done ({T()}, {n_done / (time.time() - T.t0):,.1f} contests/s)", flush=True)
    res = pl.concat([pl.read_parquet(f) for f in sorted(parts.glob("*.parquet"))])
    outp = RES / f"sel_{a.tag}_{a.set}.parquet"
    res.write_parquet(outp)
    print(f"wrote {outp}: {res.height:,} options of {res['cand'].n_unique():,} contests ({T()})")


if __name__ == "__main__":
    main()
