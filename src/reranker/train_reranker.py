"""Experiment A: fine-tune a cross-encoder on H_B hard pairs (bundle/rr_train.parquet).

python train_reranker.py --arch qwen --model models/Qwen3-Reranker-0.6B --out ckpt/rr_qwen06_full
python train_reranker.py --arch bge  --model models/bge-reranker-v2-m3  --out ckpt/rr_bge_full --lr 2e-5
python train_reranker.py --arch qwen ... --lora 16 --lr 2e-4                    (LoRA ablation)
Smoke (CPU): --device cpu --max-rows 256 --bs 8 --max-steps 4 --precision fp32
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, Timer, attach_text, bundle_dir, load_records, set_env_caches  # noqa: E402
from rr_model import CrossEncoder, length_batches  # noqa: E402


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = y.sum()
    nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))


@torch.no_grad()
def evaluate(ce, df, bs):
    ce.model.eval()
    ta, tb = df["ta"].to_list(), df["tb"].to_list()
    out = np.zeros(len(ta), dtype=np.float32)
    for b in length_batches([len(x) + len(z) for x, z in zip(ta, tb)], bs * 2, shuffle=False):
        with ce.autocast():
            out[b] = ce.score(ce.encode([ta[i] for i in b], [tb[i] for i in b])).cpu().numpy()
    ce.model.train()
    y = df["y"].to_numpy().astype(np.float32)
    p = 1 / (1 + np.exp(-out))
    ll = float(-np.mean(y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7)))
    lgb_p = df["p"].to_numpy()
    lgb_ll = float(-np.mean(y * np.log(lgb_p + 1e-7) + (1 - y) * np.log(1 - lgb_p + 1e-7)))
    return {"auc": auc(y, out), "logloss": ll, "acc@0": float(((out > 0) == (y > 0.5)).mean()),
            "lgb_auc": auc(y, lgb_p), "lgb_logloss": lgb_ll, "n": len(y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[1]))
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--arch", choices=["qwen", "bge"], required=True)
    ap.add_argument("--model", required=True, help="local model dir (relative to --root or absolute)")
    ap.add_argument("--out", required=True, help="checkpoint dir (relative to research/gpu_exp or absolute)")
    ap.add_argument("--text", choices=["raw", "clean"], default="raw")
    ap.add_argument("--fmt", choices=["lite", "ditto"], default="lite")
    ap.add_argument("--max-len", type=int, default=None, help="default 160 for qwen (72-token prompt), 128 for bge")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--lora", type=int, default=0, help="LoRA rank (0 = full fine-tune)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--precision", choices=["amp", "bf16", "fp32"], default="amp",
                    help="amp = fp32 master weights + bf16 autocast (default); bf16 = pure bf16 (less memory)")
    ap.add_argument("--hard-weight", type=float, default=1.0, help="loss weight of kind=hard pairs")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--holdout", type=float, default=0.02, help="fraction of S1s held out for monitoring")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny-random", action="store_true", help="bge smoke test without weights")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    R = Path(a.root)
    set_env_caches(R)
    torch.manual_seed(a.seed)
    T = Timer()
    B = bundle_dir(a)
    out = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    mpath = a.model if Path(a.model).is_absolute() else str(R / a.model)

    df = pl.read_parquet(B / "rr_train.parquet")
    if a.max_rows:
        df = df.head(a.max_rows)
    a.max_len = a.max_len or (160 if a.arch == "qwen" else 128)
    df = attach_text(df, load_records(B), a.text, a.fmt)
    ho = (df["s1"].hash(seed=99) % 10_000) < int(a.holdout * 10_000)
    dev, df = df.filter(ho), df.filter(~ho)
    if dev.height > 20_000:
        dev = dev.sample(20_000, seed=0)
    print(f"train {df.height:,} pairs (pos {df['y'].mean():.3f}), dev {dev.height:,} ({T()})", flush=True)

    wdtype = torch.float32 if a.precision in ("amp", "fp32") else torch.bfloat16
    ce = CrossEncoder(a.arch, mpath, a.max_len, a.device, wdtype, a.lora, a.grad_ckpt, a.tiny_random)
    use_amp = a.precision == "amp"
    dev_type = "cuda" if a.device.startswith("cuda") else "cpu"
    ce.autocast = lambda: torch.autocast(dev_type, dtype=torch.bfloat16, enabled=use_amp or a.precision == "bf16")
    ce.model.train()
    params = [p for p in ce.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98), eps=1e-6)

    ta, tb = df["ta"].to_list(), df["tb"].to_list()
    y = torch.tensor(df["y"].to_numpy().astype(np.float32))
    w = torch.tensor(np.where(df["kind"].to_numpy() == "hard", a.hard_weight, 1.0).astype(np.float32))
    lens = [len(x) + len(z) for x, z in zip(ta, tb)]
    n_ep = math.ceil(a.epochs)
    steps_per_ep = math.ceil(len(ta) / a.bs / a.grad_accum)
    total = a.max_steps or int(steps_per_ep * a.epochs)
    warm = max(1, int(a.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    print(f"{total:,} optimizer steps, bs {a.bs}x{a.grad_accum}, lr {a.lr}, {'LoRA r=' + str(a.lora) if a.lora else 'full FT'}"
          f", precision {a.precision}", flush=True)
    step, micro, seen, t_last, best, n_skip = 0, 0, 0, time.time(), None, 0
    log = []
    for ep in range(n_ep):
        for b in length_batches(lens, a.bs, shuffle=True, seed=a.seed + ep):
            enc = ce.encode([ta[i] for i in b], [tb[i] for i in b])
            with ce.autocast():
                s = ce.score(enc)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                s, y[b].to(s.device), weight=w[b].to(s.device)) / a.grad_accum
            loss.backward()
            micro += 1
            seen += len(b)
            if micro % a.grad_accum:
                continue
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(gn):  # a bad batch: drop this update instead of poisoning the weights
                opt.zero_grad(set_to_none=True)
                n_skip += 1
                print(f"  skipped non-finite update at step {step} (total skipped {n_skip})", flush=True)
                continue
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0 or step == 1:
                rate = seen / (time.time() - t_last + 1e-9)
                mem = torch.cuda.max_memory_allocated() / 2**30 if dev_type == "cuda" else 0
                print(f"step {step}/{total} loss {loss.item() * a.grad_accum:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{rate:,.0f} pairs/s  maxmem {mem:.1f} GiB ({T()})", flush=True)
                seen, t_last = 0, time.time()
            if step % a.eval_every == 0 or step == total:
                m = evaluate(ce, dev, a.bs)
                m["step"] = step
                log.append(m)
                print("DEV", m, flush=True)
            if step >= total:
                break
        if step >= total:
            break
    ce.save(out, {"text": a.text, "fmt": a.fmt, "args": vars(a), "dev_log": log})
    print(f"saved {out} ({T()})")


if __name__ == "__main__":
    main()
