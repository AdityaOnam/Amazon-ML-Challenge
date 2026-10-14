"""R10: fine-tune the listwise select cross-encoder on one half's training contests (bundle_sel/train_<half>).

python train_select.py --half B --model models/bge-reranker-v2-m3 --out ckpt/sel_B      (model_B: scores oof_A + test)
python train_select.py --half A --model models/bge-reranker-v2-m3 --out ckpt/sel_A      (model_A: scores oof_B + test)
Smoke (CPU): --device cpu --max-rows 256 --bs 4 --max-steps 4 --tiny-random --model smoke/bge-reranker-v2-m3-tok
Loss: softmax cross-entropy over [none, options] (one-to-one holds exactly, so the target is one option or none;
several true options would get a uniform soft target). Held-out 2% of S1s (every contest touching them) = dev:
logs contest accuracy, option AUC and log-loss next to the stacked-p baseline (argmax p if max p >= 0.5, else none).
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
from common import HERE, Timer, serialise, set_env_caches  # noqa: E402
from sel_model import SelectCE, length_batches  # noqa: E402


def auc(y, s):
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = y.sum()
    nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))


def record_text(bundle: Path, text="raw", fmt="lite"):
    r = pl.read_parquet(bundle / "records.parquet")
    nc, ac = ("business_name", "business_address")
    return dict(zip(r["entity_id"].to_list(),
                    [serialise(n, a, c, fmt) for n, a, c in zip(r[nc].to_list(), r[ac].to_list(), r["country"].to_list())]))


def targets(ys):
    """list of per-contest bool lists -> (B, 1+K) soft target, column 0 = none."""
    K = max(len(y) for y in ys)
    t = np.zeros((len(ys), K + 1), dtype=np.float32)
    for b, y in enumerate(ys):
        y = np.asarray(y, dtype=np.float32)
        if y.sum() > 0:
            t[b, 1:1 + len(y)] = y / y.sum()
        else:
            t[b, 0] = 1.0
    return torch.tensor(t)


def ce_loss(logits, t):
    lp = torch.log_softmax(logits, dim=1)
    return -torch.where(t > 0, t * lp, torch.zeros_like(lp)).sum(1)


def run_contests(model, df, txt, bs, autocast, rng=None):
    """Score contests (optionally with shuffled option order) -> list of per-contest prob arrays [none, opts in
    ORIGINAL order]."""
    s1s, cands = df["s1"].to_list(), df["cand"].to_list()
    out = [None] * len(s1s)
    lens = [sum(len(txt[x]) for x in s) + len(txt[c]) for s, c in zip(s1s, cands)]
    for b in length_batches(lens, bs, shuffle=False):
        perms = [rng.permutation(len(s1s[i])) if rng is not None else np.arange(len(s1s[i])) for i in b]
        enc = model.encode([txt[cands[i]] for i in b], [[txt[s1s[i][j]] for j in p] for i, p in zip(b, perms)])
        with autocast():
            pr = torch.softmax(model(enc), dim=1).float().cpu().numpy()
        for row, i, p in zip(pr, b, perms):
            o = np.zeros(len(p) + 1, dtype=np.float32)
            o[0] = row[0]
            o[1 + p] = row[1:1 + len(p)]
            out[i] = o
    return out


@torch.no_grad()
def evaluate(model, dev, txt, bs, autocast):
    model.eval()
    probs = run_contests(model, dev, txt, bs, autocast, rng=np.random.default_rng(123))
    model.train()
    ys, ps = dev["y"].to_list(), dev["p"].to_list()
    hard = dev["hard"].to_numpy()
    acc, bacc, ll, oy, os_, op = [], [], [], [], [], []
    for pr, y, p in zip(probs, ys, ps):
        y = np.asarray(y, dtype=bool)
        tgt = 1 + int(np.argmax(y)) if y.any() else 0
        acc.append(int(np.argmax(pr)) == tgt)
        bp = 1 + int(np.argmax(p)) if max(p) >= 0.5 else 0
        bacc.append(bp == tgt)
        ll.append(-math.log(max(pr[tgt], 1e-7)))
        oy += y.tolist()
        os_ += pr[1:].tolist()
        op += list(p)
    acc, bacc, oy = np.array(acc), np.array(bacc), np.array(oy, dtype=np.int8)
    return {"n": len(probs), "acc": float(acc.mean()), "acc_hard": float(acc[hard].mean()) if hard.any() else None,
            "p_acc": float(bacc.mean()), "p_acc_hard": float(bacc[hard].mean()) if hard.any() else None,
            "logloss": float(np.mean(ll)), "opt_auc": auc(oy, np.array(os_)), "p_opt_auc": auc(oy, np.array(op)),
            "n_hard": int(hard.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[1]))
    ap.add_argument("--bundle", default=None, help="default research/gpu_exp/bundle_sel")
    ap.add_argument("--half", choices=["A", "B"], required=True, help="train on train_<half>.parquet")
    ap.add_argument("--model", required=True, help="base model dir (relative to --root or absolute)")
    ap.add_argument("--out", required=True, help="checkpoint dir (relative to research/gpu_exp or absolute)")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--k", type=int, default=6, help="max options fed per contest (bundle has <= 6)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3, help="lr of the none bias")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--hard-weight", type=float, default=1.5, help="loss weight of hard contests (p2 > 0.05)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--precision", choices=["amp", "bf16", "fp32"], default="amp",
                    help="amp = fp32 master weights + bf16 autocast (default)")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--holdout", type=float, default=0.02, help="fraction of S1s held out (with their contests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny-random", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    R = Path(a.root)
    set_env_caches(R)
    torch.manual_seed(a.seed)
    T = Timer()
    B = Path(a.bundle) if a.bundle else HERE / "bundle_sel"
    out = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    mpath = a.model if Path(a.model).is_absolute() else str(R / a.model)

    df = pl.read_parquet(B / f"train_{a.half}.parquet")
    if a.max_rows:
        df = df.head(a.max_rows)
    df = df.with_columns([pl.col(c).list.head(a.k) for c in ("s1", "y", "p")])
    ho_s1 = df.select(pl.col("s1").explode()).unique().filter((pl.col("s1").hash(seed=99) % 10_000) < int(a.holdout * 10_000))
    is_dev = df["s1"].list.eval(pl.element().is_in(ho_s1["s1"].implode())).list.any()
    dev, df = df.filter(is_dev), df.filter(~is_dev)
    if dev.height > 5_000:
        dev = dev.sample(5_000, seed=0)
    txt = record_text(B)
    print(f"half {a.half}: train {df.height:,} contests (hard {df['hard'].mean():.3f}, true {(df['y'].list.sum() > 0).mean():.3f}, "
          f"{df['s1'].list.len().mean():.2f} options), dev {dev.height:,} ({T()})", flush=True)

    wdtype = torch.float32 if a.precision in ("amp", "fp32") else torch.bfloat16
    model = SelectCE(mpath, a.max_len, a.device, wdtype, a.tiny_random, a.grad_ckpt)
    dev_type = "cuda" if a.device.startswith("cuda") else "cpu"
    autocast = lambda: torch.autocast(dev_type, dtype=torch.bfloat16, enabled=a.precision != "fp32")  # noqa: E731
    model.train()
    body = [p for n, p in model.named_parameters() if p.requires_grad and n != "none_bias"]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": [model.none_bias], "lr": a.head_lr, "weight_decay": 0.0}],
                            lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98), eps=1e-6)
    s1s, cands, ys = df["s1"].to_list(), df["cand"].to_list(), df["y"].to_list()
    w = torch.tensor(np.where(df["hard"].to_numpy(), a.hard_weight, 1.0).astype(np.float32))
    lens = [sum(len(txt[x]) for x in s) + len(txt[c]) for s, c in zip(s1s, cands)]
    n_ep = math.ceil(a.epochs)
    steps_per_ep = math.ceil(len(s1s) / a.bs / a.grad_accum)
    total = a.max_steps or max(1, int(steps_per_ep * a.epochs))
    warm = max(1, int(a.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    print(f"{total:,} optimizer steps, bs {a.bs}x{a.grad_accum}, lr {a.lr}, max_len {a.max_len}, precision {a.precision}",
          flush=True)
    m0 = evaluate(model, dev, txt, a.bs * 2, autocast) if dev.height else {}
    print("DEV step 0", m0, flush=True)
    log = [dict(m0, step=0)]
    step, micro, seen, t_last, n_skip = 0, 0, 0, time.time(), 0
    rng = np.random.default_rng(a.seed)
    for ep in range(n_ep):
        for b in length_batches(lens, a.bs, shuffle=True, seed=a.seed + ep):
            perms = [rng.permutation(len(s1s[i])) for i in b]  # fresh option order every time a contest is seen
            enc = model.encode([txt[cands[i]] for i in b], [[txt[s1s[i][j]] for j in p] for i, p in zip(b, perms)])
            t = targets([[ys[i][j] for j in p] for i, p in zip(b, perms)]).to(a.device)
            with autocast():
                logits = model(enc)
            loss = (ce_loss(logits.float(), t) * w[b].to(a.device)).sum() / w[b].sum().to(a.device) / a.grad_accum
            loss.backward()
            micro += 1
            seen += len(b)
            if micro % a.grad_accum:
                continue
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
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
                      f"{rate:,.1f} contests/s  maxmem {mem:.1f} GiB ({T()})", flush=True)
                seen, t_last = 0, time.time()
            if dev.height and (step % a.eval_every == 0 or step == total):
                m = evaluate(model, dev, txt, a.bs * 2, autocast)
                m["step"] = step
                log.append(m)
                print("DEV", m, flush=True)
            if step >= total:
                break
        if step >= total:
            break
    model.save(out, {"half": a.half, "text": "raw", "fmt": "lite", "k": a.k, "args": vars(a), "dev_log": log,
                     "n_skipped": n_skip})
    print(f"saved {out} ({T()})")


if __name__ == "__main__":
    main()
