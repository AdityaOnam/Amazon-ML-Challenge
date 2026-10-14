"""R10 listwise select cross-encoder (bge-reranker-v2-m3 / XLM-R large backbone, Apache-2.0).

One sequence per contest (candidate + up to K competing S1s):
    <s> cand </s></s> (A) s1_a (B) s1_b ... (F) s1_f </s>
Option i's logit = the pretrained relevance head (dense -> tanh -> out_proj, reused, not re-initialised) applied
to the MEAN of the hidden states over option i's span (marker + record tokens). Mean-pooling reads the whole
option, so it does not depend on the model learning a marker-token readout from scratch; the "(A)".."(F)"
markers give each option a distinct identity for cross-option attention. The NONE logit is the same head on
the <s> token plus a learned bias. softmax over [none, opt_1..opt_K] -> sel_prob per option, p_none per contest.
Options are shuffled (training: fresh permutation per epoch; scoring: --tta permutations averaged), so the model
cannot learn the p-order the options were exported in.
Truncation: per-record "water-filling" cap so every record keeps its head (name first) inside --max-len.
"""
import json
from pathlib import Path

import numpy as np
import torch

MARKS = "ABCDEFGHIJ"


class SelectCE(torch.nn.Module):
    def __init__(self, path, max_len=384, device="cuda", dtype=torch.float32, tiny_random=False, grad_ckpt=False):
        super().__init__()
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        self.max_len, self.device = max_len, device
        self.tok = AutoTokenizer.from_pretrained(path)
        if tiny_random:  # smoke without weights: same tokenizer / architecture, 2 tiny layers
            cfg = AutoConfig.from_pretrained(path)
            cfg.update(dict(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=2,
                            num_labels=1))
            self.model = AutoModelForSequenceClassification.from_config(cfg)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1)
        self.model.to(dtype)
        meta = Path(path) / "sel_meta.json"
        nb = json.loads(meta.read_text()).get("none_bias", 0.0) if meta.exists() else 0.0
        self.none_bias = torch.nn.Parameter(torch.tensor([nb], dtype=torch.float32))
        if grad_ckpt:
            self.model.gradient_checkpointing_enable()
        self.marks = [self.tok(f" ({m})", add_special_tokens=False)["input_ids"] for m in MARKS]
        self.cls, self.sep, self.pad = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id
        self.to(device)

    # ------------------------------------------------------------ tokenisation
    def encode(self, cands, options):
        """cands: list[str]; options: list[list[str]] (1..K each). Returns input_ids, attention_mask, seg
        (token -> option index 1..K, 0 elsewhere), valid (B, K) mask."""
        K = max(len(o) for o in options)
        flat = list(cands) + [x for o in options for x in o]
        ids_all = self.tok(flat, add_special_tokens=False)["input_ids"]
        c_ids, pos = ids_all[:len(cands)], len(cands)
        rows, segs = [], []
        for b, opts in enumerate(options):
            o_ids = ids_all[pos:pos + len(opts)]
            pos += len(opts)
            recs = [c_ids[b]] + o_ids
            room = self.max_len - 4 - sum(len(self.marks[i]) for i in range(len(opts)))
            cap = _water_fill([len(r) for r in recs], max(room, 8 * len(recs)))
            ids = [self.cls] + recs[0][:cap] + [self.sep, self.sep]
            seg = [0] * len(ids)
            for i, r in enumerate(o_ids):
                piece = self.marks[i] + r[:cap]
                ids += piece
                seg += [i + 1] * len(piece)
            ids.append(self.sep)
            seg.append(0)
            rows.append(ids[:self.max_len])
            segs.append(seg[:self.max_len])
        L = max(len(r) for r in rows)
        inp = torch.full((len(rows), L), self.pad, dtype=torch.long)
        att = torch.zeros((len(rows), L), dtype=torch.long)
        sg = torch.zeros((len(rows), L), dtype=torch.long)
        for b, (r, s) in enumerate(zip(rows, segs)):
            inp[b, :len(r)] = torch.tensor(r)
            att[b, :len(r)] = 1
            sg[b, :len(s)] = torch.tensor(s)
        valid = torch.tensor([[i < len(o) for i in range(K)] for o in options])
        return {k: v.to(self.device) for k, v in dict(input_ids=inp, attention_mask=att, seg=sg, valid=valid).items()}

    # ------------------------------------------------------------ forward
    def _head(self, x):
        h = self.model.classifier
        if hasattr(h, "dense"):  # XLMRobertaClassificationHead: dropout -> dense -> tanh -> dropout -> out_proj
            x = h.dropout(x)
            x = torch.tanh(h.dense(x))
            x = h.dropout(x)
            return h.out_proj(x)[..., 0]
        return h(x)[..., 0]

    def forward(self, enc):
        """-> logits (B, 1+K): column 0 = none, columns 1..K = options (invalid options = -inf)."""
        hs = self.model.base_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        K = enc["valid"].shape[1]
        oh = (enc["seg"].unsqueeze(1) == torch.arange(1, K + 1, device=hs.device).view(1, K, 1)).to(hs.dtype)
        pooled = torch.bmm(oh, hs) / oh.sum(-1, keepdim=True).clamp(min=1)  # (B, K, H)
        opt = self._head(pooled).float()
        none = self._head(hs[:, 0]).float() + self.none_bias
        opt = opt.masked_fill(~enc["valid"], float("-inf"))
        return torch.cat([none.unsqueeze(1), opt], dim=1)

    def save(self, out: Path, meta: dict):
        out.mkdir(parents=True, exist_ok=True)
        self.model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
        self.tok.save_pretrained(out)
        (out / "sel_meta.json").write_text(json.dumps(
            {**meta, "none_bias": float(self.none_bias.item()), "max_len": self.max_len, "kind": "select"}, indent=2))


def _water_fill(lengths, budget):
    """Largest per-record cap c with sum(min(len, c)) <= budget (every record keeps its first c tokens)."""
    if sum(lengths) <= budget:
        return max(lengths)
    ls = sorted(lengths)
    used, n = 0, len(ls)
    for i, x in enumerate(ls):
        if used + x * (n - i) > budget:
            return max(1, (budget - used) // (n - i))
        used += x
    return ls[-1]


def permute(opts, rng):
    """opts: dict of equal-length lists -> same dict permuted, plus the permutation."""
    n = len(next(iter(opts.values())))
    perm = rng.permutation(n)
    return {k: [v[i] for i in perm] for k, v in opts.items()}, perm


def length_batches(lengths, bs, shuffle, seed=0, mega=64):
    n = len(lengths)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    L = np.asarray(lengths)
    batches = []
    for s in range(0, n, mega * bs):
        chunk = idx[s:s + mega * bs]
        chunk = chunk[np.argsort(L[chunk], kind="stable")]
        batches += [chunk[i:i + bs] for i in range(0, len(chunk), bs)]
    if shuffle:
        rng.shuffle(batches)
    return batches
