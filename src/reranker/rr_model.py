"""Cross-encoder wrapper shared by train/score/bench: one `score(batch) -> logit` API for two architectures.

arch=qwen : Qwen3-Reranker (causal LM). score = logit_yes - logit_no at the last position (the model card's
            yes/no readout, i.e. the log-odds of P(yes) under a 2-way softmax). Left padding, logits_to_keep=1.
arch=bge  : bge-reranker-v2-m3 (XLMRobertaForSequenceClassification, 1 label). score = the classifier logit.
Both are trained with BCE-with-logits on that score, so the pretrained relevance head is reused, not re-initialised.
"""
import json
from pathlib import Path

import torch

from common import QWEN_INSTR, QWEN_PREFIX, QWEN_SUFFIX


class CrossEncoder:
    def __init__(self, arch, path, max_len=128, device="cuda", dtype=torch.bfloat16, lora=0, grad_ckpt=False,
                 tiny_random=False):
        from transformers import AutoTokenizer
        self.arch, self.max_len, self.device = arch, max_len, device
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left" if arch == "qwen" else "right")
        if arch == "qwen":
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
            self.yes = self.tok.convert_tokens_to_ids("yes")
            self.no = self.tok.convert_tokens_to_ids("no")
            self.pre = self.tok(QWEN_PREFIX, add_special_tokens=False)["input_ids"]
            self.suf = self.tok(QWEN_SUFFIX, add_special_tokens=False)["input_ids"]
        else:
            from transformers import AutoConfig, AutoModelForSequenceClassification
            if tiny_random:  # smoke test without the 2.3 GB weights: same tokenizer/arch, 2 tiny layers
                cfg = AutoConfig.from_pretrained(path)
                cfg.update(dict(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=2))
                self.model = AutoModelForSequenceClassification.from_config(cfg).to(dtype)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, dtype=dtype)
        if grad_ckpt:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        if lora:
            from peft import LoraConfig, get_peft_model
            tm = (["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if arch == "qwen"
                  else ["query", "key", "value", "dense"])
            cfg = LoraConfig(r=lora, lora_alpha=2 * lora, lora_dropout=0.05, target_modules=tm,
                             modules_to_save=None if arch == "qwen" else ["classifier"])
            self.model = get_peft_model(self.model, cfg)
            self.model.print_trainable_parameters()
        self.model.to(device)

    # ------------------------------------------------------------ tokenisation
    def encode(self, ta, tb):
        if self.arch == "qwen":
            ids = [self.pre + x + self.suf for x in self._qwen_bodies(ta, tb)]
            enc = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        else:
            enc = self.tok(list(ta), list(tb), truncation="longest_first", max_length=self.max_len,
                           padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def _qwen_bodies(self, ta, tb, truncate=True):
        """<Instruct>..<Query>: A\n<Document>: B with per-record 'longest first' truncation, so the document
        side (the candidate) is never silently chopped off at the end."""
        if not hasattr(self, "_head"):
            self._head = self.tok(f"<Instruct>: {QWEN_INSTR}\n<Query>: ", add_special_tokens=False)["input_ids"]
            self._mid = self.tok("\n<Document>: ", add_special_tokens=False)["input_ids"]
        room = self.max_len - len(self.pre) - len(self.suf) - len(self._head) - len(self._mid)
        qa = self.tok(list(ta), add_special_tokens=False)["input_ids"]
        qb = self.tok(list(tb), add_special_tokens=False)["input_ids"]
        out = []
        for x, z in zip(qa, qb):
            if truncate and len(x) + len(z) > room:
                lx = min(len(x), max(room // 2, room - len(z)))
                x, z = x[:lx], z[:room - lx]
            out.append(self._head + x + self._mid + z)
        return out

    def n_tokens(self, ta, tb):
        if self.arch == "qwen":
            return [len(x) + len(self.pre) + len(self.suf) for x in self._qwen_bodies(ta, tb, truncate=False)]
        return [len(x) for x in self.tok(list(ta), list(tb))["input_ids"]]

    # ------------------------------------------------------------ forward
    def score(self, enc):
        if self.arch == "qwen":
            out = self.model(**enc, logits_to_keep=1)
            lg = out.logits[:, -1, :]
            return (lg[:, self.yes] - lg[:, self.no]).float()
        return self.model(**enc).logits[:, 0].float()

    def save(self, out: Path, meta: dict):
        out.mkdir(parents=True, exist_ok=True)
        m = self.model
        if hasattr(m, "merge_and_unload"):  # LoRA: merge so scoring needs no peft
            m = m.merge_and_unload()
        m.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)  # bf16 on disk: half the size
        self.tok.save_pretrained(out)
        (out / "rr_meta.json").write_text(json.dumps({**meta, "arch": self.arch, "max_len": self.max_len}, indent=2))


def length_batches(lengths, bs, shuffle, seed=0, mega=64):
    """Index batches with little padding: sort by length inside mega-chunks of mega*bs, then shuffle batches."""
    import numpy as np
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
