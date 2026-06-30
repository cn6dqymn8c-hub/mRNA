#!/usr/bin/env python3
"""
General-LLM sequence encoder (e.g. DeepSeekMoE) adapted to neuronal mRNA
localization via QLoRA, as a CONTROLLED benchmark entry — same split / labels /
masked-BCE objective / val-early-stopping / evaluation as every other model, so its
numbers are directly comparable to k-mer, RNA-FM and the fusion model.

IMPORTANT CAVEAT (state this in the paper): the base model is pretrained on natural
language + code, NOT nucleotides, and we keep its original text BPE tokenizer. Any
"transfer" is therefore from a different modality; this entry tests, rather than
assumes, whether a general LLM helps. It is not expected to beat a nucleotide model
or even k-mer.

Interface mirrors finetune_model / train_task_specific:
    train_llm_encoder(genes, Y, classes, tr_idx, va_idx, te_idx, args,
                      sample_weight=None, label_mask=None) -> (prob_va, prob_te)

Requires (on the GPU machine): transformers, peft, bitsandbytes, accelerate.
A 16B MoE in 4-bit fits a single 24GB GPU; set CUDA_VISIBLE_DEVICES to a free card.
"""
from __future__ import annotations

import numpy as np


def _pool(last_hidden, attn_mask, mode):
    """last_hidden (B,T,H), attn_mask (B,T) -> (B,H). Causal-decoder friendly."""
    import torch
    m = attn_mask.unsqueeze(-1).to(last_hidden.dtype)            # (B,T,1)
    if mode == "last":
        # last non-pad token per row (right padding)
        idx = attn_mask.sum(dim=1).clamp(min=1).long() - 1       # (B,)
        return last_hidden[torch.arange(last_hidden.size(0)), idx]
    # mean over valid tokens
    return (last_hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def train_llm_encoder(genes, Y, classes, tr_idx, va_idx, te_idx, args,
                      sample_weight=None, label_mask=None):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_id = args.llm_encoder
    nC = len(classes)
    seqs = [str(s).upper() for s in genes["sequence"].tolist()]
    max_tokens = int(getattr(args, "llm_max_tokens", 1024))
    pool_mode = str(getattr(args, "llm_pool", "mean"))
    use_4bit = bool(getattr(args, "llm_4bit", True))
    use_lora = bool(getattr(args, "llm_lora", True))
    batch = int(getattr(args, "llm_batch", 4))
    accum = max(1, int(getattr(args, "llm_grad_accum", 8)))
    epochs = int(getattr(args, "llm_epochs", 3))
    patience = int(getattr(args, "llm_patience", 2))
    lr = float(getattr(args, "llm_lr", 1e-4))
    seed = int(getattr(args, "seed", 0))
    torch.manual_seed(seed)

    Y = np.asarray(Y, dtype=np.float32)
    label_mask = (np.ones_like(Y, dtype=np.float32) if label_mask is None
                  else np.asarray(label_mask, dtype=np.float32))
    if sample_weight is None:
        sample_weight = np.ones_like(Y, dtype=np.float32)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float32)
        if sample_weight.ndim == 1:
            sample_weight = np.repeat(sample_weight[:, None], nC, axis=1)

    # ---- tokenizer (KEPT as the model's original text BPE, per design) ----
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- base model (4-bit) ----
    compute_dtype = torch.bfloat16
    qcfg = None
    if use_4bit:
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=compute_dtype,
                                  bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, quantization_config=qcfg,
        torch_dtype=compute_dtype, device_map="auto", output_hidden_states=True,
    )
    base.config.use_cache = False
    if use_4bit:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:
        base.gradient_checkpointing_enable()

    if use_lora:
        tgt = getattr(args, "llm_target_modules", None)
        target_modules = ([t.strip() for t in tgt.split(",")] if tgt else "all-linear")
        lcfg = LoraConfig(r=int(getattr(args, "llm_lora_r", 16)),
                          lora_alpha=int(getattr(args, "llm_lora_alpha", 32)),
                          lora_dropout=0.05, bias="none",
                          target_modules=target_modules, task_type="FEATURE_EXTRACTION")
        base = get_peft_model(base, lcfg)
        base.print_trainable_parameters()

    device = next(base.parameters()).device
    hidden = int(base.config.hidden_size)
    head = nn.Linear(hidden, nC).to(device=device, dtype=torch.float32)

    known = label_mask[tr_idx].sum(axis=0)
    pos = (Y[tr_idx] * label_mask[tr_idx]).sum(axis=0)
    pos_weight = torch.tensor(np.clip((known - pos) / np.clip(pos, 1, None), 0.2, 5.0),
                              dtype=torch.float32, device=device)

    trainable = [p for p in base.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)

    over = int(np.array([len(s) for s in seqs]).max() > max_tokens * 6)
    if over:
        print(f"[llm] note: long transcripts are truncated to --llm-max-tokens={max_tokens} "
              "BPE tokens (5'->3').", flush=True)

    def embed_batch(jlist):
        texts = [seqs[j] for j in jlist]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_tokens)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = base(**enc)
        h = out.hidden_states[-1]                       # (B,T,H), compute_dtype
        return _pool(h, enc["attention_mask"], pool_mode).float()

    def run(idx, train):
        base.train(train); head.train(train)
        order = list(idx)
        if train:
            rng.shuffle(order)
        out = np.zeros((len(idx), nC), dtype=np.float32)
        pos_of = {int(j): p for p, j in enumerate(idx)}
        if train:
            opt.zero_grad(set_to_none=True)
        for bi in range(0, len(order), batch):
            jl = order[bi:bi + batch]
            with torch.set_grad_enabled(train):
                z = embed_batch(jl)
                logits = head(z)
                ys = torch.tensor(np.stack([Y[j] for j in jl]), device=device)
                ws = torch.tensor(np.stack([sample_weight[j] for j in jl]), device=device)
                ms = torch.tensor(np.stack([label_mask[j] for j in jl]), device=device)
                raw = F.binary_cross_entropy_with_logits(logits, ys, pos_weight=pos_weight,
                                                         reduction="none")
                eff = ms * ws
                loss = (raw * eff).sum() / eff.sum().clamp(min=1.0)
            if train:
                (loss / accum).backward()
                if (bi // batch + 1) % accum == 0:
                    opt.step(); opt.zero_grad(set_to_none=True)
            lg = logits.detach().float().cpu().numpy()
            for k, j in enumerate(jl):
                out[pos_of[int(j)]] = lg[k]
        if train:
            opt.step(); opt.zero_grad(set_to_none=True)
        return out

    def macro_auc(idx, prob):
        vals = []
        for j in range(nC):
            v = label_mask[idx, j].astype(bool)
            yt = Y[idx, j][v]
            if 0 < yt.sum() < len(yt):
                vals.append(roc_auc_score(yt, prob[v, j]))
        return float(np.mean(vals)) if vals else 0.0

    best_auc, best, stale = -1.0, None, 0
    for ep in range(epochs):
        run(tr_idx, True)
        va_prob = 1.0 / (1.0 + np.exp(-run(va_idx, False)))
        auc = macro_auc(va_idx, va_prob)
        print(f"[llm] epoch {ep + 1}/{epochs} val_macro_auc={auc:.4f}", flush=True)
        if auc > best_auc + 1e-6:
            best_auc, stale = auc, 0
            best = ({k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    {k: v.detach().cpu().clone()
                     for k, v in base.state_dict().items() if "lora" in k.lower()})
        else:
            stale += 1
            if stale >= patience:
                print(f"[llm] early stop after {stale} non-improving epochs."); break

    if best is not None:
        head.load_state_dict(best[0])
        if best[1]:
            base.load_state_dict(best[1], strict=False)
    va_prob = 1.0 / (1.0 + np.exp(-run(va_idx, False)))
    te_prob = 1.0 / (1.0 + np.exp(-run(te_idx, False)))
    print(f"[llm] best val_macro_auc={best_auc:.4f}")
    return va_prob, te_prob
