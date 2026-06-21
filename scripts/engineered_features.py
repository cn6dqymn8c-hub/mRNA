#!/usr/bin/env python3
"""
Cheap, hand-crafted sequence features for mRNA localization, to STACK with k-mer /
foundation-model embeddings via train_rnafm_multilabel.py --features.

Blocks (selectable as --features names):
  length     : [len, log1p(len)] — the 3'UTR-length CONFOUND; use as a baseline.
  engineered : length + GC + dinucleotide composition + localization-motif density.
  structure  : ViennaRNA MFE features (optional; needs the `RNA`/ViennaRNA package).

All operate on the ALREADY region-harmonized RNA sequence in genes['sequence']
(so for --region utr3 these are 3'UTR features automatically). Pure numpy except
`structure`, which is opt-in.
"""
from __future__ import annotations

from itertools import product

import numpy as np

_DINUC = ["".join(p) for p in product("ACGU", repeat=2)]

# Localization / RBP-associated motifs (RNA alphabet), reported as per-kb density.
_MOTIFS = {
    "ARE_AUUUA": "AUUUA",            # AU-rich element
    "CPE_UUUUAU": "UUUUAU",          # cytoplasmic polyadenylation element (CPEB)
    "polyU_UUUUU": "UUUUU",          # poly-pyrimidine / poly-U
    "GU_UGUGU": "UGUGU",             # GU-rich (CELF/RBFOX-adjacent)
    "Gquad_GGG": "GGG",              # G-quadruplex proxy
    "polyA_AAUAAA": "AAUAAA",        # canonical poly(A) signal
}


def _clean(s: object) -> str:
    return str(s).upper().replace("T", "U")


def _count_overlap(s: str, sub: str) -> int:
    n, start = 0, 0
    while True:
        i = s.find(sub, start)
        if i < 0:
            break
        n += 1
        start = i + 1
    return n


def length_features(seqs):
    L = np.array([len(_clean(s)) for s in seqs], dtype=np.float32)
    return np.stack([L, np.log1p(L)], axis=1), ["len", "log_len"]


def engineered_features(seqs, structure: bool = False):
    feats, names = [], []
    lf, ln = length_features(seqs)
    feats.append(lf); names += ln

    n_cols = 1 + len(_DINUC) + len(_MOTIFS)
    comp = np.zeros((len(seqs), n_cols), dtype=np.float32)
    cols = ["gc"] + [f"di_{d}" for d in _DINUC] + [f"mot_{k}" for k in _MOTIFS]
    didx = {d: i for i, d in enumerate(_DINUC)}
    motif_items = list(_MOTIFS.items())
    base = 1 + len(_DINUC)
    for r, s in enumerate(seqs):
        s = _clean(s)
        n = len(s)
        if n == 0:
            continue
        comp[r, 0] = (s.count("G") + s.count("C")) / n
        if n >= 2:
            dv = np.zeros(len(_DINUC), dtype=np.float32)
            for j in range(n - 1):
                k = didx.get(s[j:j + 2])
                if k is not None:
                    dv[k] += 1
            comp[r, 1:1 + len(_DINUC)] = dv / (n - 1)
        for m, (_key, sub) in enumerate(motif_items):
            comp[r, base + m] = _count_overlap(s, sub) / n * 1000.0  # per-kb
    feats.append(comp); names += cols

    if structure:
        sf, sn = structure_features(seqs)
        feats.append(sf); names += sn

    return np.concatenate(feats, axis=1).astype(np.float32), names


def structure_features(seqs, fold_cap: int = 600):
    """ViennaRNA MFE of (capped) sequence. fold_cap bounds compute on long inputs."""
    try:
        import RNA
    except Exception as e:
        raise SystemExit(
            "[error] --features structure needs the ViennaRNA python package "
            f"(`pip install ViennaRNA`). import error: {e}"
        )
    mfe = np.zeros((len(seqs), 2), dtype=np.float32)
    for r, s in enumerate(seqs):
        s = _clean(s)[:fold_cap] or "A"
        try:
            _, e = RNA.fold(s)
        except Exception:
            e = 0.0
        mfe[r, 0] = float(e)
        mfe[r, 1] = float(e) / max(len(s), 1)
    return mfe, ["mfe", "mfe_per_nt"]


def build_feature_block(name, seqs, kmer_k=4):
    """Return (matrix, tag) for a non-FM --features block. FM is handled by the
    caller (it needs the model + cache)."""
    if name == "length":
        X, _ = length_features(seqs)
        return X, "len"
    if name == "engineered":
        X, _ = engineered_features(seqs, structure=False)
        return X, "eng"
    if name == "structure":
        X, _ = structure_features(seqs)
        return X, "struct"
    if name == "kmer":
        from train_rnafm_multilabel import kmer_features
        return kmer_features(seqs, kmer_k), f"kmer{kmer_k}"
    raise SystemExit(f"[error] unknown --features block: {name}")
