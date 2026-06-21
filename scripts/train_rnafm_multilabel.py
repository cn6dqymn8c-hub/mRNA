#!/usr/bin/env python3
"""
RNA-FM / mRNA-FM / DNABERT-2 multi-label neuronal mRNA subcellular-localization
benchmark pipeline (frozen embeddings, end-to-end fine-tune, or purpose-built
task-specific architectures), all sharing one data/split/label/eval path.

Region routing: only rows explicitly marked full-length (cdna/transcript/mrna)
are GTF-extracted; every other row (native isoform 3'UTR, ALE interval, blank
sequence_type) is assumed already in the requested region and passes through.

Feature families (one controlled comparison, vary only the encoder):
  --baseline kmer                         bag-of-kmers
  --model-dir <multimolecule|dnabert>     frozen embeddings + logistic/mlp head
  --model-dir <...> --finetune            end-to-end encoder fine-tune
  --arch rnatracker|dm3loc                purpose-built localization net (one-hot)

Quick smoke test:
    python scripts/train_rnafm_multilabel.py --selftest --model-dir ./rnafm
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Label harmonization
# ----------------------------------------------------------------------------
ASSAY_LABELS = {"Ribosome", "Cytoplasm"}
# Deterministic order when assay labels are promoted to target classes
# (--keep-assay-labels). These are a translation/fractionation axis, distinct
# from the anatomical compartments; report them separately.
ASSAY_LABEL_ORDER = ["Ribosome", "Cytoplasm"]

LABEL_ALIASES = {
    "cell_body": "Cell_body",
    "cellbody": "Cell_body",
    "soma": "Cell_body",
    "dendrite": "Dendrite",
    "dendritic": "Dendrite",
    "neuropil": "Neuropil",
    "synap": "Neuropil",
    "synapse": "Neuropil",
    "synaptic": "Neuropil",
    "axon": "Axon",
    "axonal": "Axon",
    "neurite": "Neurite",
    "neuronal_process": "Neurite",
    "neuronal_processes": "Neurite",
    "ribosome": "Ribosome",
    "ribosomes": "Ribosome",
    "cytoplasm": "Cytoplasm",
    "cytoplasmic": "Cytoplasm",
}

NEURITE_LABELS = {"Dendrite", "Neuropil", "Axon", "Neurite"}
SOMA_LABELS = {"Cell_body"}
FINE_LABELS = ["Cell_body", "Dendrite", "Neuropil", "Axon", "Neurite"]


def canonical_label(value: object) -> str:
    raw = str(value).strip()
    key = raw.casefold().replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return LABEL_ALIASES.get(key, raw)


def split_location(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).replace(";", ",")
    out, seen = [], set()
    for part in text.split(","):
        p = canonical_label(part)
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def clean_seq(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).upper().replace("T", "U")


def collect_observed_labels(df: pd.DataFrame) -> set[str]:
    labels: set[str] = set()
    for value in df["location"]:
        labels.update(split_location(value))
    return labels


# ----------------------------------------------------------------------------
# sequence_type classification (drives region routing)
# ----------------------------------------------------------------------------
def seqtype_is_fulllength(st: str) -> bool:
    """True only for explicit full-length markers (cdna/transcript/mrna/full).
    A blank sequence_type returns False here and is handled separately in
    apply_region (defaults to extraction, overridable via --native-region-sources)."""
    s = str(st).lower().strip()
    return ("cdna" in s) or ("transcript" in s) or ("mrna" in s) or ("full" in s)


def seqtype_is_cds(st: str) -> bool:
    s = str(st).lower()
    return ("cds" in s) or ("orf" in s) or ("coding_sequence" in s)


# ----------------------------------------------------------------------------
# Data loading and representative selection
# ----------------------------------------------------------------------------
def select_representative_rows(big: pd.DataFrame) -> pd.DataFrame:
    big = big.copy()
    if "transcript_is_canonical" not in big.columns:
        big["transcript_is_canonical"] = np.nan
    if "transcript_id" not in big.columns:
        big["transcript_id"] = ""
    big["_canon"] = big["transcript_is_canonical"].fillna(0).astype(float)
    big["_len"] = big["sequence"].astype(str).str.len()
    big = big.sort_values(
        ["source", "species", "gene_name", "_canon", "_len"],
        ascending=[True, True, True, False, False],
        kind="mergesort",
    )
    return big.drop_duplicates(subset=["source", "species", "gene_name"], keep="first").copy()


def load_representative_table(input_dir: Path) -> pd.DataFrame:
    return select_representative_rows(load_all_rows(input_dir))


def build_utr_length_map(gtf_paths) -> tuple[dict, dict]:
    import gzip, re
    tx_re = re.compile(r'transcript_id "([^"]+)"')
    L5, L3 = {}, {}
    for p in gtf_paths:
        if not os.path.exists(p):
            print(f"[gtf] missing, skipped: {p}")
            continue
        op = gzip.open(p, "rt") if str(p).endswith(".gz") else open(p)
        with op as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 9 or f[2] not in ("five_prime_utr", "three_prime_utr"):
                    continue
                m = tx_re.search(f[8])
                if not m:
                    continue
                tid = m.group(1)
                length = int(f[4]) - int(f[3]) + 1
                d = L5 if f[2] == "five_prime_utr" else L3
                d[tid] = d.get(tid, 0) + length
    print(f"[gtf] UTR lengths parsed: 5'UTR {len(L5)} tx, 3'UTR {len(L3)} tx")
    return L5, L3


def seqtype_is_native_utr3(st: str) -> bool:
    """Sequence already IS a 3'UTR-level region (isoform 3'UTR / spliced 3'UTR /
    reporter fragment / ALE interval). Passes through unchanged under utr3."""
    s = str(st).lower()
    return (("3utr" in s) or ("utr3" in s) or ("three_prime" in s)
            or ("ale" in s) or ("genomic_interval" in s) or ("last_exon" in s))


def apply_region(rep: pd.DataFrame, region: str, L5map: dict, L3map: dict,
                 native_sources=None) -> pd.DataFrame:
    """Harmonize each row to `region`, routed by sequence_type + an explicit
    per-source override.

    A BLANK sequence_type is AMBIGUOUS: it can mean a native 3'UTR (e.g. Andreassi
    isoform 3'UTRs) OR a full-length bulk cDNA representative (e.g. rows in a merged
    bulk+isoform file). To avoid silently feeding a full-length transcript into a
    3'UTR experiment, a blank/unknown sequence_type DEFAULTS to GTF-extraction
    (i.e. treated as full-length). Use ``native_sources`` (CLI --native-region-sources)
    to name sources whose rows are already in the requested region and must pass
    through untouched. Blank rows that were routed to extraction are reported so
    misclassification is visible, not silent.
    """
    if region == "full":
        return rep
    native = {str(s).strip().lower() for s in (native_sources or []) if str(s).strip()}

    def is_native_src(src):
        srcl = str(src).strip().lower()
        return any(tok in srcl for tok in native) if native else False

    rep = rep.copy()
    st_col = (rep["sequence_type"].astype(str) if "sequence_type" in rep.columns
              else pd.Series([""] * len(rep), index=rep.index))
    src_col = (rep["source"].astype(str) if "source" in rep.columns
               else pd.Series([""] * len(rep), index=rep.index))
    seqs = []
    n_pass = n_extract = n_missing = n_bad = n_wrongtype = 0
    blank_extracted = {}

    for src, st, tid_full, s in zip(
        src_col, st_col, rep["transcript_id"].astype(str), rep["sequence"].astype(str)
    ):
        n = len(s)
        if region == "utr3":
            if is_native_src(src) or seqtype_is_native_utr3(st):
                seqs.append(s); n_pass += 1; continue
            blank = str(st).strip() == ""
            if not (seqtype_is_fulllength(st) or blank):
                seqs.append(None); n_wrongtype += 1; continue   # e.g. cds-only row
            if blank:
                blank_extracted[str(src)] = blank_extracted.get(str(src), 0) + 1
            tid = tid_full.split(".")[0]
            l3 = L3map.get(tid)
            if tid == "" or l3 is None:
                seqs.append(None); n_missing += 1; continue
            if l3 <= 0 or l3 >= n:
                seqs.append(None); n_bad += 1; continue
            seqs.append(s[n - l3:]); n_extract += 1
        else:  # cds
            if seqtype_is_cds(st) or (is_native_src(src) and seqtype_is_cds(st)):
                seqs.append(s); n_pass += 1; continue
            if not seqtype_is_fulllength(st):
                seqs.append(None); n_wrongtype += 1; continue   # 3'UTR/ALE/blank has no usable CDS
            tid = tid_full.split(".")[0]
            l5 = L5map.get(tid)
            l3 = L3map.get(tid)
            if tid == "" or l5 is None or l3 is None:
                seqs.append(None); n_missing += 1; continue
            cds_len = n - l5 - l3
            if cds_len < 3 or (l5 + l3) >= n:
                seqs.append(None); n_bad += 1; continue
            seqs.append(s[l5: n - l3]); n_extract += 1

    rep["sequence"] = seqs
    kept = rep["sequence"].notna()
    print(
        f"[region={region}] kept {int(kept.sum())}/{len(rep)} "
        f"(native_passthrough={n_pass}, gtf_extracted={n_extract}, "
        f"not_in_gtf={n_missing}, bad_coords={n_bad}, wrong_seqtype={n_wrongtype})"
    )
    if blank_extracted:
        print(
            f"[region={region}] blank sequence_type -> GTF-extraction (treated as "
            f"full-length): {blank_extracted}. If any of these sources are native "
            f"{region} sequences, pass --native-region-sources to keep them untouched."
        )
    return rep[kept].copy()


def build_gene_dataset(rep: pd.DataFrame, keep_assay: bool) -> pd.DataFrame:
    from collections import Counter

    def filt(labels):
        return labels if keep_assay else [x for x in labels if x not in ASSAY_LABELS]

    rows = []
    rep = rep.copy()
    if "_canon" not in rep.columns:
        rep["_canon"] = rep.get(
            "transcript_is_canonical", pd.Series(0, index=rep.index)
        ).fillna(0).astype(float)
    rep["_len"] = rep["sequence"].astype(str).str.len()

    for (species, gene), g in rep.groupby(["species", "gene_name"], sort=False):
        per_source_sets, label_source_counts, src_names = [], Counter(), []
        n_neur = n_soma = 0
        for src, sg in g.groupby("source", sort=False):
            s: set[str] = set()
            for v in sg["location"]:
                s.update(filt(split_location(v)))
            per_source_sets.append(s)
            src_names.append(str(src))
            for lab in s:
                label_source_counts[lab] += 1
            if s & NEURITE_LABELS:
                n_neur += 1
            if s & SOMA_LABELS:
                n_soma += 1

        n_sources = len(per_source_sets)
        union = sorted(set().union(*per_source_sets)) if per_source_sets else []
        denom = n_neur + n_soma
        neurite_frac = (n_neur / denom) if denom else np.nan
        best = g.sort_values(
            ["_canon", "_len"], ascending=[False, False], kind="mergesort"
        ).iloc[0]
        rows.append({
            "species": species,
            "gene_name": gene,
            "sequence": clean_seq(best["sequence"]),
            "seq_len": int(best["_len"]),
            "labels_union": union,
            "label_source_counts": dict(label_source_counts),
            "sources": src_names,
            "n_sources": n_sources,
            "neurite_frac": neurite_frac,
            "binary_conflict": bool(n_neur > 0 and n_soma > 0),
        })
    return pd.DataFrame(rows)


def load_all_rows(input_dir: Path) -> pd.DataFrame:
    """Load every transcript row (no gene collapse). Carries sequence_type so
    region harmonization can tell full-length cdna from native-region rows."""
    frames = []
    for path in sorted(glob.glob(str(input_dir / "*.csv"))):
        df = pd.read_csv(path, low_memory=False)
        need = {"source", "species", "gene_name", "location", "sequence"}
        if not need.issubset(df.columns):
            print(f"[skip] {os.path.basename(path)} missing {need - set(df.columns)}")
            continue
        if "transcript_is_canonical" not in df.columns:
            df["transcript_is_canonical"] = np.nan
        if "transcript_id" not in df.columns:
            df["transcript_id"] = ""
        if "sequence_type" not in df.columns:
            df["sequence_type"] = ""
        frames.append(
            df[list(need) + ["transcript_is_canonical", "transcript_id", "sequence_type"]]
        )
    if not frames:
        raise SystemExit(f"No usable CSVs in {input_dir}")
    big = pd.concat(frames, ignore_index=True)
    big = big[big["sequence"].notna() & (big["sequence"].astype(str).str.len() > 0)].copy()
    big["sequence"] = big["sequence"].map(clean_seq)
    big["sequence_type"] = big["sequence_type"].fillna("").astype(str)
    big["_canon"] = big["transcript_is_canonical"].fillna(0).astype(float)
    return big


def build_sample_dataset(big: pd.DataFrame, sample_level: str, keep_assay: bool) -> pd.DataFrame:
    from collections import Counter

    def filt(labels):
        return labels if keep_assay else [x for x in labels if x not in ASSAY_LABELS]

    big = big.copy()
    big["_len"] = big["sequence"].astype(str).str.len()
    if "_canon" not in big.columns:
        big["_canon"] = (big["transcript_is_canonical"].fillna(0).astype(float)
                         if "transcript_is_canonical" in big.columns else 0.0)

    if sample_level in ("source_gene", "sequence_union"):
        unit_keys = ["source", "species", "gene_name"]
    elif sample_level in ("isoform", "isoform_sequence_union"):
        # Native 3'UTR isoform sources have no transcript_id, so the distinct
        # SEQUENCE is the isoform identity.
        unit_keys = ["source", "species", "gene_name", "transcript_id", "sequence"]
    else:
        raise ValueError(sample_level)

    units = []
    for _, g in big.groupby(unit_keys, sort=False, dropna=False):
        labs = set()
        for v in g["location"]:
            labs.update(filt(split_location(v)))
        best = g.sort_values(["_canon", "_len"], ascending=[False, False]).iloc[0]
        units.append({"source": best["source"], "species": best["species"],
                      "gene_name": best["gene_name"], "sequence": clean_seq(best["sequence"]),
                      "_len": int(best["_len"]), "labset": labs})
    udf = pd.DataFrame(units)

    if sample_level in ("sequence_union", "isoform_sequence_union"):
        groups = udf.groupby("sequence", sort=False)
    else:
        udf["_g"] = np.arange(len(udf))
        groups = udf.groupby("_g", sort=False)

    out = []
    for _, g in groups:
        per_source_sets, label_source_counts = {}, Counter()
        for src, labset in zip(g["source"], g["labset"]):
            per_source_sets.setdefault(src, set()).update(labset)
        for s in per_source_sets.values():
            for lab in s:
                label_source_counts[lab] += 1
        n_neur = sum(1 for s in per_source_sets.values() if s & NEURITE_LABELS)
        n_soma = sum(1 for s in per_source_sets.values() if s & SOMA_LABELS)
        union = sorted(set().union(*per_source_sets.values())) if per_source_sets else []
        denom = n_neur + n_soma
        best = g.sort_values("_len", ascending=False).iloc[0]
        out.append({
            "species": best["species"], "gene_name": best["gene_name"],
            "sequence": best["sequence"], "seq_len": int(best["_len"]),
            "labels_union": union, "label_source_counts": dict(label_source_counts),
            "sources": sorted(per_source_sets.keys()),
            "n_sources": len(per_source_sets),
            "neurite_frac": (n_neur / denom) if denom else np.nan,
            "binary_conflict": bool(n_neur > 0 and n_soma > 0),
        })
    print(f"[sample-level={sample_level}] {len(big)} rows -> {len(udf)} units -> {len(out)} samples")
    return pd.DataFrame(out)


def apply_label_scheme(
    df: pd.DataFrame,
    scheme: str,
    min_support: int,
    label_agg: str,
    min_label_sources: int,
    min_sources: int = 1,
    source_label_sets: dict[str, set[str]] | None = None,
    use_source_mask: bool = False,
    keep_assay: bool = False,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    df = df.copy()
    if min_sources > 1:
        before = len(df)
        df = df[df["n_sources"] >= min_sources].copy()
        print(
            f"[min_sources>={min_sources}] kept {len(df)}/{before} samples "
            "(multi-source corroborated)"
        )

    if scheme == "soma_vs_neurite":
        frac = df["neurite_frac"]
        df = df[frac.notna()].copy()
        frac = df["neurite_frac"].to_numpy(dtype=float)

        if label_agg == "consensus":
            keep = (frac == 0) | (frac == 1)
            df = df[keep].copy()
            frac = frac[keep]
            y = (frac == 1).astype(int)
            w = np.ones(len(df), dtype=np.float32)
        elif label_agg == "majority":
            y = (frac > 0.5).astype(int)
            w = np.ones(len(df), dtype=np.float32)
        elif label_agg == "soft":
            y = (frac >= 0.5).astype(int)
            w = np.abs(2 * frac - 1).astype(np.float32)
            w = np.where(w == 0, 1e-3, w)
        elif label_agg == "union":
            y = (frac > 0).astype(int)
            w = np.ones(len(df), dtype=np.float32)
        else:
            raise ValueError(label_agg)

        df["target"] = [[int(v)] for v in y]
        df["sample_weight"] = w
        return df, ["is_neurite"], w, np.ones((len(df), 1), dtype=np.float32)

    classes_all = list(FINE_LABELS)
    if keep_assay:
        # Promote assay-readout labels (Ribosome/Cytoplasm) to extra target classes.
        # Per-label source-mask keeps them honest (only sources that ran the assay
        # contribute negatives); they live on a different (translation/fractionation)
        # axis than the anatomical compartments and should be reported separately.
        classes_all = classes_all + [c for c in ASSAY_LABEL_ORDER if c not in classes_all]
    n = len(df)
    target_all = np.zeros((n, len(classes_all)), dtype=np.int8)
    mask_all = np.zeros((n, len(classes_all)), dtype=np.float32)
    weight_all = np.zeros((n, len(classes_all)), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        counts = row["label_source_counts"]
        sources = [str(s) for s in row["sources"]]
        n_total_sources = max(int(row["n_sources"]), 1)

        for j, c in enumerate(classes_all):
            pos_count = int(counts.get(c, 0))
            if use_source_mask:
                eligible_count = sum(
                    1 for s in sources if c in (source_label_sets or {}).get(s, set())
                )
                known = eligible_count > 0
            else:
                eligible_count = n_total_sources
                known = True

            if not known:
                continue

            frac = pos_count / max(eligible_count, 1)
            if label_agg == "union":
                y = int(pos_count >= max(min_label_sources, 1))
                confidence = 1.0
            elif label_agg == "majority":
                y = int(frac > 0.5)
                confidence = 1.0
            elif label_agg == "consensus":
                y = int(
                    eligible_count >= max(min_label_sources, 1)
                    and pos_count == eligible_count
                )
                confidence = 1.0
            elif label_agg == "soft":
                y = int(frac >= 0.5)
                confidence = max(abs(2.0 * frac - 1.0), 1e-3)
            else:
                raise ValueError(label_agg)

            target_all[i, j] = y
            mask_all[i, j] = 1.0
            weight_all[i, j] = confidence

    support = (target_all * mask_all).sum(axis=0)
    keep_cols = [j for j, s in enumerate(support) if s >= min_support]
    if not keep_cols:
        raise SystemExit(
            "[error] no fine labels reached --min-support after aggregation. "
            f"Supports: {dict(zip(classes_all, support.astype(int)))}"
        )

    classes = [classes_all[j] for j in keep_cols]
    target = target_all[:, keep_cols]
    label_mask = mask_all[:, keep_cols]
    weight = weight_all[:, keep_cols]

    keep_rows = label_mask.sum(axis=1) > 0
    df = df.loc[keep_rows].copy()
    target = target[keep_rows]
    label_mask = label_mask[keep_rows]
    weight = weight[keep_rows]

    df["target"] = [row.tolist() for row in target]
    df["sample_weight"] = weight.mean(axis=1)
    return df, classes, weight, label_mask


def load_model(model_dir: str, device: str):
    import torch
    try:
        import multimolecule  # noqa: F401  (registers RNA-FM/mRNA-FM/UTR-BERT configs)
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "multimolecule not installed. Run:\n"
            "  pip install multimolecule torch transformers\n"
            f"(import error: {e})"
        )
    md = os.path.expanduser(model_dir)
    if os.path.isdir(md):
        model_dir = os.path.abspath(md)
    else:
        is_hub_id = ("/" in model_dir) and not model_dir.startswith((".", "/", "~"))
        if not is_hub_id:
            raise SystemExit(
                f"[error] model dir not found locally: {model_dir}\n"
                "Download it first, e.g.:\n"
                "  huggingface-cli download multimolecule/utrbert-3mer --local-dir <abs_path>\n"
                "then pass that ABSOLUTE path to --model-dir.")

    name = os.path.basename(os.path.normpath(model_dir)).lower()
    if "mrnabert" in name:
        raise SystemExit(
            "[error] mRNABERT (YYLY66/mRNABERT) is NOT supported by the generic path.\n"
            "Its tokenizer expects REGION-DEMARCATED, space-separated input — UTR as "
            "single nucleotides ('A T C G') and CDS as codons ('ATG GCC TTT'), with the\n"
            "CDS boundary found upstream (e.g. ORFfinder). Feeding a raw sequence would "
            "be silently mis-tokenized into garbage embeddings.\n"
            "A dedicated mRNABERT adapter (CDS-aware dual formatting) is required; it is "
            "not yet implemented. Remove mRNABERT from your run, or implement the adapter."
        )
    # ALiBi / long-context BERT-style models (no hard positional cap, variable
    # nt-per-token). DNABERT-2 (BPE) fits here.
    is_alibi = "dnabert" in name
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device).eval()
    is_codon = bool(getattr(model.config, "codon", False))
    max_pos = int(getattr(model.config, "max_position_embeddings", 1026))
    if is_alibi:
        # Window by NUCLEOTIDES (nt_per_token=1); BPE compresses to <= nt tokens, so a
        # window never overflows. _max_tokens is generous so --max-tokens controls the
        # per-window nt length.
        model._nt_per_token = 1
        model._max_tokens = max(max_pos - 4, 4096) if max_pos > 16 else 4096
    else:
        model._nt_per_token = 3 if is_codon else 1
        model._max_tokens = max(8, max_pos - 4)
    print(f"[model] {model_dir} hidden={model.config.hidden_size} "
          f"kind={'alibi' if is_alibi else ('codon' if is_codon else 'nt')} "
          f"nt_per_token={model._nt_per_token} max_pos={max_pos} "
          f"-> max_tokens_cap={model._max_tokens}")
    return tok, model


def embed_sequences(
    seqs: list[str],
    tok,
    model,
    device: str,
    max_tokens: int = 1024,
    batch_size: int = 8,
    pool: str = "mean",
    window_pool: str = "mean",
) -> np.ndarray:
    import torch

    cap = getattr(model, "_max_tokens", max_tokens)
    if max_tokens > cap:
        print(f"[embed] capping max_tokens {max_tokens} -> {cap} (model limit)")
    max_tokens = min(max_tokens, cap)

    windows, owner = [], []
    nt_per_token = getattr(model, "_nt_per_token", 3)
    is_codon = nt_per_token == 3
    win_nt = max_tokens * nt_per_token

    for i, s in enumerate(seqs):
        s = s if s else "A"
        if is_codon:
            s = s[: len(s) - (len(s) % 3)] or "AUG"
        if len(s) <= win_nt:
            windows.append(s)
            owner.append(i)
        else:
            for start in range(0, len(s), win_nt):
                windows.append(s[start:start + win_nt])
                owner.append(i)

    dim = model.config.hidden_size
    win_emb = np.zeros((len(windows), dim), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(windows), batch_size):
            chunk = windows[b:b + batch_size]
            try:
                enc = tok(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_tokens + 2,
                    return_special_tokens_mask=True,
                )
            except TypeError:
                enc = tok(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_tokens + 2,
                )

            special = enc.pop("special_tokens_mask", None)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            attention = enc["attention_mask"].unsqueeze(-1).float()

            if pool == "cls":
                vec = hidden[:, 0, :]
            else:
                token_mask = attention
                if special is not None:
                    special = special.to(device).unsqueeze(-1).float()
                    token_mask = attention * (1.0 - special)
                    empty = token_mask.sum(dim=1, keepdim=True) == 0
                    token_mask = torch.where(empty, attention, token_mask)
                summed = (hidden * token_mask).sum(dim=1)
                vec = summed / token_mask.sum(dim=1).clamp(min=1)

            win_emb[b:b + len(chunk)] = vec.float().cpu().numpy()
            if (b // batch_size) % 50 == 0:
                print(f"  embedded {b + len(chunk)}/{len(windows)} windows", flush=True)

    owner = np.asarray(owner)
    emb = np.zeros((len(seqs), dim), dtype=np.float32)
    for i in range(len(seqs)):
        rows = win_emb[owner == i]
        emb[i] = rows.max(axis=0) if window_pool == "max" else rows.mean(axis=0)
    return emb


def kmer_features(seqs: list[str], k: int = 4) -> np.ndarray:
    from itertools import product
    kmers = ["".join(p) for p in product("ACGU", repeat=k)]
    idx = {km: i for i, km in enumerate(kmers)}
    X = np.zeros((len(seqs), len(kmers)), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper().replace("T", "U")
        counts = np.zeros(len(kmers), dtype=np.float32)
        for j in range(len(s) - k + 1):
            ii = idx.get(s[j:j + k])
            if ii is not None:
                counts[ii] += 1
        tot = counts.sum()
        if tot > 0:
            X[i] = counts / tot
    return X


def sequence_fingerprint(seqs: list[str]) -> str:
    h = hashlib.sha256()
    for seq in seqs:
        h.update(str(seq).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def model_artifact_signature(model_dir: str) -> dict:
    expanded = os.path.expanduser(model_dir)
    if not os.path.isdir(expanded):
        return {"kind": "hub_or_missing", "id": model_dir}

    root = Path(expanded).resolve()
    files = []
    for name in (
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ):
        p = root / name
        if p.exists():
            st = p.stat()
            files.append({"name": name, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    return {"kind": "local", "path": str(root), "files": files}


def get_embeddings(df: pd.DataFrame, args, cache_path: Path) -> np.ndarray:
    seqs = df["sequence"].tolist()
    expected_meta = {
        "n_sequences": len(seqs),
        "sequence_fingerprint": sequence_fingerprint(seqs),
        "model": model_artifact_signature(args.model_dir),
        "sample_level": args.sample_level,
        "region": args.region,
        "pool": args.pool,
        "window_pool": args.window_pool,
        "max_tokens": args.max_tokens,
    }
    meta_path = cache_path.with_suffix(".meta.json")

    if cache_path.exists() and meta_path.exists() and not args.no_cache:
        try:
            cached = np.load(cache_path)
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if cached.shape[0] == len(df) and cached_meta == expected_meta:
                print(f"[cache] loaded verified embeddings {cached.shape} from {cache_path}")
                return cached
            print("[cache] metadata mismatch; recomputing embeddings.")
        except Exception as e:
            print(f"[cache] unreadable cache; recomputing ({e})")
    elif cache_path.exists() and not args.no_cache:
        print("[cache] legacy cache without metadata ignored; recomputing safely.")

    tok, model = load_model(args.model_dir, args.device)
    print(f"[embed] {len(df)} sequences, pool={args.pool}, max_tokens={args.max_tokens}")
    emb = embed_sequences(
        seqs,
        tok,
        model,
        args.device,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        pool=args.pool,
        window_pool=args.window_pool,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    meta_path.write_text(json.dumps(expected_meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[cache] saved verified {emb.shape} -> {cache_path}")
    return emb


def evaluate(y_true, y_prob, classes, thresholds, label_mask=None):
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if label_mask is None:
        label_mask = np.ones_like(y_true, dtype=bool)
    else:
        label_mask = np.asarray(label_mask).astype(bool)

    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(int)
    rows = []
    for j, c in enumerate(classes):
        valid = label_mask[:, j]
        yt, yp, ys = y_true[valid, j], y_pred[valid, j], y_prob[valid, j]
        n_eval = int(valid.sum())
        sup = int(yt.sum()) if n_eval else 0

        if n_eval == 0:
            rows.append(
                dict(label=c, n_evaluable=0, support=0, precision=float("nan"),
                     recall=float("nan"), f1=float("nan"), roc_auc=float("nan"),
                     pr_auc=float("nan"), prior=float("nan"), pr_lift=float("nan"))
            )
            continue

        try:
            roc = roc_auc_score(yt, ys) if 0 < sup < n_eval else float("nan")
        except ValueError:
            roc = float("nan")
        try:
            pr = average_precision_score(yt, ys) if sup > 0 else float("nan")
        except ValueError:
            pr = float("nan")

        p, r, f, _ = precision_recall_fscore_support(
            yt, yp, average="binary", zero_division=0
        )
        prior = float(yt.mean())
        pr_lift = (pr / prior) if (prior > 0 and np.isfinite(pr)) else float("nan")
        rows.append(
            dict(label=c, n_evaluable=n_eval, support=sup, precision=p, recall=r, f1=f,
                 roc_auc=roc, pr_auc=pr, prior=round(prior, 6),
                 pr_lift=round(pr_lift, 4) if np.isfinite(pr_lift) else pr_lift)
        )

    per = pd.DataFrame(rows)
    valid_flat = label_mask.ravel()
    yt_flat = y_true.ravel()[valid_flat]
    yp_flat = y_pred.ravel()[valid_flat]
    valid_rows = label_mask.sum(axis=1) > 0
    subset = []
    for i in np.where(valid_rows)[0]:
        valid = label_mask[i]
        subset.append(bool(np.array_equal(y_pred[i, valid], y_true[i, valid])))

    overall = dict(
        n_samples=int(y_true.shape[0]),
        n_valid_label_entries=int(valid_flat.sum()),
        subset_accuracy=float(np.mean(subset)) if subset else float("nan"),
        micro_f1=f1_score(yt_flat, yp_flat, average="binary", zero_division=0)
        if len(yt_flat) else float("nan"),
        macro_f1=float(per["f1"].mean(skipna=True)),
        macro_roc_auc=float(per["roc_auc"].mean(skipna=True)),
        macro_pr_auc=float(per["pr_auc"].mean(skipna=True)),
        mean_pr_lift=float(per["pr_lift"].mean(skipna=True)),
    )
    return per, overall


def pick_thresholds(y_true, y_prob, classes, label_mask=None):
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if label_mask is None:
        label_mask = np.ones_like(y_true, dtype=bool)
    else:
        label_mask = np.asarray(label_mask).astype(bool)

    th = np.full(len(classes), 0.5, dtype=float)
    for j in range(len(classes)):
        valid = label_mask[:, j]
        yt, yp = y_true[valid, j], y_prob[valid, j]
        if len(yt) == 0 or yt.sum() == 0:
            th[j] = 1.0
            continue
        if yt.sum() == len(yt):
            th[j] = 0.0
            continue
        best_f, best_t = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            f = f1_score(yt, (yp >= t).astype(int), zero_division=0)
            if f > best_f:
                best_f, best_t = f, float(t)
        th[j] = best_t
    return th


def _label_weight(sample_weight, mask, label_index):
    if sample_weight is None:
        return None
    sw = np.asarray(sample_weight)
    if sw.ndim == 1:
        return sw[mask]
    return sw[mask, label_index]


def train_head(Xtr, Ytr, Xva, classifier, sample_weight=None, label_mask=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
    models = []
    prob_va = np.zeros((Xva.shape[0], Ytr.shape[1]), dtype=float)

    for j in range(Ytr.shape[1]):
        m = label_mask[:, j].astype(bool) if label_mask is not None else np.ones(len(Xtr_s), dtype=bool)
        yj = Ytr[m, j]
        wj = _label_weight(sample_weight, m, j)

        if len(yj) == 0:
            models.append({"kind": "constant", "prob": 0.0, "reason": "no_known_train_labels"})
            continue
        if yj.sum() == 0:
            models.append({"kind": "constant", "prob": 0.0, "reason": "all_known_negative"})
            continue
        if yj.sum() == len(yj):
            models.append({"kind": "constant", "prob": 1.0, "reason": "all_known_positive"})
            prob_va[:, j] = 1.0
            continue

        if classifier == "mlp":
            if wj is not None and not np.allclose(wj, wj[0]):
                raise ValueError(
                    "MLPClassifier does not reliably support per-sample weights. "
                    "Use --classifier logistic for --label-agg soft, or implement "
                    "the MLP head in PyTorch."
                )
            from sklearn.neural_network import MLPClassifier

            clf = MLPClassifier(
                hidden_layer_sizes=(256,),
                max_iter=300,
                early_stopping=True,
                random_state=0,
            )
            clf.fit(Xtr_s[m], yj)
        else:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
            clf.fit(Xtr_s[m], yj, sample_weight=wj)

        models.append(clf)
        prob_va[:, j] = clf.predict_proba(Xva_s)[:, 1]
    return scaler, models, prob_va


def predict_head(scaler, models, X, n_classes):
    Xs = scaler.transform(X)
    prob = np.zeros((X.shape[0], n_classes), dtype=float)
    for j, clf in enumerate(models):
        if isinstance(clf, dict) and clf.get("kind") == "constant":
            prob[:, j] = float(clf["prob"])
        elif clf is not None:
            prob[:, j] = clf.predict_proba(Xs)[:, 1]
    return prob


def finetune_model(
    genes, Y, classes, tr_idx, va_idx, te_idx, args,
    sample_weight=None, label_mask=None,
):
    """End-to-end encoder + linear-head fine-tuning with masked BCE.
    Long-sequence handling: args.ft_long_seq_policy in
    {truncate, sliding_mean, pooled_repr}."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import roc_auc_score

    tok, base = load_model(args.model_dir, args.device)
    device = args.device
    nt = getattr(base, "_nt_per_token", 1)
    max_tok = args.ft_max_len or getattr(base, "_max_tokens", 510)
    max_tok = min(max_tok, getattr(base, "_max_tokens", max_tok))
    win_nt = max_tok * nt
    seqs = genes["sequence"].tolist()
    nC = len(classes)
    policy = args.ft_long_seq_policy

    Y = np.asarray(Y, dtype=np.float32)
    if label_mask is None:
        label_mask = np.ones_like(Y, dtype=np.float32)
    else:
        label_mask = np.asarray(label_mask, dtype=np.float32)
    if sample_weight is None:
        sample_weight = np.ones_like(Y, dtype=np.float32)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float32)
        if sample_weight.ndim == 1:
            sample_weight = np.repeat(sample_weight[:, None], nC, axis=1)

    def windows_for(s):
        s = str(s) if s else "A"
        if nt == 3:
            s = s[: len(s) - (len(s) % 3)] or "AUG"
        if policy == "truncate" or len(s) <= win_nt:
            return [s[:win_nt] or "A"]
        return [s[i:i + win_nt] for i in range(0, len(s), win_nt)]

    win_lists = [windows_for(seqs[j]) for j in range(len(seqs))]

    used = np.unique(np.concatenate([tr_idx, va_idx, te_idx]))
    nwin = np.array([len(win_lists[j]) for j in used], dtype=int)
    if policy == "truncate":
        n_trunc = int(sum(1 for j in used if len(str(seqs[j])) > win_nt))
        if n_trunc:
            print(
                f"[finetune][WARN] policy=truncate: {n_trunc}/{len(used)} sequences "
                f"exceed {win_nt} nt and will be TRUNCATED to the proximal window. "
                "Use --ft-long-seq-policy sliding_mean|pooled_repr to cover the full "
                "transcript.",
                flush=True,
            )
    else:
        print(
            f"[finetune] policy={policy}: {int((nwin > 1).sum())}/{len(used)} "
            f"transcripts span >1 window; avg {nwin.mean():.2f} windows/transcript "
            f"(~{nwin.mean():.1f}x forward/backward vs truncate).",
            flush=True,
        )

    head = nn.Linear(base.config.hidden_size, nC).to(device)
    base = base.to(device)
    known_train = label_mask[tr_idx].sum(axis=0)
    pos = (Y[tr_idx] * label_mask[tr_idx]).sum(axis=0)
    neg = known_train - pos
    pos_weight = torch.tensor(
        np.clip(neg / np.clip(pos, 1, None), 0.2, 5.0),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        for j in range(nC):
            if known_train[j] == 0 or pos[j] == 0:
                head.weight[j].zero_(); head.bias[j].fill_(-10.0)
            elif pos[j] == known_train[j]:
                head.weight[j].zero_(); head.bias[j].fill_(10.0)

    opt = torch.optim.AdamW(
        list(base.parameters()) + list(head.parameters()), lr=args.ft_lr, weight_decay=0.01,
    )
    use_amp = str(device).startswith("cuda")
    grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def _token_mean(hidden, attention_mask):
        att = attention_mask.unsqueeze(-1).float()
        return (hidden * att).sum(1) / att.sum(1).clamp(min=1)

    def _hidden(enc):
        out = base(**enc)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    class WindowDS(Dataset):
        def __init__(self, indices):
            self.items = []
            for j in indices:
                wl = win_lists[j]
                for w in wl:
                    self.items.append((int(j), w, len(wl)))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            j, w, n = self.items[i]
            return w, Y[j], sample_weight[j], label_mask[j], 1.0 / n, j

    def collate_window(batch):
        xs = [b[0] for b in batch]
        ys = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
        ws = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
        ms = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        norm = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        owners = [b[5] for b in batch]
        enc = tok(xs, return_tensors="pt", padding=True, truncation=True, max_length=max_tok + 2)
        return enc, ys, ws, ms, norm, owners

    def run_window(indices, train):
        dl = DataLoader(WindowDS(indices), batch_size=args.ft_batch, shuffle=train, collate_fn=collate_window)
        base.train(train); head.train(train)
        pos_of = {int(j): p for p, j in enumerate(indices)}
        sum_logits = np.zeros((len(indices), nC), dtype=np.float64)
        cnt = np.zeros((len(indices), 1), dtype=np.float64)

        for enc, ys, ws, ms, norm, owners in dl:
            enc = {k: v.to(device) for k, v in enc.items()}
            ys, ws, ms, norm = ys.to(device), ws.to(device), ms.to(device), norm.to(device)
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast(enabled=use_amp):
                hidden = _hidden(enc)
                logits = head(_token_mean(hidden, enc["attention_mask"]))
                raw_loss = F.binary_cross_entropy_with_logits(
                    logits, ys, pos_weight=pos_weight, reduction="none"
                )
                effective = ms * ws * norm.unsqueeze(1)
                loss = (raw_loss * effective).sum() / effective.sum().clamp(min=1.0)
            if train:
                opt.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()
            lg = logits.detach().float().cpu().numpy()
            for k, j in enumerate(owners):
                p = pos_of[int(j)]
                sum_logits[p] += lg[k]
                cnt[p, 0] += 1
        cnt[cnt == 0] = 1.0
        return (sum_logits / cnt).astype(np.float32)

    class TxDS(Dataset):
        def __init__(self, indices):
            self.indices = [int(j) for j in indices]

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            j = self.indices[i]
            return win_lists[j], Y[j], sample_weight[j], label_mask[j], j

    def collate_tx(batch):
        xs, seg = [], []
        for bi, b in enumerate(batch):
            for w in b[0]:
                xs.append(w)
                seg.append(bi)
        ys = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
        ws = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
        ms = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        owners = [b[4] for b in batch]
        enc = tok(xs, return_tensors="pt", padding=True, truncation=True, max_length=max_tok + 2)
        return enc, ys, ws, ms, torch.tensor(seg, dtype=torch.long), owners

    def run_pooled(indices, train):
        dl = DataLoader(TxDS(indices), batch_size=args.ft_batch, shuffle=train, collate_fn=collate_tx)
        base.train(train); head.train(train)
        logits_chunks, owners_all = [], []

        for enc, ys, ws, ms, seg, owners in dl:
            enc = {k: v.to(device) for k, v in enc.items()}
            ys, ws, ms, seg = ys.to(device), ws.to(device), ms.to(device), seg.to(device)
            B = ys.shape[0]
            with torch.set_grad_enabled(train), torch.cuda.amp.autocast(enabled=use_amp):
                hidden = _hidden(enc)
                win_vec = _token_mean(hidden, enc["attention_mask"])
                tx_vec = torch.zeros(B, win_vec.shape[1], device=device, dtype=win_vec.dtype)
                tx_vec.index_add_(0, seg, win_vec)
                counts = torch.zeros(B, 1, device=device, dtype=win_vec.dtype)
                counts.index_add_(0, seg, torch.ones(seg.shape[0], 1, device=device, dtype=win_vec.dtype))
                tx_vec = tx_vec / counts.clamp(min=1)
                logits = head(tx_vec)
                raw_loss = F.binary_cross_entropy_with_logits(
                    logits, ys, pos_weight=pos_weight, reduction="none"
                )
                effective = ms * ws
                loss = (raw_loss * effective).sum() / effective.sum().clamp(min=1.0)
            if train:
                opt.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()
            logits_chunks.append(logits.detach().float().cpu().numpy())
            owners_all.extend(int(o) for o in owners)

        res = np.zeros((len(indices), nC), dtype=np.float32)
        if logits_chunks:
            cat = np.concatenate(logits_chunks)
            pos_of = {int(j): p for p, j in enumerate(indices)}
            for row, j in zip(cat, owners_all):
                res[pos_of[j]] = row
        return res

    def run(indices, train):
        if policy == "pooled_repr":
            return run_pooled(indices, train)
        return run_window(indices, train)

    def macro_auc(indices, prob):
        vals = []
        for j in range(nC):
            valid = label_mask[indices, j].astype(bool)
            yt = Y[indices, j][valid]
            if 0 < yt.sum() < len(yt):
                vals.append(roc_auc_score(yt, prob[valid, j]))
        return float(np.mean(vals)) if vals else 0.0

    best_auc, best, stale = -1.0, None, 0
    for ep in range(args.ft_epochs):
        run(tr_idx, True)
        va_prob = 1.0 / (1.0 + np.exp(-run(va_idx, False)))
        auc = macro_auc(va_idx, va_prob)
        print(f"[finetune] epoch {ep + 1}/{args.ft_epochs} val_macro_auc={auc:.4f}", flush=True)
        if auc > best_auc + 1e-6:
            best_auc = auc
            stale = 0
            best = (
                {k: v.detach().cpu().clone() for k, v in base.state_dict().items()},
                {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            )
        else:
            stale += 1
            if stale >= args.ft_patience:
                print(f"[finetune] early stop after {stale} non-improving epoch(s).")
                break

    if best is not None:
        base.load_state_dict(best[0])
        head.load_state_dict(best[1])
    va_prob = 1.0 / (1.0 + np.exp(-run(va_idx, False)))
    te_prob = 1.0 / (1.0 + np.exp(-run(te_idx, False)))
    print(f"[finetune] best val_macro_auc={best_auc:.4f}")
    return va_prob, te_prob


# ----------------------------------------------------------------------------
# Leakage-safe grouping and split
# ----------------------------------------------------------------------------
_SPECIES_ALIASES = {
    "human": "human", "homo_sapiens": "human", "homo sapiens": "human",
    "mouse": "mouse", "mus_musculus": "mouse", "mus musculus": "mouse",
    "rat": "rat", "rattus_norvegicus": "rat", "rattus norvegicus": "rat",
}


def canonical_species(value: object) -> str:
    raw = str(value).strip()
    key = raw.casefold().replace("-", "_").replace(" ", "_")
    return _SPECIES_ALIASES.get(key, key)


def _ortholog_key(species: object, gene: object) -> tuple[str, str]:
    return canonical_species(species), str(gene).strip().upper()


def load_ortholog_map(path: Path | None) -> dict[tuple[str, str], str]:
    if path is None:
        print("[ortholog] no --ortholog-map supplied; using species|gene fallback.")
        return {}
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[error] --ortholog-map not found: {path}")

    try:
        odf = pd.read_csv(path, sep=None, engine="python", dtype=str)
    except Exception:
        odf = pd.read_csv(path, sep="\t", dtype=str)
    odf.columns = [str(c).strip() for c in odf.columns]
    lower = {c: c.casefold().replace("-", "_").replace(" ", "_") for c in odf.columns}

    group_col = next(
        (c for c, k in lower.items()
         if k in {"ortholog_group", "ortholog_group_id", "homology_group", "group_id"}
         or ("ortholog" in k and "group" in k)),
        None,
    )
    if group_col is None:
        raise SystemExit(
            "[error] --ortholog-map needs an ortholog-group column. "
            f"Found columns: {odf.columns.tolist()}"
        )

    species_col = next((c for c, k in lower.items() if k in {"species", "organism"}), None)
    gene_col = next(
        (c for c, k in lower.items() if k in {"gene_name", "gene_symbol", "gene", "symbol"}),
        None,
    )

    mapping: dict[tuple[str, str], str] = {}
    conflicts = 0

    def add(species, gene, group):
        nonlocal conflicts
        if pd.isna(species) or pd.isna(gene) or pd.isna(group):
            return
        group = str(group).strip()
        gene = str(gene).strip()
        if not group or not gene:
            return
        key = _ortholog_key(species, gene)
        previous = mapping.get(key)
        if previous is not None and previous != group:
            conflicts += 1
            return
        mapping[key] = group

    if species_col is not None and gene_col is not None:
        for _, r in odf.iterrows():
            add(r[species_col], r[gene_col], r[group_col])
    else:
        wide_cols = []
        for c, k in lower.items():
            if c == group_col:
                continue
            sp = next((s for s in ("human", "mouse", "rat") if s in k), None)
            if sp and ("gene" in k or "symbol" in k):
                wide_cols.append((c, sp))
        if not wide_cols:
            raise SystemExit(
                "[error] cannot infer gene/species columns from --ortholog-map. "
                f"Found columns: {odf.columns.tolist()}"
            )
        for _, r in odf.iterrows():
            for c, sp in wide_cols:
                add(sp, r[c], r[group_col])

    print(
        f"[ortholog] loaded {len(mapping)} species-gene entries from {path}"
        + (f" ({conflicts} conflicting entries ignored)" if conflicts else "")
    )
    return mapping


class _UnionFind:
    def __init__(self, values):
        self.parent = {v: v for v in values}

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            nxt = self.parent[value]
            self.parent[value] = root
            value = nxt
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if str(ra) <= str(rb):
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def build_leakage_safe_groups(genes: pd.DataFrame, ortholog_map: dict[tuple[str, str], str]) -> np.ndarray:
    base_groups = []
    n_mapped = 0
    for sp, gene in zip(genes["species"], genes["gene_name"]):
        mapped = ortholog_map.get(_ortholog_key(sp, gene))
        if mapped:
            base_groups.append(f"orth:{mapped}")
            n_mapped += 1
        else:
            base_groups.append(f"gene:{canonical_species(sp)}|{str(gene).strip().upper()}")

    uf = _UnionFind(set(base_groups))
    hashes = genes["sequence"].astype(str).str.upper().map(
        lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    )
    for _, indices in pd.Series(np.arange(len(genes))).groupby(hashes, sort=False):
        bases = [base_groups[i] for i in indices.to_numpy()]
        for b in bases[1:]:
            uf.union(bases[0], b)

    out = np.asarray([uf.find(b) for b in base_groups], dtype=object)
    print(
        f"[groups] {len(genes)} samples -> {len(set(out))} leakage-safe groups; "
        f"{n_mapped}/{len(genes)} samples mapped to an ortholog group."
    )
    return out


def _split_balance_score(Y, label_mask, parts, ratios):
    Y = np.asarray(Y, dtype=float)
    M = np.asarray(label_mask, dtype=float)
    n_total = max(len(Y), 1)
    total_pos = (Y * M).sum(axis=0)
    total_known = M.sum(axis=0)
    score = 0.0

    for split_i, ids in enumerate(parts):
        frac = len(ids) / n_total
        score += 100.0 * (frac - ratios[split_i]) ** 2
        if len(ids) == 0:
            score += 1e6
            continue
        pos = (Y[ids] * M[ids]).sum(axis=0)
        known = M[ids].sum(axis=0)
        score += float(np.mean(((pos - total_pos * ratios[split_i]) / np.sqrt(total_pos + 1.0)) ** 2))
        score += 0.20 * float(np.mean(((known - total_known * ratios[split_i]) / np.sqrt(total_known + 1.0)) ** 2))
        if split_i == 0:
            score += 1000.0 * float(((total_pos > 0) & (pos == 0)).sum())
        else:
            score += 20.0 * float(((total_pos >= 3) & (pos == 0)).sum())
    return score


def grouped_multilabel_split(Y, groups, label_mask, seed, attempts=256):
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.asarray(groups)
    idx = np.arange(len(groups))
    ratios = (0.70, 0.15, 0.15)
    best = None

    for trial in range(max(int(attempts), 1)):
        rs = int(seed) + trial * 2
        tr, tmp = next(
            GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=rs).split(idx, groups=groups)
        )
        va_rel, te_rel = next(
            GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=rs + 1).split(tmp, groups=groups[tmp])
        )
        va, te = tmp[va_rel], tmp[te_rel]
        score = _split_balance_score(Y, label_mask, (tr, va, te), ratios)
        if best is None or score < best[0]:
            best = (score, tr, va, te)

    assert best is not None
    score, tr_idx, va_idx, te_idx = best
    split_sets = [set(groups[x]) for x in (tr_idx, va_idx, te_idx)]
    if split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]:
        raise RuntimeError("group leakage detected after split construction")
    print(
        f"[split] best grouped multi-label split from {max(int(attempts), 1)} attempts: "
        f"train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)} score={score:.3f}"
    )
    return tr_idx, va_idx, te_idx, float(score)


def load_external_split(path: Path, genes: pd.DataFrame, groups: np.ndarray):
    """Reuse a frozen split saved by a previous run (split_assignments.csv with
    columns species, gene_name, split). Keyed on (canonical_species, GENE upper),
    so a gene-level frozen split applies cleanly to an isoform-level run. Every
    sample must be covered, and no leakage-safe group may straddle two splits."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[error] --split-assignments not found: {path}")
    sp = pd.read_csv(path, dtype=str)
    need = {"species", "gene_name", "split"}
    if not need.issubset(sp.columns):
        raise SystemExit(
            f"[error] --split-assignments needs columns {need}; found {sp.columns.tolist()}"
        )
    key2split = {}
    for _, r in sp.iterrows():
        key = (canonical_species(r["species"]), str(r["gene_name"]).strip().upper())
        key2split[key] = str(r["split"]).strip().lower()

    tr, va, te, missing = [], [], [], []
    for i, (s, g) in enumerate(zip(genes["species"], genes["gene_name"])):
        lab = key2split.get((canonical_species(s), str(g).strip().upper()))
        if lab == "train":
            tr.append(i)
        elif lab in ("val", "validation"):
            va.append(i)
        elif lab == "test":
            te.append(i)
        else:
            missing.append((s, g))
    if missing:
        ex = missing[:5]
        raise SystemExit(
            f"[error] --split-assignments does not cover {len(missing)} samples "
            f"(e.g. {ex}). The frozen split must include every sample to stay "
            "leakage-safe. Regenerate the split on this exact data universe."
        )
    tr, va, te = np.array(tr, int), np.array(va, int), np.array(te, int)

    g = np.asarray(groups)
    s_tr, s_va, s_te = set(g[tr]), set(g[va]), set(g[te])
    if s_tr & s_va or s_tr & s_te or s_va & s_te:
        bad = (s_tr & s_va) | (s_tr & s_te) | (s_va & s_te)
        raise SystemExit(
            f"[error] external split leaks: {len(bad)} leakage-safe group(s) span "
            "multiple splits. Regenerate the frozen split on this universe."
        )
    print(
        f"[split] loaded frozen split from {path}: "
        f"train={len(tr)} val={len(va)} test={len(te)}"
    )
    return tr, va, te


def make_split_support_table(Y, label_mask, classes, tr_idx, va_idx, te_idx):
    rows = []
    for split_name, ids in (("train", tr_idx), ("val", va_idx), ("test", te_idx)):
        for j, label in enumerate(classes):
            valid = np.asarray(label_mask)[ids, j].astype(bool)
            yt = np.asarray(Y)[ids, j][valid]
            rows.append({
                "split": split_name, "label": label, "n_samples": int(len(ids)),
                "n_evaluable": int(valid.sum()),
                "positives": int(yt.sum()) if len(yt) else 0,
                "prior": float(yt.mean()) if len(yt) else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=False)
    ap.add_argument("--model-dir", default="./rnafm")
    ap.add_argument("--output-dir", type=Path, default=Path("results/rnafm_run"))
    ap.add_argument("--region", choices=["full", "cds", "utr3"], default="full",
                    help="region fed to the model. Only full-length (cdna/transcript/"
                    "mrna) rows are GTF-extracted; every other row passes through.")
    ap.add_argument("--gtf", nargs="*", default=[
        "data_训练/Homo_sapiens.GRCh38.116.gtf.gz",
        "data_训练/Mus_musculus.GRCm39.116.gtf.gz",
        "data_训练/Rattus_norvegicus.GRCr8.116.gtf.gz",
    ], help="Ensembl GTF(s) for UTR-length lookup (only used when --region != full)")
    ap.add_argument("--native-region-sources", nargs="*", default=[],
                    help="source-name substrings whose rows are ALREADY in --region "
                    "(e.g. native isoform 3'UTR sources). They pass through untouched "
                    "regardless of sequence_type. Everything else with a blank "
                    "sequence_type is treated as full-length and GTF-extracted, so a "
                    "merged bulk+isoform file's blank-tagged bulk transcripts are not "
                    "silently used at full length under --region utr3.")
    ap.add_argument("--label-scheme", choices=["soma_vs_neurite", "fine"], default="soma_vs_neurite")
    ap.add_argument("--keep-assay-labels", action="store_true",
                    help="also model the assay-readout labels Ribosome/Cytoplasm as "
                    "fine target classes (a translation/fractionation axis, distinct "
                    "from anatomical compartments). Requires --source-mask; report them "
                    "separately from the localization compartments.")
    ap.add_argument("--species", nargs="*", default=None,
                    help="keep only these species (canonical: mouse/rat/human). "
                    "Default keeps all. e.g. --species mouse rat drops the tiny human "
                    "slice for a clean rodent benchmark.")
    ap.add_argument("--label-agg", choices=["consensus", "majority", "soft", "union"], default="soft")
    ap.add_argument("--min-label-sources", type=int, default=1)
    ap.add_argument("--sample-level",
                    choices=["gene", "isoform", "isoform_sequence_union", "source_gene", "sequence_union"],
                    default="gene")
    ap.add_argument("--min-support", type=int, default=150)
    ap.add_argument("--source-mask", action="store_true",
                    help="infer per-label observability from each source's label "
                    "vocabulary. Required for --label-scheme fine.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-sources", type=int, default=1)
    ap.add_argument("--classifier", choices=["logistic", "mlp"], default="logistic")
    ap.add_argument("--baseline", choices=["none", "kmer"], default="none")
    ap.add_argument("--kmer-k", type=int, default=4)
    ap.add_argument("--features", nargs="*", default=None,
                    choices=["fm", "kmer", "engineered", "length", "structure"],
                    help="STACK feature families into one matrix for the logistic/mlp "
                    "head, e.g. --features fm engineered. Overrides the single-source "
                    "path; not usable with --arch/--finetune. 'length'/'engineered' "
                    "include the 3'UTR-length CONFOUND (also run --features length as a "
                    "baseline to quantify it); 'structure' needs ViennaRNA.")
    # ---- feature family selector --------------------------------------------
    ap.add_argument("--arch", choices=["fm", "rnatracker", "dm3loc"], default="fm",
                    help="fm = foundation-model embeddings/finetune (default); "
                    "rnatracker / dm3loc = purpose-built localization nets trained "
                    "from scratch on one-hot sequence (same split/labels/eval).")
    ap.add_argument("--ts-max-len", type=int, default=4000, help="one-hot length for --arch nets")
    ap.add_argument("--ts-epochs", type=int, default=30)
    ap.add_argument("--ts-patience", type=int, default=5)
    ap.add_argument("--ts-batch", type=int, default=32)
    ap.add_argument("--ts-lr", type=float, default=1e-3)
    # ---- foundation-model fine-tune -----------------------------------------
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--ft-epochs", type=int, default=4)
    ap.add_argument("--ft-patience", type=int, default=2)
    ap.add_argument("--ft-lr", type=float, default=2e-5)
    ap.add_argument("--ft-batch", type=int, default=8)
    ap.add_argument("--ft-max-len", type=int, default=0, help="tokens per window during fine-tune (0 = model cap)")
    ap.add_argument("--ft-long-seq-policy", choices=["truncate", "sliding_mean", "pooled_repr"],
                    default="truncate",
                    help="fine-tune long-seq handling: truncate / sliding_mean / pooled_repr.")
    ap.add_argument("--pool", choices=["mean", "cls"], default="mean")
    ap.add_argument("--window-pool", choices=["mean", "max"], default="mean")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--emb-cache-dir", type=Path, default=Path("results/embedding_cache"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--ortholog-map", type=Path, default=None)
    ap.add_argument("--split-attempts", type=int, default=256)
    ap.add_argument("--split-assignments", type=Path, default=None,
                    help="reuse a frozen split (a previous run's split_assignments.csv) "
                    "so multiple models/scripts compare on the EXACT same partition.")
    ap.add_argument("--restrict-to-split", action="store_true",
                    help="with --split-assignments, DROP samples not present in the "
                    "frozen split (instead of erroring). Use for the region ablation: "
                    "run utr3/cds/full on exactly the same gene intersection.")
    ap.add_argument("--train-on-all", action="store_true",
                    help="DEPLOYMENT fit: train the head on ALL samples (no held-out "
                    "test). Reported metrics become IN-SAMPLE; use this only to build "
                    "the final shipped artifact after selecting the config on a frozen "
                    "split. The honest test number must come from the benchmark run.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    is_task_specific = args.arch in ("rnatracker", "dm3loc")

    if args.selftest:
        import torch
        args.device = args.device if torch.cuda.is_available() else "cpu"
        tok, model = load_model(args.model_dir, args.device)
        demo = ["AUGGCGAACCUUGGCUGCUGGGCUGGUUCUCUUUGUGGCC", "AUG" + "GCU" * 600]
        emb = embed_sequences(demo, tok, model, args.device,
                              max_tokens=args.max_tokens, batch_size=2, pool=args.pool)
        print("OK selftest. embedding shape:", emb.shape, "hidden_size:", model.config.hidden_size)
        return

    if args.input_dir is None:
        raise SystemExit("[error] --input-dir is required unless --selftest is used.")
    if args.label_scheme == "fine" and not args.source_mask:
        raise SystemExit("[error] --label-scheme fine requires --source-mask.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, using CPU (slow).")
        args.device = "cpu"

    all_rows = load_all_rows(args.input_dir)
    if args.species:
        keep = {canonical_species(s) for s in args.species}
        before = len(all_rows)
        all_rows = all_rows[all_rows["species"].map(canonical_species).isin(keep)].copy()
        print(f"[species] kept {len(all_rows)}/{before} rows for species={sorted(keep)}")
        if len(all_rows) == 0:
            raise SystemExit(f"[error] no rows for --species {args.species}")
    if args.region != "full":
        L5map, L3map = build_utr_length_map(args.gtf)
        all_rows = apply_region(all_rows, args.region, L5map, L3map,
                                native_sources=args.native_region_sources)
        if len(all_rows) == 0:
            raise SystemExit(f"[error] region={args.region}: 0 rows survived region harmonization.")

    rep = select_representative_rows(all_rows) if args.sample_level == "gene" else all_rows

    observed_labels = collect_observed_labels(rep)
    known_labels = set(FINE_LABELS) | ASSAY_LABELS
    unknown_labels = sorted(observed_labels - known_labels)
    pd.DataFrame({"unknown_or_unmodeled_label": unknown_labels}).to_csv(
        args.output_dir / "unknown_label_audit.csv", index=False
    )
    if unknown_labels:
        print(f"[labels] unmodeled labels (recorded in audit): {unknown_labels}")

    if args.sample_level == "gene":
        genes = build_gene_dataset(rep, args.keep_assay_labels).reset_index(drop=True)
    else:
        genes = build_sample_dataset(rep, args.sample_level, args.keep_assay_labels).reset_index(drop=True)

    src_measured: dict[str, set[str]] = {}
    for src, sg in rep.groupby("source", sort=False):
        labs: set[str] = set()
        for v in sg["location"]:
            labs.update(split_location(v))
        src_measured[str(src)] = labs
    pd.DataFrame({
        "source": list(src_measured),
        "observed_labels": [",".join(sorted(v)) for v in src_measured.values()],
    }).to_csv(args.output_dir / "source_label_capability_audit.csv", index=False)

    if args.source_mask and args.label_scheme == "soma_vs_neurite":
        two_sided_sources = {
            s for s, labs in src_measured.items()
            if (labs & SOMA_LABELS) and (labs & NEURITE_LABELS)
        }
        has_two_sided_src = genes["sources"].apply(
            lambda ss: any(str(s) in two_sided_sources for s in ss)
        )
        is_soma_only = genes["neurite_frac"].eq(0)
        drop = is_soma_only & ~has_two_sided_src
        before = len(genes)
        genes = genes.loc[~drop].reset_index(drop=True)
        print(f"[source-mask] two-sided sources: {sorted(two_sided_sources)}")
        print(f"[source-mask] removed {int(drop.sum())} unreliable soma-only negatives; "
              f"kept {len(genes)}/{before}")

    if len(genes) == 0:
        raise SystemExit("[error] no samples remain after loading/source-mask filtering.")

    genes = genes.reset_index(drop=True)
    genes["_uid"] = np.arange(len(genes))

    multi = genes[genes["n_sources"] >= 2]
    audit = dict(
        samples_total=int(len(genes)),
        samples_multi_source=int(len(multi)),
        binary_conflict_samples=int(genes["binary_conflict"].sum()),
        binary_conflict_frac_of_multi=(float(multi["binary_conflict"].mean()) if len(multi) else np.nan),
    )
    print("[conflict]", audit)
    pd.DataFrame([audit]).to_csv(args.output_dir / "conflict_audit.csv", index=False)

    if args.features and (is_task_specific or args.finetune):
        raise SystemExit("[error] --features (stacked head) is not usable with --arch / --finetune.")

    if is_task_specific:
        feat_tag = args.arch
        emb_full = None
    elif args.finetune:
        feat_tag = os.path.basename(os.path.normpath(args.model_dir))
        emb_full = None
    elif args.features:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from engineered_features import build_feature_block
        seqs_all = genes["sequence"].tolist()
        blocks, tags = [], []
        for fblk in args.features:
            if fblk == "fm":
                model_tag = os.path.basename(os.path.normpath(args.model_dir))
                emb_sig = (f"{model_tag}_{args.sample_level}_{args.region}_"
                           f"{args.pool}_{args.window_pool}_{args.max_tokens}")
                args.emb_cache_dir.mkdir(parents=True, exist_ok=True)
                blocks.append(get_embeddings(genes, args, args.emb_cache_dir / f"emb_{emb_sig}.npy"))
                tags.append(model_tag)
            else:
                X, tag = build_feature_block(fblk, seqs_all, kmer_k=args.kmer_k)
                blocks.append(X.astype(np.float32))
                tags.append(tag)
        emb_full = np.concatenate(blocks, axis=1).astype(np.float32)
        feat_tag = "+".join(tags)
        print(f"[features] stacked {tags} -> {emb_full.shape}")
    elif args.baseline == "kmer":
        feat_tag = f"kmer{args.kmer_k}_{args.region}"
        print(f"[baseline] computing {args.kmer_k}-mer features for {len(genes)} sequences ...")
        emb_full = kmer_features(genes["sequence"].tolist(), args.kmer_k)
        print(f"[baseline] kmer features shape {emb_full.shape}")
    else:
        model_tag = os.path.basename(os.path.normpath(args.model_dir))
        feat_tag = model_tag
        emb_sig = f"{model_tag}_{args.sample_level}_{args.region}_{args.pool}_{args.window_pool}_{args.max_tokens}"
        args.emb_cache_dir.mkdir(parents=True, exist_ok=True)
        emb_full = get_embeddings(genes, args, args.emb_cache_dir / f"emb_{emb_sig}.npy")

    genes, classes, sw_all, label_mask_full = apply_label_scheme(
        genes, args.label_scheme, args.min_support, args.label_agg, args.min_label_sources,
        args.min_sources, source_label_sets=src_measured, use_source_mask=args.source_mask,
        keep_assay=args.keep_assay_labels,
    )
    genes = genes.reset_index(drop=True)
    if len(genes) == 0:
        raise SystemExit("[error] no samples remain after label aggregation.")
    Y = np.asarray(genes["target"].tolist(), dtype=np.int8)
    print(f"[data] agg={args.label_agg} {len(genes)} samples, classes={classes}")
    print("[data] per-class support:",
          {c: int((Y[:, j] * label_mask_full[:, j]).sum()) for j, c in enumerate(classes)})
    if args.label_scheme == "fine":
        print("[source-mask fine] per-label evaluable coverage:",
              {c: round(float(label_mask_full[:, j].mean()), 3) for j, c in enumerate(classes)})

    emb = None if (args.finetune or is_task_specific) else emb_full[genes["_uid"].to_numpy()]

    ortholog_map = load_ortholog_map(args.ortholog_map)
    groups = build_leakage_safe_groups(genes, ortholog_map)
    genes["split_group"] = groups

    if args.split_assignments is not None and args.restrict_to_split:
        sp_keys = pd.read_csv(args.split_assignments, dtype=str)
        keyset = {(canonical_species(s), str(g).strip().upper())
                  for s, g in zip(sp_keys["species"], sp_keys["gene_name"])}
        keep = np.array([(canonical_species(s), str(g).strip().upper()) in keyset
                         for s, g in zip(genes["species"], genes["gene_name"])])
        n0 = len(genes)
        genes = genes.loc[keep].reset_index(drop=True)
        Y = Y[keep]
        label_mask_full = label_mask_full[keep]
        sw_all = np.asarray(sw_all)[keep]
        groups = groups[keep]
        if emb is not None:
            emb = emb[keep]
        genes["split_group"] = groups
        print(f"[restrict-to-split] kept {len(genes)}/{n0} samples present in the frozen split")
        if len(genes) == 0:
            raise SystemExit("[error] --restrict-to-split removed all samples (no overlap with split).")

    if args.split_assignments is not None:
        tr_idx, va_idx, te_idx = load_external_split(args.split_assignments, genes, groups)
        split_score = float("nan")
    else:
        tr_idx, va_idx, te_idx, split_score = grouped_multilabel_split(
            Y, groups, label_mask_full, args.seed, attempts=args.split_attempts,
        )

    if args.train_on_all:
        print("[train-on-all] DEPLOYMENT fit: head trained on ALL samples; reported "
              "metrics are IN-SAMPLE, NOT a held-out test. Use the benchmark run's "
              "frozen-split metrics for reporting; this run only produces the artifact.")
        all_idx = np.arange(len(genes))
        tr_idx = va_idx = te_idx = all_idx
        split_score = float("nan")

    make_split_support_table(Y, label_mask_full, classes, tr_idx, va_idx, te_idx).to_csv(
        args.output_dir / "split_label_support.csv", index=False
    )
    pd.DataFrame({
        "row_index": np.arange(len(genes)),
        "species": genes["species"],
        "gene_name": genes["gene_name"],
        "split_group": groups,
        "split": np.where(np.isin(np.arange(len(genes)), tr_idx), "train",
                          np.where(np.isin(np.arange(len(genes)), va_idx), "val", "test")),
    }).to_csv(args.output_dir / "split_assignments.csv", index=False)

    if is_task_specific:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from task_specific_models import train_task_specific
        prob_va, prob_te = train_task_specific(
            genes, Y, classes, tr_idx, va_idx, te_idx, args, args.arch,
            sample_weight=sw_all, label_mask=label_mask_full,
        )
        scaler = models = None
        run_name = f"{args.arch}_{args.region}"
    elif args.finetune:
        prob_va, prob_te = finetune_model(
            genes, Y, classes, tr_idx, va_idx, te_idx, args,
            sample_weight=sw_all, label_mask=label_mask_full,
        )
        scaler = models = None
        run_name = f"{feat_tag}_finetune_{args.region}_{args.ft_long_seq_policy}"
    else:
        scaler, models, prob_va = train_head(
            emb[tr_idx], Y[tr_idx], emb[va_idx], args.classifier,
            sample_weight=np.asarray(sw_all)[tr_idx], label_mask=label_mask_full[tr_idx],
        )
        prob_te = predict_head(scaler, models, emb[te_idx], len(classes))
        run_name = f"{feat_tag}_{args.classifier}"

    thresholds = pick_thresholds(Y[va_idx], prob_va, classes, label_mask=label_mask_full[va_idx])
    per, overall = evaluate(Y[te_idx], prob_te, classes, thresholds, label_mask=label_mask_full[te_idx])
    # Validation metrics are recorded so model SELECTION can use val (never test).
    # Threshold-free metrics (macro_roc_auc / macro_pr_auc) on val are the honest
    # selection criterion; val F1 is mildly optimistic (thresholds picked on val).
    per_val, overall_val = evaluate(
        Y[va_idx], prob_va, classes, thresholds, label_mask=label_mask_full[va_idx]
    )

    prior = np.zeros(len(classes), dtype=float)
    for j in range(len(classes)):
        valid = label_mask_full[tr_idx, j].astype(bool)
        prior[j] = float(Y[tr_idx, j][valid].mean()) if valid.any() else 0.0

    per_prior, ov_prior = evaluate(
        Y[te_idx], np.tile(prior, (len(te_idx), 1)), classes,
        np.full(len(classes), 0.5), label_mask=label_mask_full[te_idx],
    )
    per_zero, ov_zero = evaluate(
        Y[te_idx], np.zeros_like(prob_te), classes,
        np.full(len(classes), 0.5), label_mask=label_mask_full[te_idx],
    )

    meta = {"model": run_name, "split_score": split_score, "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)), "n_test": int(len(te_idx))}
    overall.update(meta); overall["split"] = "test"
    overall_val.update(meta); overall_val["split"] = "val"
    ov_prior.update({"model": "label_prior_probability", "split_score": split_score, "split": "test"})
    ov_zero.update({"model": "all_zero", "split_score": split_score, "split": "test"})
    per_prior["model"] = "label_prior_probability"
    per_zero["model"] = "all_zero"
    per["model"] = run_name

    per.to_csv(args.output_dir / "per_label_metrics.csv", index=False)
    pd.concat([per_prior, per_zero], ignore_index=True).to_csv(
        args.output_dir / "baseline_per_label_metrics.csv", index=False
    )
    # val row first so select_best can pick on validation without touching test.
    pd.DataFrame([overall_val, overall, ov_prior, ov_zero]).to_csv(
        args.output_dir / "overall_metrics.csv", index=False
    )
    pd.DataFrame({"label": classes, "threshold": thresholds}).to_csv(
        args.output_dir / "label_thresholds.csv", index=False
    )

    # Per-sample TEST predictions, so a group-level bootstrap can test whether
    # model differences are significant (join two runs on species+gene_name,
    # resample by split_group).
    pred = pd.DataFrame({
        "species": genes["species"].astype(str).to_numpy()[te_idx],
        "gene_name": genes["gene_name"].astype(str).to_numpy()[te_idx],
        "split_group": np.asarray(groups)[te_idx],
    })
    for j, c in enumerate(classes):
        pred[f"y_{c}"] = Y[te_idx][:, j]
        pred[f"mask_{c}"] = label_mask_full[te_idx][:, j].astype(int)
        pred[f"prob_{c}"] = prob_te[:, j]
    pred.to_csv(args.output_dir / "test_predictions.csv", index=False)

    sp_rows = []
    te_species = genes["species"].astype(str).to_numpy()[te_idx]
    for sp in sorted(np.unique(te_species)):
        m = te_species == sp
        if int(m.sum()) < 10:
            continue
        _, ov = evaluate(Y[te_idx][m], prob_te[m], classes, thresholds, label_mask=label_mask_full[te_idx][m])
        ov["species"] = sp
        ov["n"] = int(m.sum())
        sp_rows.append(ov)
    if sp_rows:
        pd.DataFrame(sp_rows).to_csv(args.output_dir / "per_species_metrics.csv", index=False)

    if models is not None:
        import joblib
        joblib.dump({"scaler": scaler, "models": models}, args.output_dir / "classifier.joblib")

    json.dump(classes, open(args.output_dir / "label_classes.json", "w", encoding="utf-8"))
    config_for_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config_for_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    genes.drop(columns=["sequence"]).to_csv(args.output_dir / "gene_table.csv", index=False)

    print("\n==== TEST per-label ====")
    print(per.to_string(index=False))
    print("\n==== overall (model vs baselines) ====")
    print(pd.DataFrame([overall, ov_prior, ov_zero]).to_string(index=False))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
