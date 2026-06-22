#!/usr/bin/env python3
"""
Run the FINAL shipped model on new sequences.

Loads a train-on-all artifact directory produced by
  train_rnafm_multilabel.py --train-on-all ...
(which saved classifier.joblib + label_thresholds.csv + label_classes.json +
run_config.json) and scores an input CSV of transcripts, reproducing the exact
preprocessing (region routing, k-mer or frozen-FM features) from run_config.json.

Supported artifacts: frozen-feature + logistic/mlp head (classifier.joblib) and the
proposed fusion model (--arch fusion, which writes fusion_model.joblib). --arch nets
(rnatracker/dm3loc) and --finetune do not save a deployable head and are not the
intended deployment path.

Usage:
    python scripts/predict.py --artifact-dir results/final_best \
        --input new_transcripts.csv --output predictions.csv
Input CSV needs at least: sequence  (and sequence_type/transcript_id/source if the
artifact's region != full, so 3'UTR/CDS can be extracted from full-length rows).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("trainmod", os.path.join(_HERE, "train_rnafm_multilabel.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, required=True,
                    help="a train-on-all run dir (has classifier.joblib + run_config.json)")
    ap.add_argument("--input", type=Path, required=True, help="CSV with a 'sequence' column")
    ap.add_argument("--output", type=Path, default=Path("predictions.csv"))
    ap.add_argument("--model-dir", default=None, help="override FM dir (else from run_config)")
    ap.add_argument("--gtf", nargs="*", default=None, help="override GTF(s) for region extraction")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import joblib
    cfg = json.loads((args.artifact_dir / "run_config.json").read_text(encoding="utf-8"))
    classes = json.loads((args.artifact_dir / "label_classes.json").read_text(encoding="utf-8"))
    thr = pd.read_csv(args.artifact_dir / "label_thresholds.csv")
    thresholds = thr.set_index("label")["threshold"].reindex(classes).to_numpy()

    # Fusion is triggered either by --arch fusion or by the --fusion flag (arch
    # stays "fm" with --features). Detect both so predict.py works across versions.
    is_fusion = bool(cfg.get("fusion")) or cfg.get("arch") == "fusion"
    if is_fusion:
        scaler = models = None  # fusion has its own artifact + scoring path
    else:
        bundle_path = args.artifact_dir / "classifier.joblib"
        if not bundle_path.exists():
            raise SystemExit(
                f"[error] {bundle_path} not found. predict.py supports frozen-feature + "
                "logistic/mlp + fusion artifacts only (not --arch nets / --finetune)."
            )
        bundle = joblib.load(bundle_path)
        scaler, models = bundle["scaler"], bundle["models"]

    region = cfg.get("region", "full")
    df = pd.read_csv(args.input, low_memory=False)
    if "sequence" not in df.columns:
        raise SystemExit("[error] input CSV needs a 'sequence' column.")
    for col, default in (("sequence_type", ""), ("transcript_id", ""), ("source", "")):
        if col not in df.columns:
            df[col] = default
    df["sequence"] = df["sequence"].map(T.clean_seq)

    # Region harmonization identical to training.
    if region != "full":
        gtf = args.gtf or cfg.get("gtf") or []
        L5, L3 = T.build_utr_length_map(gtf)
        native = cfg.get("native_region_sources") or []
        df = T.apply_region(df, region, L5, L3, native_sources=native)
        if len(df) == 0:
            raise SystemExit(f"[error] region={region}: 0 input rows survived harmonization.")

    seqs = df["sequence"].tolist()

    # Featurize exactly as the artifact was trained.
    def _fm_block(seqs_):
        model_dir = args.model_dir or cfg.get("model_dir")
        device = args.device or cfg.get("device", "cpu")
        import torch
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        tok, model = T.load_model(model_dir, device)
        return T.embed_sequences(seqs_, tok, model, device,
                                 max_tokens=int(cfg.get("max_tokens", 1024)),
                                 batch_size=int(cfg.get("batch_size", 8)),
                                 pool=cfg.get("pool", "mean"),
                                 window_pool=cfg.get("window_pool", "mean"))

    if cfg.get("arch", "fm") in ("rnatracker", "dm3loc") or cfg.get("finetune"):
        raise SystemExit("[error] this artifact is --arch nets/--finetune; not supported by predict.py.")
    elif is_fusion:
        # Rebuild the SAME ordered feature views fusion was trained on, kept as a
        # list (not concatenated); per-view scaling + the fused net live in the artifact.
        from engineered_features import build_feature_block
        from fusion_model import score_fusion
        blocks = []
        for fblk in cfg["features"]:
            blocks.append(_fm_block(seqs) if fblk == "fm"
                          else build_feature_block(fblk, seqs, kmer_k=int(cfg.get("kmer_k", 4)))[0])
        blocks = [b.astype(np.float32) for b in blocks]
        device = args.device or cfg.get("device", "cpu")
        prob = score_fusion(args.artifact_dir, blocks, device=device)
    else:
        if cfg.get("features"):
            from engineered_features import build_feature_block
            blocks = []
            for fblk in cfg["features"]:
                blocks.append(_fm_block(seqs) if fblk == "fm"
                              else build_feature_block(fblk, seqs, kmer_k=int(cfg.get("kmer_k", 4)))[0])
            X = np.concatenate([b.astype(np.float32) for b in blocks], axis=1)
        elif cfg.get("baseline") == "kmer":
            X = T.kmer_features(seqs, int(cfg.get("kmer_k", 4)))
        else:
            model_dir = args.model_dir or cfg.get("model_dir")
            device = args.device or cfg.get("device", "cpu")
            import torch
            if str(device).startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            tok, model = T.load_model(model_dir, device)
            X = T.embed_sequences(seqs, tok, model, device,
                                  max_tokens=int(cfg.get("max_tokens", 1024)),
                                  batch_size=int(cfg.get("batch_size", 8)),
                                  pool=cfg.get("pool", "mean"),
                                  window_pool=cfg.get("window_pool", "mean"))
        prob = T.predict_head(scaler, models, X, len(classes))
    pred = (prob >= np.asarray(thresholds, dtype=float).reshape(1, -1)).astype(int)

    out = df.drop(columns=["sequence"]).reset_index(drop=True)
    for j, c in enumerate(classes):
        out[f"prob_{c}"] = prob[:, j]
        out[f"pred_{c}"] = pred[:, j]
    out.to_csv(args.output, index=False)
    print(f"[predict] {len(out)} rows scored with classes={classes} -> {args.output}")


if __name__ == "__main__":
    main()
