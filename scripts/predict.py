#!/usr/bin/env python3
"""
Run the FINAL shipped model on new sequences.

Loads a train-on-all artifact directory produced by
  train_rnafm_multilabel.py --train-on-all ...
(which saved classifier.joblib + label_thresholds.csv + label_classes.json +
run_config.json) and scores an input CSV of transcripts, reproducing the exact
preprocessing (region routing, k-mer or frozen-FM features) from run_config.json.

Only the frozen-feature + logistic/mlp head artifacts are supported here (the ones
that write classifier.joblib). --arch nets and --finetune do not save a joblib head
and are not the intended deployment path.

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

    bundle_path = args.artifact_dir / "classifier.joblib"
    if not bundle_path.exists():
        raise SystemExit(
            f"[error] {bundle_path} not found. predict.py supports frozen-feature + "
            "logistic/mlp artifacts only (not --arch / --finetune)."
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
    if cfg.get("baseline") == "kmer":
        X = T.kmer_features(seqs, int(cfg.get("kmer_k", 4)))
    elif cfg.get("arch", "fm") in ("rnatracker", "dm3loc") or cfg.get("finetune"):
        raise SystemExit("[error] this artifact is --arch/--finetune; not supported by predict.py.")
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
