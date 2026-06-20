#!/usr/bin/env python3
"""
Select the best benchmark configuration on VALIDATION (never test).

Walks a results directory, reads every run's overall_metrics.csv + run_config.json,
ranks the actual models (not baselines) by a threshold-free VALIDATION metric, and
writes model_selection.csv. The winning config is what you then retrain on all data
(train_rnafm_multilabel.py --train-on-all) to ship.

IMPORTANT: this script deliberately ignores the test split. Picking a model by its
test score is selection-on-test and inflates the reported number.

Usage:
    python scripts/select_best.py --results-dir results --metric macro_pr_auc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASELINE_MODELS = {"label_prior_probability", "all_zero"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--metric", default="macro_pr_auc",
                    choices=["macro_pr_auc", "macro_roc_auc", "mean_pr_lift", "macro_f1"],
                    help="threshold-free metrics (pr/roc auc) are the honest selectors")
    ap.add_argument("--group-by", nargs="*", default=[],
                    help="optional config keys to pick a winner WITHIN each group, "
                    "e.g. --group-by region sample_level")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for ov_path in sorted(args.results_dir.rglob("overall_metrics.csv")):
        run_dir = ov_path.parent
        try:
            ov = pd.read_csv(ov_path)
        except Exception as e:
            print(f"[skip] {ov_path}: {e}")
            continue
        if "split" not in ov.columns:
            print(f"[skip] {run_dir}: no 'split' column (re-run with updated trainer to log val).")
            continue
        val = ov[(ov["split"] == "val") & (~ov["model"].isin(BASELINE_MODELS))]
        if val.empty:
            continue
        r = val.iloc[0].to_dict()
        cfg = {}
        cfg_path = run_dir / "run_config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
        rows.append({
            "run_dir": str(run_dir),
            "model": r.get("model"),
            "region": cfg.get("region"),
            "sample_level": cfg.get("sample_level"),
            "arch": cfg.get("arch"),
            "baseline": cfg.get("baseline"),
            "model_dir": cfg.get("model_dir"),
            "label_scheme": cfg.get("label_scheme"),
            "species": ",".join(cfg.get("species") or []) if cfg.get("species") else "all",
            "val_macro_pr_auc": r.get("macro_pr_auc"),
            "val_macro_roc_auc": r.get("macro_roc_auc"),
            "val_mean_pr_lift": r.get("mean_pr_lift"),
            "val_macro_f1": r.get("macro_f1"),
            "n_val": r.get("n_val"),
        })

    if not rows:
        raise SystemExit(
            f"No val metrics found under {args.results_dir}. Re-run training with the "
            "updated trainer (it now logs a val row in overall_metrics.csv)."
        )

    df = pd.DataFrame(rows)
    metric_col = {"macro_pr_auc": "val_macro_pr_auc", "macro_roc_auc": "val_macro_roc_auc",
                  "mean_pr_lift": "val_mean_pr_lift", "macro_f1": "val_macro_f1"}[args.metric]
    df = df.sort_values(metric_col, ascending=False, na_position="last").reset_index(drop=True)

    out = args.out or (args.results_dir / "model_selection.csv")
    df.to_csv(out, index=False)

    print(f"=== model selection by VALIDATION {args.metric} (test untouched) ===")
    show = ["model", "region", "sample_level", "label_scheme", metric_col, "val_macro_roc_auc", "n_val"]
    print(df[show].head(20).to_string(index=False))

    if args.group_by:
        print(f"\n=== best per {args.group_by} ===")
        best = df.sort_values(metric_col, ascending=False).groupby(args.group_by, dropna=False).head(1)
        print(best[args.group_by + ["model", metric_col]].to_string(index=False))

    win = df.iloc[0]
    print(f"\n>>> SELECTED: {win['model']}  ({win['region']}/{win['sample_level']})  "
          f"{args.metric}={win[metric_col]}")
    print(f">>> run_dir: {win['run_dir']}")
    print(">>> Ship it: rerun that exact config with --train-on-all to fit on all data,")
    print(">>> then predict with scripts/predict.py. Report the TEST number from the")
    print(">>> benchmark run above, NOT the train-on-all in-sample number.")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
