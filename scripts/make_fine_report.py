#!/usr/bin/env python3
"""
Fine-task (5-compartment multi-label) results report — statistics + figures for the
five results sections. Reads the per-run outputs produced by run_all.sh `fine` and
`fineablation`, builds tidy tables, runs leakage-safe group bootstraps (macro over
the five compartments and per-compartment), and renders the section figures.

Settings (region x granularity), main benchmark = full x isoform:
    fine_utr3_gene    fine_cds_gene    fine_full_gene    fine_full_isoform
Compartments: Cell_body, Dendrite, Neuropil, Axon, Neurite.

Tables written under --results-dir:
    fine_summary_long.csv       one row per (setting, model): val/test macro metrics
    fine_per_label_long.csv     one row per (setting, model, compartment)
    fine_bootstrap.csv          group-bootstrap comparisons (macro + per-compartment)

Figures under results/figures/:
  S2  fig_fine_modelmap_{pr_auc,roc_auc}   model x setting test-metric heatmap (* = sig > k-mer)
      fig_fine_region_ablation             matched-gene RNA-FM utr3/cds/full (region the only variable)
      fig_fine_granularity                 gene vs isoform (full) — descriptive point estimates
  S3  fig_fine_fusion_forest               fusion - k-mer macro Δ ± CI per setting
      fig_fine_components                  length/engineered/best-FM/fusion (main benchmark)
      fig_fine_gate                        gate reliance on the FM view (main benchmark fusion)
  S4  fig_fine_percompartment              per-compartment AUPRC/AUROC (main benchmark fusion)
      fig_fine_support_vs_perf             train positive support vs test AUPRC (+ Spearman ρ)
      fig_fine_percompartment_gain         per-compartment fusion - k-mer Δ ± CI
  S5  fig_fine_perspecies                  macro metrics on mouse/rat/human test subsets

Usage:
    python scripts/make_fine_report.py --results-dir results --n-boot 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import bootstrap_compare, bootstrap_compare_macro  # noqa: E402

# ---- configuration ---------------------------------------------------------
# (dir, region label, granularity, fusion subdir).  Main benchmark = last entry.
SETTINGS = [
    ("fine_utr3_gene", "3'UTR", "gene", "fusion_rnafm_eng"),
    ("fine_cds_gene", "CDS", "gene", "fusion_mrnafm_eng"),
    ("fine_full_gene", "full", "gene", "fusion_rnafm_eng"),
    ("fine_full_isoform", "full", "isoform", "fusion_rnafm_eng"),
]
MAIN = "fine_full_isoform"
COMPARTMENTS = ["Cell_body", "Dendrite", "Neuropil", "Axon", "Neurite"]
FM_VIEWS = ["utrbert", "rnafm", "mrnafm", "dnabert2"]   # single foundation-model views
MODEL_ORDER = ["kmer", "length", "engineered", "rnatracker", "dm3loc",
               "utrbert", "rnafm", "mrnafm", "dnabert2", "deepseek_moe",
               "fusion_rnafm_eng", "fusion_mrnafm_eng"]
MODEL_LABEL = {"kmer": "k-mer", "length": "length", "engineered": "engineered",
               "rnatracker": "RNATracker", "dm3loc": "DM3Loc", "utrbert": "UTR-BERT",
               "rnafm": "RNA-FM", "mrnafm": "mRNA-FM", "dnabert2": "DNABERT-2",
               "deepseek_moe": "DeepSeek-MoE",
               "fusion_rnafm_eng": "fusion", "fusion_mrnafm_eng": "fusion"}
C_LLM = "#e07b39"
ABL = "fine_ablation_region"
BASELINES = {"label_prior_probability", "all_zero"}
C_FUS, C_KMER, C_FM, C_ENG, C_NET = "#d1495b", "#8d99ae", "#3a7ca5", "#5c946e", "#9b6a9e"


CANON_ORDER = ["kmer", "length", "engineered", "rnatracker", "dm3loc",
               "utrbert", "rnafm", "mrnafm", "dnabert2", "deepseek_moe", "fusion"]


def canon(model):
    """Collapse the region-specific fusion subdirs into one logical 'fusion' row."""
    return "fusion" if str(model).startswith("fusion") else model


def _color(model):
    if model.startswith("fusion"):
        return C_FUS
    if model == "kmer":
        return C_KMER
    if model in ("rnatracker", "dm3loc"):
        return C_NET
    if model in ("length", "engineered"):
        return C_ENG
    if model == "deepseek_moe":
        return C_LLM
    return C_FM


# ---- loaders ---------------------------------------------------------------
def _overall(run_dir: Path):
    """Return (test_row, val_row, prior_macro) dicts/values or (None, None, nan)."""
    p = run_dir / "overall_metrics.csv"
    if not p.exists():
        return None, None, np.nan
    ov = pd.read_csv(p)
    if "split" not in ov.columns:
        ov["split"] = "test"
    test = ov[(ov.split == "test") & (~ov.model.isin(BASELINES))]
    val = ov[(ov.split == "val") & (~ov.model.isin(BASELINES))]
    prior = ov[(ov.split == "test") & (ov.model == "label_prior_probability")]
    return (test.iloc[0].to_dict() if len(test) else None,
            val.iloc[0].to_dict() if len(val) else None,
            float(prior.iloc[0]["macro_pr_auc"]) if len(prior) else np.nan)


def load_summary(results: Path):
    """summary_long: one row per (setting, model) with val/test macro metrics."""
    rows, per_rows = [], []
    for sdir, region, gran, _ in SETTINGS:
        base = results / sdir
        if not base.exists():
            continue
        # train positive support per compartment (model-independent; read once)
        support = {}
        for m in MODEL_ORDER:
            sp = base / m / "split_label_support.csv"
            if sp.exists():
                t = pd.read_csv(sp)
                t = t[t.split == "train"]
                support = {r["label"]: int(r["positives"]) for _, r in t.iterrows()}
                break
        for m in sorted(os.listdir(base)):
            run = base / m
            if not (run / "overall_metrics.csv").exists():
                continue
            test, val, prior = _overall(run)
            if test is None:
                continue
            rows.append({
                "setting": sdir, "region": region, "granularity": gran, "model": m,
                "val_pr_auc": (val or {}).get("macro_pr_auc", np.nan),
                "val_roc_auc": (val or {}).get("macro_roc_auc", np.nan),
                "test_pr_auc": test.get("macro_pr_auc", np.nan),
                "test_roc_auc": test.get("macro_roc_auc", np.nan),
                "test_macro_f1": test.get("macro_f1", np.nan),
                "prior_pr_auc": prior, "n_test": test.get("n_test", np.nan),
            })
            pl = run / "per_label_metrics.csv"
            if pl.exists():
                for _, r in pd.read_csv(pl).iterrows():
                    per_rows.append({
                        "setting": sdir, "region": region, "granularity": gran, "model": m,
                        "label": r["label"], "roc_auc": r.get("roc_auc", np.nan),
                        "pr_auc": r.get("pr_auc", np.nan), "f1": r.get("f1", np.nan),
                        "test_support": r.get("support", np.nan), "prior": r.get("prior", np.nan),
                        "train_positives": support.get(r["label"], np.nan),
                    })
    return pd.DataFrame(rows), pd.DataFrame(per_rows)


def best_fm(summ, setting, by="val_pr_auc"):
    """val-selected best single foundation-model view present in a setting."""
    d = summ[(summ.setting == setting) & (summ.model.isin(FM_VIEWS))]
    if d.empty or d[by].isna().all():
        return None
    return d.sort_values(by, ascending=False).iloc[0]["model"]


# ---- bootstrap battery -----------------------------------------------------
def run_bootstraps(results: Path, summ, n_boot, seed):
    rows = []

    def add(group, setting, a_sub, b_sub, r):
        if r:
            rows.append({"group": group, "setting": setting, "A": a_sub, "B": b_sub, **r})

    for sdir, _, _, fus in SETTINGS:
        base = results / sdir
        if not (base / "kmer").exists():
            continue
        present = set(summ[summ.setting == sdir].model)
        # macro: every non-k-mer model vs k-mer (supports the negative result + fusion)
        for m in present:
            if m == "kmer":
                continue
            for metric in ("pr_auc", "roc_auc"):
                add("vs_kmer", sdir, m, "kmer",
                    bootstrap_compare_macro(base / m, base / "kmer", COMPARTMENTS,
                                            metric, n_boot, seed))
        # macro: fusion vs its components (main fusion of this setting)
        bf = best_fm(summ, sdir)
        for comp in [c for c in ("length", "engineered", bf) if c and c in present]:
            for metric in ("pr_auc", "roc_auc"):
                add("component", sdir, fus, comp,
                    bootstrap_compare_macro(base / fus, base / comp, COMPARTMENTS,
                                            metric, n_boot, seed))
        # per-compartment: fusion vs k-mer
        if fus in present:
            for lab in COMPARTMENTS:
                for metric in ("pr_auc", "roc_auc"):
                    add("percompartment", sdir, fus, "kmer",
                        _label_row(base / fus, base / "kmer", lab, metric, n_boot, seed))

    # region ablation (matched genes, RNA-FM, region the only variable)
    abase = results / ABL
    if abase.exists():
        reg = {r: abase / f"rnafm_{r}" for r in ("utr3", "cds", "full")}
        for x, y in [("full", "utr3"), ("cds", "utr3"), ("full", "cds")]:
            if reg[x].exists() and reg[y].exists():
                for metric in ("pr_auc", "roc_auc"):
                    add("region", ABL, f"rnafm_{x}", f"rnafm_{y}",
                        bootstrap_compare_macro(reg[x], reg[y], COMPARTMENTS,
                                                metric, n_boot, seed))
    return pd.DataFrame(rows)


def _label_row(a, b, label, metric, n_boot, seed):
    r = bootstrap_compare(a, b, label=label, metric=metric, n_boot=n_boot, seed=seed)
    if r:
        r["label"] = label
    return r


# ---- figures ---------------------------------------------------------------
def _save(fig, out_base):
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {out_base}.png / .pdf")


def fig_modelmap(summ, boot, metric, outdir):
    import matplotlib.pyplot as plt
    col = f"test_{metric}"
    settings = [s for s in SETTINGS if s[0] in set(summ.setting)]
    present_canon = set(summ.model.map(canon))
    models = [m for m in CANON_ORDER if m in present_canon]
    if not settings or not models:
        return
    M = np.full((len(models), len(settings)), np.nan)
    sig = {}
    bk = boot[(boot.group == "vs_kmer") & (boot.metric == metric)] if not boot.empty else pd.DataFrame()
    bk = bk.assign(canonA=bk.A.map(canon)) if len(bk) else bk
    for j, (sdir, *_rest) in enumerate(settings):
        for i, m in enumerate(models):
            d = summ[(summ.setting == sdir) & (summ.model.map(canon) == m)]
            if len(d):
                M[i, j] = float(d[col].iloc[0])
            if len(bk):
                rr = bk[(bk.setting == sdir) & (bk.canonA == m)]
                if len(rr):
                    sig[(i, j)] = bool(rr.iloc[0]["significant"] and rr.iloc[0]["diff"] > 0)

    fig, ax = plt.subplots(figsize=(1.6 * len(settings) + 2.5, 0.5 * len(models) + 1.8))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels([f"{r}\n{g}" for _, r, g, _ in settings])
    ax.set_yticks(range(len(models))); ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in models])
    for j in range(len(settings)):
        col_vals = M[:, j]
        best_i = int(np.nanargmax(col_vals)) if np.isfinite(col_vals).any() else -1
        for i in range(len(models)):
            if not np.isfinite(M[i, j]):
                continue
            star = "*" if sig.get((i, j)) else ""
            txt = f"{M[i, j]:.3f}{star}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] < np.nanmean(M) else "black",
                    fontweight="bold" if i == best_i else "normal")
        if best_i >= 0:
            ax.add_patch(plt.Rectangle((j - 0.5, best_i - 0.5), 1, 1, fill=False,
                                       edgecolor="red", lw=1.6))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(f"test macro-{metric.upper().replace('_', '-')}")
    ax.set_title(f"Model × setting (red box = best/col; * = sig > k-mer, group bootstrap)\n"
                 f"macro-{metric.upper().replace('_', '-')}", fontsize=10)
    _save(fig, str(outdir / f"fig_fine_modelmap_{metric}"))


def fig_vs_kmer_delta(summ, boot, metric, outdir):
    """Headline negative-result figure: per setting, every model's macro Δ vs k-mer
    (with group-bootstrap CI). Zero line = k-mer. Almost all models cluster at ~0;
    only fusion clears it — the 'no single representation beats k-mer' story, visible
    at a glance instead of buried in a heatmap's color."""
    import matplotlib.pyplot as plt
    if boot.empty:
        print("  [skip] fig_fine_vs_kmer_delta: needs bootstrap (run without --no-bootstrap "
              "or provide a cached fine_bootstrap.csv)")
        return
    sub = boot[(boot.group == "vs_kmer") & (boot.metric == metric)].copy()
    if sub.empty:
        return
    sub["mc"] = sub.A.map(canon)
    settings = [s for s in SETTINGS if s[0] in set(sub.setting)]
    models = [m for m in CANON_ORDER if m != "kmer" and m in set(sub.mc)]
    yof = {m: len(models) - 1 - i for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, len(settings), figsize=(3.4 * len(settings) + 1.2, 4.4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for k, (sdir, region, gran, _) in enumerate(settings):
        ax = axes[k]
        d = sub[sub.setting == sdir]
        for _, r in d.iterrows():
            yi = yof[r.mc]
            ax.errorbar(r["diff"], yi, xerr=[[r["diff"] - r.ci_lo], [r.ci_hi - r["diff"]]],
                        fmt="o", color=_color(r.A), capsize=2.5, ms=5)
            if r.significant and r["diff"] > 0:
                ax.text(r.ci_hi, yi, " *", va="center", fontsize=10, fontweight="bold")
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"{region} {gran}", fontsize=10)
        ax.set_xlabel(f"Δ macro-{metric.upper().replace('_', '-')}")
        ax.grid(axis="x", ls=":", alpha=0.4)
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels([MODEL_LABEL.get(m, m) for m in models][::-1])
    fig.suptitle(f"Each model vs k-mer (Δ macro-{metric.upper().replace('_', '-')}, "
                 f"group-bootstrap 95% CI; 0 = k-mer, * = significantly above)", fontsize=11)
    fig.tight_layout()
    _save(fig, str(outdir / f"fig_fine_vs_kmer_delta_{metric}"))


def fig_region_ablation(results, boot, outdir):
    import matplotlib.pyplot as plt
    abase = results / ABL
    regions = [("utr3", "3'UTR"), ("cds", "CDS"), ("full", "full")]
    vals = {}
    for r, _ in regions:
        test, _, _ = _overall(abase / f"rnafm_{r}")
        if test:
            vals[r] = (test.get("macro_pr_auc", np.nan), test.get("macro_roc_auc", np.nan))
    if not vals:
        print("  [skip] fig_fine_region_ablation: no fine_ablation_region runs "
              "(run: bash scripts/run_all.sh fineablation)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for k, (mi, mname) in enumerate([(0, "AUPRC"), (1, "AUROC")]):
        ax = axes[k]
        ys = [vals.get(r, (np.nan, np.nan))[mi] for r, _ in regions]
        ax.bar([lab for _, lab in regions], ys, color=[C_FM, C_FM, C_FUS],
               edgecolor="black", linewidth=0.4)
        ax.set_ylabel(f"macro-{mname}")
        if np.isfinite(ys).any():
            ax.set_ylim(min(0.5, np.nanmin(ys) - 0.02), np.nanmax(ys) + 0.04)
        metric = "pr_auc" if mi == 0 else "roc_auc"
        ra = boot[(boot.group == "region") & (boot.metric == metric)] if not boot.empty else pd.DataFrame()
        txt = [f"{r.A.replace('rnafm_', '')} vs {r.B.replace('rnafm_', '')}: "
               f"Δ={r['diff']:+.3f} p={r.p:.3f}{'*' if r.significant else ''}"
               for _, r in ra.iterrows()]
        if txt:
            ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=7)
    fig.suptitle("Region ablation — matched genes, RNA-FM (region is the only variable)")
    fig.tight_layout()
    _save(fig, str(outdir / "fig_fine_region_ablation"))


def fig_granularity(summ, outdir):
    import matplotlib.pyplot as plt
    pairs = {"gene": "fine_full_gene", "isoform": "fine_full_isoform"}
    if not all(s in set(summ.setting) for s in pairs.values()):
        print("  [skip] fig_fine_granularity: need both fine_full_gene and fine_full_isoform")
        return
    present_canon = set(summ[summ.setting.isin(pairs.values())].model.map(canon))
    models = [m for m in CANON_ORDER if m in present_canon]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, metric in enumerate(("pr_auc", "roc_auc")):
        ax = axes[k]; col = f"test_{metric}"
        x = np.arange(len(models)); w = 0.38
        for gi, (gname, sdir) in enumerate(pairs.items()):
            d = summ[summ.setting == sdir].copy()
            d["mc"] = d.model.map(canon)
            d = d.drop_duplicates("mc").set_index("mc")
            ys = [float(d.loc[m, col]) if m in d.index else np.nan for m in models]
            ax.bar(x + (gi - 0.5) * w, ys, width=w, label=gname,
                   color=C_FM if gi == 0 else C_FUS, edgecolor="black", linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models],
                                             rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(f"test macro-{metric.upper().replace('_', '-')}")
        ax.legend(title="granularity (full)", fontsize=8)
    fig.suptitle("Gene-representative vs sequence-resolved isoform (full-length) — "
                 "descriptive (different splits; no shared-sample test)", fontsize=10)
    fig.tight_layout()
    _save(fig, str(outdir / "fig_fine_granularity"))


def fig_fusion_forest(boot, outdir):
    import matplotlib.pyplot as plt
    if boot.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    settings = [s for s in SETTINGS]
    for k, metric in enumerate(("pr_auc", "roc_auc")):
        ax = axes[k]
        sub = boot[(boot.group == "vs_kmer") & (boot.metric == metric) &
                   (boot.A.str.startswith("fusion"))]
        order = [s[0] for s in settings if s[0] in set(sub.setting)]
        sub = sub.set_index("setting").reindex(order).reset_index()
        if sub.empty:
            continue
        y = np.arange(len(sub))[::-1]
        ax.errorbar(sub["diff"], y, xerr=[sub["diff"] - sub.ci_lo, sub.ci_hi - sub["diff"]],
                    fmt="o", color=C_FUS, capsize=3)
        ax.axvline(0, color="black", lw=0.8)
        labmap = {s[0]: f"{s[1]} {s[2]}" for s in settings}
        ax.set_yticks(y); ax.set_yticklabels([labmap[s] for s in sub.setting], fontsize=8)
        ax.set_xlabel(f"Δ macro-{metric.upper().replace('_', '-')}  (fusion − k-mer)")
        for yi, (_, r) in zip(y, sub.iterrows()):
            ax.text(r.ci_hi, yi, " *" if r.significant else " ns", va="center", fontsize=8)
    fig.suptitle("Fusion vs k-mer (group bootstrap, 95% CI)")
    fig.tight_layout()
    _save(fig, str(outdir / "fig_fine_fusion_forest"))


def fig_components(summ, boot, outdir):
    import matplotlib.pyplot as plt
    sdir = MAIN
    if sdir not in set(summ.setting):
        print(f"  [skip] fig_fine_components: {MAIN} missing")
        return
    bf = best_fm(summ, sdir) or "rnafm"
    fus = next((s[3] for s in SETTINGS if s[0] == sdir), "fusion_rnafm_eng")
    comps = [("length", "length", C_ENG), ("engineered", "engineered", C_ENG),
             (bf, MODEL_LABEL.get(bf, bf), C_FM), (fus, "fusion", C_FUS)]
    d = summ[summ.setting == sdir].set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, metric in enumerate(("pr_auc", "roc_auc")):
        ax = axes[k]; col = f"test_{metric}"
        ys = [float(d.loc[m, col]) if m in d.index else np.nan for m, _, _ in comps]
        ax.bar([lab for _, lab, _ in comps], ys, color=[c for *_, c in comps],
               edgecolor="black", linewidth=0.3)
        ax.set_ylabel(f"test macro-{metric.upper().replace('_', '-')}")
        if np.isfinite(ys).any():
            ax.set_ylim(min(0.5, np.nanmin(ys) - 0.02), np.nanmax(ys) + 0.04)
        cc = boot[(boot.group == "component") & (boot.metric == metric) &
                  (boot.setting == sdir)] if not boot.empty else pd.DataFrame()
        txt = [f"fusion vs {r.B}: Δ={r['diff']:+.3f} p={r.p:.3f}{'*' if r.significant else ''}"
               for _, r in cc.iterrows()]
        if txt:
            ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=7)
    fig.suptitle(f"Component decomposition — main benchmark ({MAIN})")
    fig.tight_layout()
    _save(fig, str(outdir / "fig_fine_components"))


def fig_gate(results, outdir):
    import matplotlib.pyplot as plt
    fus = next((s[3] for s in SETTINGS if s[0] == MAIN), "fusion_rnafm_eng")
    run = results / MAIN / fus
    gp, pp = run / "gate_weights.csv", run / "test_predictions.csv"
    if not gp.exists() or not pp.exists():
        print("  [skip] fig_fine_gate: no gate_weights.csv (fusion run not present)")
        return
    gate, preds = pd.read_csv(gp), pd.read_csv(pp)
    if len(gate) != len(preds):
        print("  [skip] fig_fine_gate: gate/preds length mismatch")
        return
    wcols = [c for c in gate.columns if c.startswith("w_")]
    fmcol = "w_fm" if "w_fm" in gate.columns else wcols[0]
    wfm = gate[fmcol].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(wfm, bins=30, color=C_FM, alpha=0.85, edgecolor="black", linewidth=0.3)
    axes[0].axvline(wfm.mean(), color="black", ls="--", lw=1, label=f"mean {wfm.mean():.2f}")
    axes[0].axvline(0.5, color="grey", ls=":", lw=1)
    axes[0].set_xlabel(f"gate weight on FM view ({fmcol})"); axes[0].set_ylabel("transcripts")
    axes[0].set_title("a  Gate reliance on FM view"); axes[0].legend(fontsize=8)
    means = []
    for lab in COMPARTMENTS:
        yk, mk = f"y_{lab}", f"mask_{lab}"
        if yk in preds.columns:
            m = (preds[mk] == 1) & (preds[yk] == 1) if mk in preds.columns else (preds[yk] == 1)
            means.append(float(np.nanmean(wfm[m.to_numpy()])) if m.any() else np.nan)
        else:
            means.append(np.nan)
    axes[1].bar(COMPARTMENTS, means, color=C_FM, edgecolor="black", linewidth=0.3)
    axes[1].axhline(wfm.mean(), color="black", ls="--", lw=1)
    axes[1].set_ylabel("mean FM-view weight (positives)")
    axes[1].set_xticks(range(len(COMPARTMENTS)))
    axes[1].set_xticklabels(COMPARTMENTS, rotation=40, ha="right", fontsize=8)
    axes[1].set_title("b  FM reliance by compartment")
    fig.suptitle(f"Gate interpretation — {MAIN} fusion (which view the model leans on)")
    fig.tight_layout()
    _save(fig, str(outdir / "fig_fine_gate"))


def fig_percompartment(per, outdir):
    import matplotlib.pyplot as plt
    fus = next((s[3] for s in SETTINGS if s[0] == MAIN), "fusion_rnafm_eng")
    d = per[(per.setting == MAIN) & (per.model == fus)].set_index("label").reindex(COMPARTMENTS)
    if d["pr_auc"].isna().all():
        print(f"  [skip] fig_fine_percompartment: no per-label metrics for {MAIN}/{fus}")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(COMPARTMENTS)); w = 0.38
    ax.bar(x - w / 2, d["pr_auc"], width=w, label="AUPRC", color=C_FUS, edgecolor="black", linewidth=0.3)
    ax.bar(x + w / 2, d["roc_auc"], width=w, label="AUROC", color=C_FM, edgecolor="black", linewidth=0.3)
    for xi, p in zip(x, d["prior"]):
        if np.isfinite(p):
            ax.plot([xi - w, xi], [p, p], color="grey", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(COMPARTMENTS, rotation=30, ha="right")
    ax.set_ylabel("test metric"); ax.legend(fontsize=8)
    ax.set_title(f"Per-compartment performance — {MAIN} fusion (grey dash = label prior)")
    fig.tight_layout()
    _save(fig, str(outdir / "fig_fine_percompartment"))


def _spearman(x, y):
    """Rank correlation without scipy."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def fig_support_vs_perf(per, outdir):
    import matplotlib.pyplot as plt
    fus = next((s[3] for s in SETTINGS if s[0] == MAIN), "fusion_rnafm_eng")
    d = per[(per.setting == MAIN) & (per.model == fus)].dropna(subset=["train_positives", "pr_auc"])
    if len(d) < 3:
        print("  [skip] fig_fine_support_vs_perf: need train_positives + pr_auc (>=3 compartments)")
        return
    rho = _spearman(d["train_positives"], d["pr_auc"])
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.scatter(d["train_positives"], d["pr_auc"], color=C_FUS, s=60, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r["label"], (r["train_positives"], r["pr_auc"]), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("train positive support  n⁺_c"); ax.set_ylabel("test AUPRC")
    ax.set_title(f"Support vs performance — {MAIN} fusion  (Spearman ρ={rho:.2f})")
    fig.tight_layout()
    _save(fig, str(outdir / "fig_fine_support_vs_perf"))


def fig_percompartment_gain(boot, outdir):
    import matplotlib.pyplot as plt
    if boot.empty:
        return
    sub = boot[(boot.group == "percompartment") & (boot.setting == MAIN)]
    if sub.empty:
        print(f"  [skip] fig_fine_percompartment_gain: no per-compartment bootstrap for {MAIN}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, metric in enumerate(("pr_auc", "roc_auc")):
        ax = axes[k]
        d = sub[sub.metric == metric].set_index("label").reindex(COMPARTMENTS).reset_index()
        y = np.arange(len(COMPARTMENTS))[::-1]
        ax.errorbar(d["diff"], y, xerr=[d["diff"] - d.ci_lo, d.ci_hi - d["diff"]],
                    fmt="o", color=C_FUS, capsize=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(COMPARTMENTS, fontsize=8)
        ax.set_xlabel(f"Δ {metric.upper().replace('_', '-')}  (fusion − k-mer)")
        for yi, (_, r) in zip(y, d.iterrows()):
            if pd.notna(r.get("significant")):
                ax.text(r.ci_hi, yi, " *" if r.significant else "", va="center", fontsize=9)
    fig.suptitle(f"Per-compartment fusion gain over k-mer — {MAIN} (group bootstrap)")
    fig.tight_layout()
    _save(fig, str(outdir / "fig_fine_percompartment_gain"))


def fig_feature_importance(results, outdir, topk=15):
    """Permutation importance of engineered features in the main-benchmark fusion model
    (drop in macro-AUPRC when a feature is shuffled across test rows). Bars colored by
    direction (enriched vs depleted in positives); the whole FM view is shown for scale.
    This is the faithful, on-task interpretability figure — magnitude is the model's
    measured reliance, direction is a descriptive feature-label association."""
    import matplotlib.pyplot as plt
    fus = next((s[3] for s in SETTINGS if s[0] == MAIN), "fusion_rnafm_eng")
    p = results / MAIN / fus / "feature_importance.csv"
    if not p.exists():
        print("  [skip] fig_fine_feature_importance: no feature_importance.csv "
              "(run fusion with permutation importance enabled)")
        return
    imp = pd.read_csv(p).sort_values("drop_pr_auc", ascending=False).head(topk).iloc[::-1]
    if imp.empty:
        return
    colors = []
    for _, r in imp.iterrows():
        if pd.isna(r["direction"]):
            colors.append(C_KMER)                      # whole-view aggregate (FM/k-mer)
        else:
            colors.append(C_FUS if r["direction"] > 0 else C_FM)
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.42 * len(imp) + 1)))
    y = np.arange(len(imp))
    ax.barh(y, imp["drop_pr_auc"], color=colors, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(imp["feature"], fontsize=8)
    ax.set_xlabel("Δ macro-AUPRC when permuted  (importance)")
    ax.set_title(f"Engineered-feature permutation importance — {MAIN} fusion")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_FUS, C_FM, C_KMER)]
    ax.legend(handles, ["enriched in positives (+)", "depleted in positives (−)",
                        "whole view (FM/k-mer)"], fontsize=7.5, loc="lower right")
    fig.tight_layout()
    _save(fig, str(outdir / "fig_fine_feature_importance"))


def fig_perspecies(results, outdir):
    import matplotlib.pyplot as plt
    fus = next((s[3] for s in SETTINGS if s[0] == MAIN), "fusion_rnafm_eng")
    p = results / MAIN / fus / "per_species_metrics.csv"
    if not p.exists():
        print("  [skip] fig_fine_perspecies: no per_species_metrics.csv")
        return
    sp = pd.read_csv(p)
    order = [s for s in ["mouse", "rat", "human"] if s in set(sp.species)]
    sp = sp.set_index("species").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w / 2, sp["macro_pr_auc"], width=w, label="AUPRC", color=C_FUS, edgecolor="black", linewidth=0.3)
    ax.bar(x + w / 2, sp["macro_roc_auc"], width=w, label="AUROC", color=C_FM, edgecolor="black", linewidth=0.3)
    for xi, (_, r) in zip(x, sp.iterrows()):
        ax.text(xi, 0.02, f"n={int(r['n'])}", ha="center", fontsize=8, color="white")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("macro metric (test subset)"); ax.legend(fontsize=8)
    ax.set_title(f"Per-species test performance — {MAIN} fusion (descriptive)")
    fig.tight_layout()
    _save(fig, str(outdir / "fig_fine_perspecies"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="skip the (slow) group bootstraps; figures needing CIs are skipped")
    args = ap.parse_args()

    summ, per = load_summary(args.results_dir)
    if summ.empty:
        raise SystemExit(f"No fine_* runs with overall_metrics.csv under {args.results_dir}. "
                         "Run: bash scripts/run_all.sh fineonly")
    summ.to_csv(args.results_dir / "fine_summary_long.csv", index=False)
    per.to_csv(args.results_dir / "fine_per_label_long.csv", index=False)
    print(f"[tables] fine_summary_long.csv ({len(summ)} rows), "
          f"fine_per_label_long.csv ({len(per)} rows)")
    for sdir, *_ in SETTINGS:
        bf = best_fm(summ, sdir)
        if bf:
            print(f"  best single FM in {sdir}: {bf} (val-selected)")

    boot = pd.DataFrame()
    boot_path = args.results_dir / "fine_bootstrap.csv"
    if args.no_bootstrap:
        if boot_path.exists():            # reuse cached bootstrap so figures still get CIs
            boot = pd.read_csv(boot_path)
            print(f"[bootstrap] --no-bootstrap: loaded cached {boot_path.name} "
                  f"({len(boot)} comparisons)")
        else:
            print("[bootstrap] --no-bootstrap and no cached fine_bootstrap.csv; "
                  "CI figures will be skipped")
    else:
        print(f"[bootstrap] n_boot={args.n_boot} (group resampling by split_group) ...")
        boot = run_bootstraps(args.results_dir, summ, args.n_boot, args.seed)
        if not boot.empty:
            boot.to_csv(boot_path, index=False)
            print(f"  fine_bootstrap.csv ({len(boot)} comparisons)")
        else:
            print("  [bootstrap] no comparable runs (missing test_predictions.csv)")

    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        print("[viz] matplotlib not installed; tables only.")
        return
    outdir = args.results_dir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[figures] -> {outdir}")
    for fn in (
        lambda: fig_modelmap(summ, boot, "pr_auc", outdir),
        lambda: fig_modelmap(summ, boot, "roc_auc", outdir),
        lambda: fig_vs_kmer_delta(summ, boot, "pr_auc", outdir),
        lambda: fig_vs_kmer_delta(summ, boot, "roc_auc", outdir),
        lambda: fig_region_ablation(args.results_dir, boot, outdir),
        lambda: fig_granularity(summ, outdir),
        lambda: fig_fusion_forest(boot, outdir),
        lambda: fig_components(summ, boot, outdir),
        lambda: fig_gate(args.results_dir, outdir),
        lambda: fig_percompartment(per, outdir),
        lambda: fig_support_vs_perf(per, outdir),
        lambda: fig_percompartment_gain(boot, outdir),
        lambda: fig_feature_importance(args.results_dir, outdir),
        lambda: fig_perspecies(args.results_dir, outdir),
    ):
        try:
            fn()
        except Exception as e:
            print(f"  [warn] figure failed: {e}")
    print("done.")


if __name__ == "__main__":
    main()
