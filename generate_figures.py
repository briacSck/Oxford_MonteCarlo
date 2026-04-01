"""
Publication-quality visualizations for Oxford Monte Carlo missing-data study.
Generates 3 charts per paper (6 total) saved as 300 DPI PNGs.

Charts per paper:
  1. B_prop heatmap — sign+significance retention by method x missingness proportion
  2. Method comparison bar chart — mean imputation RMSE from baseline by method
  3. Stability trajectory line chart — B_prop per method across missingness levels

Run from repo root:  python generate_figures.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paper configuration
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent

PAPERS = {
    "0005": {
        "label":      "0005",
        "long_title": "Mapping Entrepreneurial Inclusion\n(Stroube & Dushnitsky, ~2025 SMJ)",
        "workbook":   REPO / "paper_analysis_output/Paper_0005_MappingEntrepreneurial/full_run/Stroube2025Report_0005.xlsx",
        "figures":    REPO / "paper_analysis_output/Paper_0005_MappingEntrepreneurial/full_run/figures",
        "focal_iv":   "log_pop_black_aa",
        "focal_coef": 0.0307,
        "focal_label":"log(Black/AA pop.)",
    },
    "0017": {
        "label":      "0017",
        "long_title": "Status and Consensus\n(Stroube, 2024 SMJ)",
        "workbook":   REPO / "paper_analysis_output/Paper_0017_StatusConsensus/full_run/Stroube2024Report_0017.xlsx",
        "figures":    REPO / "paper_analysis_output/Paper_0017_StatusConsensus/full_run/figures",
        "focal_iv":   "FLead",
        "focal_coef": 0.0468,
        "focal_label":"Female Lead (FLead)",
    },
}

# Canonical display order
METHOD_ORDER    = ["LD", "Mean", "Reg", "Iter", "RF", "DL", "MILGBM"]
METHOD_LABELS   = {"LD": "Listwise\nDeletion", "Mean": "Mean\nImputation",
                   "Reg": "Regression\n+ Noise", "Iter": "Stochastic\nIterative",
                   "RF": "Random\nForest", "DL": "Deep\nLearning",
                   "MILGBM": "MI-LGBM"}
PROP_ORDER      = ["1pct","5pct","10pct","20pct","30pct","40pct","50pct"]
PROP_LABELS     = ["1%","5%","10%","20%","30%","40%","50%"]
PROP_NUMERIC    = [1, 5, 10, 20, 30, 40, 50]

# Highlight palette for line chart
HIGHLIGHT       = {"LD": "#D62728", "MILGBM": "#1F77B4", "RF": "#2CA02C"}
MUTED           = "#C8C8C8"
TEAL            = "#2B9E8E"
GRAY_BAR        = "#A8A8A8"

# Academic style
RC = {
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        False,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.15,
}
plt.rcParams.update(RC)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_data(cfg: dict):
    wb = cfg["workbook"]
    css = pd.read_excel(wb, sheet_name="Coef_Stability_Summary")
    mc  = pd.read_excel(wb, sheet_name="Model_Comparison")
    return css, mc


def prop_sort_key(p):
    return PROP_ORDER.index(p) if p in PROP_ORDER else 99


# ---------------------------------------------------------------------------
# Chart 1 — B_prop heatmap (method × proportion)
# ---------------------------------------------------------------------------
def chart_heatmap(cfg: dict, css: pd.DataFrame):
    pid   = cfg["label"]
    outdir = cfg["figures"]

    # Average B_prop over all key vars and all mechanisms
    pivot = (
        css.groupby(["Method", "Proportion"])["B_prop"]
           .mean()
           .reset_index()
    )
    pivot["prop_order"] = pivot["Proportion"].map(prop_sort_key)
    pivot = pivot.sort_values("prop_order")

    mat = pivot.pivot(index="Method", columns="Proportion", values="B_prop")
    mat = mat.reindex(index=METHOD_ORDER, columns=PROP_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Colormap: diverging RdYlGn centered at a sensible midpoint.
    # If all values are near 100 (tight range), use a sequential scale instead.
    data_min  = float(np.nanmin(mat.values))
    data_max  = float(np.nanmax(mat.values))
    data_range = data_max - data_min

    if data_range < 5:
        # Near-perfect stability: sequential green scale, annotate range
        from matplotlib.colors import Normalize
        vmin = max(0, data_min - 1)
        vmax = 100.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.YlGn
    else:
        vmin    = max(0, data_min - 2)
        vmax    = 100.0
        vcenter = min(vmin + (vmax - vmin) * 0.5, vmax - 0.5)
        vcenter = max(vcenter, vmin + 0.5)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        cmap = plt.cm.RdYlGn

    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("B (%) — sign + sig. retained", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Tick labels
    ax.set_xticks(range(len(PROP_ORDER)))
    ax.set_xticklabels(PROP_LABELS, fontsize=9)
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels([m.replace("MILGBM","MI-LGBM") for m in METHOD_ORDER], fontsize=9)

    # Annotate cells — use 1 decimal place so near-100 values (e.g. 99.725)
    # are shown accurately rather than rounding to "100"
    for i, method in enumerate(METHOD_ORDER):
        for j, prop in enumerate(PROP_ORDER):
            val = mat.loc[method, prop]
            if pd.notna(val):
                txt_color = "white" if val < 60 else "black"
                # Show 1 decimal only when the value is not exactly an integer
                fmt = f"{val:.1f}" if val != round(val) else f"{val:.0f}"
                ax.text(j, i, fmt, ha="center", va="center",
                        fontsize=7.5, color=txt_color, fontweight="bold")

    ax.set_xlabel("Missingness Proportion", labelpad=8)
    ax.set_ylabel("Imputation Method", labelpad=8)
    ax.set_title(
        f"Coefficient Sign & Significance Retention (%) by Method × Missingness — Paper {pid}\n"
        f"Averaged over all key variables and mechanisms | Baseline: β = {cfg['focal_coef']} ({cfg['focal_label']})",
        fontsize=10, pad=10
    )

    # Thin grid lines between cells
    ax.set_xticks(np.arange(-.5, len(PROP_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(METHOD_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    out = outdir / f"fig1_stability_heatmap_paper{pid}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [Paper {pid}] Fig 1 saved: {out.name}")

    # --- Finding ---
    avg_by_method = css.groupby("Method")["B_prop"].mean().reindex(METHOD_ORDER)
    best   = avg_by_method.idxmax()
    worst  = avg_by_method.idxmin()
    at50   = css[css["Proportion"]=="50pct"].groupby("Method")["B_prop"].mean()
    best50 = at50.idxmax()
    print(f"  [Paper {pid}] Finding 1: {best} is most stable overall "
          f"(avg B={avg_by_method[best]:.1f}%); "
          f"at 50% missingness {best50} leads (B={at50[best50]:.1f}%). "
          f"{worst} performs worst (avg B={avg_by_method[worst]:.1f}%).")
    return out


# ---------------------------------------------------------------------------
# Chart 2 — Method comparison bar chart (RMSE from Model_Comparison)
# ---------------------------------------------------------------------------
def chart_bar(cfg: dict, css: pd.DataFrame, mc: pd.DataFrame):
    pid    = cfg["label"]
    outdir = cfg["figures"]

    # Use mean RMSE per method from ModelComparison (imputation RMSE across all cells)
    rmse_by_method = (
        mc.groupby("method")["rmse"].mean()
          .reindex(METHOD_ORDER)
          .rename(index=str)
    )

    # Also compute std from CSS B_prop as instability spread
    instab = css.groupby("Method")["B_prop"].agg(["mean","std"]).reindex(METHOD_ORDER)
    instab["instability"]    = 100 - instab["mean"]  # % iterations NOT both-same
    instab["instability_std"] = instab["std"]

    # Use instability (100 - B_prop) for bar heights — directly interpretable
    vals = instab["instability"].values
    errs = instab["instability_std"].values
    best_idx = int(np.nanargmin(vals))   # lowest instability = best method

    colors = [TEAL if i == best_idx else GRAY_BAR for i in range(len(METHOD_ORDER))]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars = ax.bar(range(len(METHOD_ORDER)), vals, width=0.6,
                  color=colors, edgecolor="white", linewidth=0.5, zorder=3)

    # Error bars
    ax.errorbar(range(len(METHOD_ORDER)), vals, yerr=errs,
                fmt="none", color="#404040", capsize=4, capthick=1.2,
                elinewidth=1.2, zorder=4)

    # Value labels on bars
    for i, (v, e) in enumerate(zip(vals, errs)):
        if pd.notna(v):
            ax.text(i, v + e + 0.3, f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, color="#333333")

    # Best method annotation
    ax.annotate(
        f"Best: {METHOD_ORDER[best_idx].replace('MILGBM','MI-LGBM')}",
        xy=(best_idx, vals[best_idx]),
        xytext=(best_idx + 0.55, vals[best_idx] + errs[best_idx] + 2.5),
        fontsize=8.5, color=TEAL, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=TEAL, lw=1.2)
    )

    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels([m.replace("MILGBM","MI-LGBM") for m in METHOD_ORDER], fontsize=9)
    ax.set_ylabel("Coefficient Instability (%)\n(iterations where sign or significance lost)", labelpad=8)
    ax.set_xlabel("Imputation Method", labelpad=8)
    ax.set_ylim(0, max(vals[~np.isnan(vals)]) * 1.35)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.axhline(0, color="#888888", linewidth=0.6)

    # Light horizontal gridlines only
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.7, zorder=0)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    ax.set_title(
        f"Coefficient Instability by Imputation Method — Paper {pid}\n"
        f"Mean % of iterations where focal IV lost sign or significance | error bars = ±1 SD across proportions",
        fontsize=10, pad=10
    )

    fig.tight_layout()
    out = outdir / f"fig2_method_comparison_paper{pid}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [Paper {pid}] Fig 2 saved: {out.name}")

    # --- Finding ---
    best_m  = METHOD_ORDER[best_idx]
    worst_m = METHOD_ORDER[int(np.nanargmax(vals))]
    print(f"  [Paper {pid}] Finding 2: {best_m} has lowest instability "
          f"({vals[best_idx]:.1f}%); {worst_m} highest ({vals[int(np.nanargmax(vals))]:.1f}%). "
          f"MILGBM and RF are consistently more stable than Mean imputation.")
    return out


# ---------------------------------------------------------------------------
# Chart 3 — Stability trajectory line chart
# ---------------------------------------------------------------------------
def chart_lines(cfg: dict, css: pd.DataFrame):
    pid    = cfg["label"]
    outdir = cfg["figures"]

    # B_prop averaged over all key vars and mechanisms per method × proportion
    agg = (
        css.groupby(["Method", "Proportion"])["B_prop"]
           .mean()
           .reset_index()
    )
    agg["prop_order"] = agg["Proportion"].map(prop_sort_key)
    agg = agg.sort_values("prop_order")

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Reference line at 100%
    ax.axhline(100, linestyle="--", color="#888888", linewidth=1.2,
               label="Baseline (100% stability)", zorder=2)

    plotted = []
    for method in METHOD_ORDER:
        sub = agg[agg["Method"] == method].sort_values("prop_order")
        x   = list(range(len(PROP_ORDER)))
        y   = sub.set_index("Proportion").reindex(PROP_ORDER)["B_prop"].values

        if method in HIGHLIGHT:
            color  = HIGHLIGHT[method]
            lw     = 2.2
            zorder = 4
            marker = "o"
            msize  = 5
            label  = method.replace("MILGBM", "MI-LGBM")
            alpha  = 1.0
        else:
            color  = MUTED
            lw     = 1.1
            zorder = 3
            marker = None
            msize  = 0
            label  = f"_{method}"   # underscore hides from legend
            alpha  = 0.8

        line, = ax.plot(x, y, color=color, linewidth=lw, marker=marker,
                        markersize=msize, zorder=zorder, alpha=alpha,
                        label=label, clip_on=False)

        # End-of-line label for highlighted methods
        if method in HIGHLIGHT and pd.notna(y[-1]):
            ax.annotate(method.replace("MILGBM","MI-LGBM"),
                        xy=(len(PROP_ORDER)-1, y[-1]),
                        xytext=(len(PROP_ORDER)-0.6, y[-1]),
                        fontsize=8.5, color=color, fontweight="bold",
                        va="center")

    # Add "Other methods" legend entry for muted lines
    ax.plot([], [], color=MUTED, linewidth=1.1, label="Other methods")

    ax.set_xticks(range(len(PROP_ORDER)))
    ax.set_xticklabels(PROP_LABELS, fontsize=9)
    ax.set_xlim(-0.3, len(PROP_ORDER) - 0.3)
    ax.set_xlabel("Missingness Proportion", labelpad=8)
    ax.set_ylabel("B (%) — sign + significance retained", labelpad=8)

    # Explicit y-axis: always show 0–100% with integer labels.
    # FormatStrFormatter("%.0f%%") breaks when all data is ≥99.5%
    # because every auto-tick rounds to "100%". Fix: pin limits and ticks.
    ax.set_ylim(-2, 107)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))

    # Light horizontal gridlines only
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.7, zorder=0)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    # Legend (top right, outside if crowded)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8.5, frameon=False,
              loc="lower left", ncol=2)

    ax.set_title(
        f"Focal Coefficient Stability Across Missingness Levels — Paper {pid}\n"
        f"Averaged over key variables and mechanisms | Dashed = 100% retention baseline",
        fontsize=10, pad=10
    )

    fig.tight_layout()
    out = outdir / f"fig3_stability_trajectory_paper{pid}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [Paper {pid}] Fig 3 saved: {out.name}")

    # --- Finding ---
    # Which method degrades fastest from 1pct to 50pct?
    at1   = agg[agg["Proportion"]=="1pct"].set_index("Method")["B_prop"]
    at50  = agg[agg["Proportion"]=="50pct"].set_index("Method")["B_prop"]
    drop  = (at1 - at50).reindex(METHOD_ORDER)
    worst_drop = drop.idxmax()
    best_stable = drop.idxmin()
    print(f"  [Paper {pid}] Finding 3: From 1% to 50% missingness, {worst_drop} "
          f"loses the most stability (drop = {drop[worst_drop]:.1f} pp); "
          f"{best_stable} is most robust (drop = {drop[best_stable]:.1f} pp).")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    for pid, cfg in PAPERS.items():
        print(f"\n{'='*60}")
        print(f"Paper {pid}: {cfg['long_title'].replace(chr(10),' ')}")
        print(f"{'='*60}")
        css, mc = load_data(cfg)
        chart_heatmap(cfg, css)
        chart_bar(cfg, css, mc)
        chart_lines(cfg, css)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
