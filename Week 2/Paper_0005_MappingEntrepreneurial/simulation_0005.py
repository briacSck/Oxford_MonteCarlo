"""
Paper 0005 — Mapping Entrepreneurial Inclusion Across US Neighborhoods
Simulation script: baseline replication + Monte Carlo missing-data analysis
Governing manual: RA_MISSING_DATA.pdf

USAGE:
  Phase 1 (baseline only):   python simulation_0005.py --mode baseline
  Phase 2 (smoke test):      python simulation_0005.py --mode smoke
  Phase 3 (full run):        python simulation_0005.py --mode full

Author: RA replication script
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
SOURCE_DTA = (
    SCRIPT_DIR.parent
    / "paper_mappingentrepreneurialinclusion"
    / "mapping_entrepreneurial_inclusion.dta"
)
DATA_CSV = SCRIPT_DIR / "DATA.csv"
LOG_FILE = SCRIPT_DIR / "simulation_0005.log"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ==============================================================================
# CONFIG
# ==============================================================================
class Config:
    # ── Simulation parameters (manual defaults) ────────────────────────────────
    MISSINGNESS_LEVELS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    NUM_ITERATIONS_PER_SCENARIO = 30
    N_IMPUTATIONS = 5          # MI method: number of completed datasets
    MICE_ITERATIONS = 5
    MAR_NMAR_STRENGTH = 1.5
    ALPHA = 0.05               # significance threshold

    # ── Smoke-test overrides (set by --mode smoke) ─────────────────────────────
    SMOKE_MISSINGNESS_LEVELS = [0.01, 0.10]
    SMOKE_ITERATIONS = 2

    # ── Regression specification ───────────────────────────────────────────────
    DEPENDENT_VAR = "log_shopify_count_1"
    FOCAL_IV = "log_pop_black_aa"
    CONTROLS = [
        "log_pop_total",
        "log_total_bachelor_deg",
        "log_pop_total_poverty",
        "log_total_social_cap",
    ]
    FE_VARS = ["state_name_fe", "MSA_fe"]   # absorbed as entity effects
    CLUSTER_VAR = "MSA_fe"
    WEIGHTS = None

    # ── Key variables (proposed — confirm after baseline lock) ─────────────────
    KEY_VARIABLES = [
        "log_pop_black_aa",
        "log_total_bachelor_deg",
        "log_pop_total_poverty",
        "log_total_social_cap",
    ]

    # ── MAR control (proposed — confirm completeness after data inspection) ─────
    MAR_CONTROL = "log_pop_total"

    # ── Imputation predictor pool ──────────────────────────────────────────────
    PREDICTOR_POOL = [
        "log_pop_black_aa",
        "log_pop_total",
        "log_total_bachelor_deg",
        "log_pop_total_poverty",
        "log_total_social_cap",
    ]

    # ── Output ─────────────────────────────────────────────────────────────────
    OUTPUT_DIR = SCRIPT_DIR
    REPORT_WORKBOOK = SCRIPT_DIR / "Stroube2024Report_0005.xlsx"  # adjust year/author after PDF check
    REGRESSION_TXT_DIR = SCRIPT_DIR / "regressiontxtoutputs"


# ==============================================================================
# STEP 1 — DATA LOADING AND INSPECTION
# ==============================================================================
def load_and_inspect_data(force_reload: bool = False) -> pd.DataFrame:
    """Load .dta, print full inspection report, export DATA.csv."""
    if DATA_CSV.exists() and not force_reload:
        log.info(f"DATA.csv already exists — loading from {DATA_CSV}")
        df = pd.read_csv(DATA_CSV)
        _print_inspection_report(df, source="DATA.csv")
        return df

    log.info(f"Loading source .dta from {SOURCE_DTA}")
    try:
        import pyreadstat
        df, meta = pyreadstat.read_dta(str(SOURCE_DTA))
        log.info("Loaded with pyreadstat")
    except ImportError:
        log.warning("pyreadstat not installed — falling back to pandas.read_stata")
        df = pd.read_stata(str(SOURCE_DTA))

    _print_inspection_report(df, source=str(SOURCE_DTA))

    # Export to DATA.csv
    df.to_csv(DATA_CSV, index=False)
    log.info(f"DATA.csv written to {DATA_CSV} ({len(df):,} rows × {len(df.columns)} cols)")
    return df


def _print_inspection_report(df: pd.DataFrame, source: str) -> None:
    """Print a structured data inspection report."""
    print("\n" + "=" * 72)
    print("DATA INSPECTION REPORT")
    print(f"Source: {source}")
    print("=" * 72)
    print(f"Shape:  {df.shape[0]:,} rows × {df.shape[1]} columns")
    print()

    print("-- Column names and dtypes --")
    for col in df.columns:
        print(f"  {col:<45} {str(df[col].dtype):<12}")
    print()

    print("-- Missing value rates (non-zero only) --")
    miss = df.isnull().mean()
    miss_nonzero = miss[miss > 0].sort_values(ascending=False)
    if miss_nonzero.empty:
        print("  No missing values detected.")
    else:
        for col, rate in miss_nonzero.items():
            print(f"  {col:<45} {rate:.4f} ({int(rate * len(df)):,} missing)")
    print()

    # Key variable diagnostics
    key_vars = [Config.DEPENDENT_VAR] + [Config.FOCAL_IV] + Config.CONTROLS + [Config.MAR_CONTROL]
    present = [v for v in key_vars if v in df.columns]
    absent = [v for v in key_vars if v not in df.columns]

    print("-- Key variable presence check --")
    for v in present:
        stats = df[v].describe()
        print(f"  OK {v:<43} "
              f"mean={stats['mean']:>10.4f}  "
              f"std={stats['std']:>10.4f}  "
              f"min={stats['min']:>10.4f}  "
              f"max={stats['max']:>10.4f}  "
              f"missing={df[v].isnull().sum()}")
    if absent:
        print(f"\n  ABSENT columns (CHECK IMMEDIATELY): {absent}")
    print()

    # FE variable check
    print("-- Fixed effects variable check --")
    for fe in Config.FE_VARS:
        if fe in df.columns:
            n_levels = df[fe].nunique()
            dtype = df[fe].dtype
            print(f"  OK {fe:<43} {n_levels:>6} unique levels  dtype={dtype}")
        else:
            print(f"  MISSING: {fe} NOT FOUND")
    print("=" * 72 + "\n")


# ==============================================================================
# STEP 2 — BASELINE REGRESSION
# ==============================================================================
def run_baseline_regression(df: pd.DataFrame) -> object:
    """
    Run the preferred main regression matching mapping_entrepreneurial_inclusion.do
    Model 3_1:
      reg log_shopify_count_1 log_pop_black_aa log_pop_total log_total_bachelor_deg
          log_pop_total_poverty log_total_social_cap i.state_name_fe i.MSA_fe,
          vce(cluster MSA_fe)

    Uses pyfixest.feols for two-way FE absorption with MSA-clustered SEs.
    This matches Stata's treatment of collinear FE dummies (state + MSA are
    partially nested) and produces coefficients that match the published table.

    NOTE: Dummy-expansion via statsmodels was tried and failed — it produced
    incorrect coefficients (38% off on focal IV) due to collinearity between
    state and MSA FEs. pyfixest uses iterative demeaning (FWL) and correctly
    handles this. Do not revert to dummy expansion.
    """
    import pyfixest as pf

    log.info("Running baseline regression (Model 3_1) with pyfixest...")

    y_var = Config.DEPENDENT_VAR
    x_vars = [Config.FOCAL_IV] + Config.CONTROLS
    fe_vars = Config.FE_VARS
    cluster_var = Config.CLUSTER_VAR

    reg_vars = [y_var] + x_vars + fe_vars
    df_reg = df[reg_vars].dropna()
    n_dropped = len(df) - len(df_reg)
    if n_dropped > 0:
        log.warning(f"Dropped {n_dropped:,} rows with missing values in regression variables")
    log.info(f"Regression sample: N = {len(df_reg):,}")

    fe_formula = " + ".join(fe_vars)
    x_formula = " + ".join(x_vars)
    formula = f"{y_var} ~ {x_formula} | {fe_formula}"

    fit = pf.feols(formula, data=df_reg, vcov={f"CRV1": cluster_var})
    log.info("pyfixest regression complete")

    _print_baseline_table(fit, n_obs=len(df_reg))
    return fit


def _run_simulation_regression(df: pd.DataFrame) -> object:
    """
    Re-run the preferred regression on an (imputed) dataset.
    Used inside the simulation loop — same spec as baseline.
    Returns the fitted pyfixest object.
    """
    import pyfixest as pf

    y_var = Config.DEPENDENT_VAR
    x_vars = [Config.FOCAL_IV] + Config.CONTROLS
    fe_vars = Config.FE_VARS
    cluster_var = Config.CLUSTER_VAR

    fe_formula = " + ".join(fe_vars)
    x_formula = " + ".join(x_vars)
    formula = f"{y_var} ~ {x_formula} | {fe_formula}"

    df_clean = df[[y_var] + x_vars + fe_vars].dropna()
    fit = pf.feols(formula, data=df_clean, vcov={f"CRV1": cluster_var})
    return fit


def _print_baseline_table(fit, n_obs: int) -> None:
    """Print a clean coefficient table (pyfixest result) vs published table."""
    sep = "=" * 72
    dash = "-" * 72
    print(sep)
    print("BASELINE REGRESSION -- PAPER TABLE 3, MODEL 3_1 (pyfixest)")
    print(f"N = {n_obs:,}")
    print(sep)
    print(f"{"Variable":<45} {"Coef":>10} {"SE":>10} {"p-value":>10} {"Sig":>4}")
    print(dash)

    tidy = fit.tidy()
    focal = Config.FOCAL_IV
    all_vars = [focal] + Config.CONTROLS

    for var in all_vars:
        if var in tidy.index:
            row = tidy.loc[var]
            coef = float(row["Estimate"])
            se = float(row["Std. Error"])
            pv = float(row["Pr(>|t|)"])
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            print(f"  {var:<43} {coef:>10.4f} {se:>10.4f} {pv:>10.4f} {sig:>4}")

    print(dash)
    try:
        summary_str = str(fit.summary())
        for line in summary_str.splitlines():
            if "R2" in line or "RMSE" in line:
                print(f"  {line.strip()}")
    except Exception:
        pass

    print(sep)
    print()
    print("COMPARE TO PUBLISHED TABLE 3, MODEL 3_1 AND FILL confignotes.txt")
    print("If focal IV coef within 4th decimal -> baseline MATCHED")
    print("If deviation > 5 pct on focal IV -> stop and debug")
    print()

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Paper 0005 simulation script")
    parser.add_argument(
        "--mode",
        choices=["baseline", "smoke", "full"],
        default="baseline",
        help="baseline: data inspection + baseline regression only; "
             "smoke: reduced simulation test; full: complete simulation",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Force reload from .dta even if DATA.csv already exists",
    )
    args = parser.parse_args()

    log.info(f"=" * 60)
    log.info(f"Paper 0005 — Mode: {args.mode.upper()}")
    log.info(f"=" * 60)

    # ── Phase 1: Data loading and baseline ────────────────────────────────────
    df = load_and_inspect_data(force_reload=args.reload)
    result = run_baseline_regression(df)

    if args.mode == "baseline":
        log.info("Baseline mode complete. Review output above and update confignotes.txt.")
        log.info("DO NOT proceed to simulation until baseline is confirmed matched.")
        return

    # ── Phase 2+: Simulation (smoke or full) ──────────────────────────────────
    log.info("Simulation not yet implemented. Run with --mode baseline first.")
    log.info("Once baseline is confirmed, simulation infrastructure will be added here.")


if __name__ == "__main__":
    main()
