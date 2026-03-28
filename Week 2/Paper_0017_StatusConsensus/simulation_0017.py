"""
Paper 0017 -- Status and Consensus: Heterogeneity in Audience Evaluations
             of Female- versus Male-Lead Films
             Stroube (2024), Strategic Management Journal, 45:994-1024

Simulation script: baseline replication + Monte Carlo missing-data analysis
Governing manual: RA_MISSING_DATA.pdf

PREFERRED MODEL: Table 2, Model 2 (m.pooled.sd)
  DV: sd_pooled
  lm(sd_pooled ~ FLead + mean_pooled + kim_violence_gore + kim_sex_nudity
                 + kim_language + major + log(bom_opening_theaters)
                 + genres.count + as.factor(bom_year) + as.factor(bom_open_month)
                 + [22 genre dummies], data = ratings)

USAGE:
  Phase 1 (baseline only):   python simulation_0017.py --mode baseline
  Phase 2 (smoke test):      python simulation_0017.py --mode smoke
  Phase 3 (full run):        python simulation_0017.py --mode full
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- Paths -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
SOURCE_CSV = (
    SCRIPT_DIR.parent
    / "paper_statusandconsensus"
    / "movie_data.csv"
)
DATA_CSV = SCRIPT_DIR / "DATA.csv"
LOG_FILE = SCRIPT_DIR / "simulation_0017.log"

# -- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================
class Config:
    # -- Simulation parameters (manual defaults) -----------------------------
    MISSINGNESS_LEVELS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    NUM_ITERATIONS_PER_SCENARIO = 30
    N_IMPUTATIONS = 5
    MICE_ITERATIONS = 5
    MAR_NMAR_STRENGTH = 1.5
    ALPHA = 0.05

    # -- Smoke-test overrides ------------------------------------------------
    SMOKE_MISSINGNESS_LEVELS = [0.01, 0.10]
    SMOKE_ITERATIONS = 2

    # -- Regression specification (matches R m.pooled.sd exactly) ------------
    DEPENDENT_VAR = "sd_pooled"
    FOCAL_IV = "FLead"          # binary; cannot be a key variable
    # mean_pooled is a required control (adjusts for rating level) -- not a key var
    CONTROLS = [
        "mean_pooled",
        "kim_violence_gore",
        "kim_sex_nudity",
        "kim_language",
        "major",
        "log_bom_opening_theaters",   # derived: log(bom_opening_theaters)
        "genres_count",               # renamed from genres.count (dot->underscore)
    ]
    # These are included in the regression but handled as factor dummies
    FE_YEAR_VAR = "bom_year"
    FE_MONTH_VAR = "bom_open_month"
    # 22 genre dummies (already binary in CSV -- included explicitly)
    GENRE_DUMMIES = [
        "genre_Action", "genre_Adventure", "genre_Animation", "genre_Biography",
        "genre_Comedy", "genre_Crime", "genre_Drama", "genre_Family",
        "genre_Fantasy", "genre_History", "genre_Horror", "genre_Music",
        "genre_Musical", "genre_Mystery", "genre_News", "genre_Romance",
        "genre_Sci_Fi", "genre_Short", "genre_Sport", "genre_Thriller",
        "genre_War", "genre_Western",
    ]
    WEIGHTS = None
    CLUSTER_VAR = None   # standard OLS SEs (no clustering in R code)

    # -- Key variables (LOCKED 2026-03-28 after baseline match) --------------
    KEY_VARIABLES = [
        "kim_violence_gore",
        "kim_sex_nudity",
        "kim_language",
        "log_bom_opening_theaters",
    ]

    # -- MAR control (LOCKED 2026-03-28) ------------------------------------
    MAR_CONTROL = "genres_count"

    # -- Imputation predictor pool ------------------------------------------
    PREDICTOR_POOL = [
        "FLead",
        "mean_pooled",
        "kim_violence_gore",
        "kim_sex_nudity",
        "kim_language",
        "major",
        "log_bom_opening_theaters",
        "genres_count",
    ]

    # -- Output --------------------------------------------------------------
    OUTPUT_DIR = SCRIPT_DIR
    REPORT_WORKBOOK = SCRIPT_DIR / "Stroube2024Report_0017.xlsx"
    REGRESSION_TXT_DIR = SCRIPT_DIR / "regressiontxtoutputs"


# ============================================================================
# STEP 1 -- DATA LOADING AND PREPROCESSING
# ============================================================================

# Column rename map: dots to underscores (patsy/Python doesn't handle dots well)
_RENAME = {
    "genres.count": "genres_count",
    "genre.Action": "genre_Action",
    "genre.Adventure": "genre_Adventure",
    "genre.Animation": "genre_Animation",
    "genre.Biography": "genre_Biography",
    "genre.Comedy": "genre_Comedy",
    "genre.Crime": "genre_Crime",
    "genre.Drama": "genre_Drama",
    "genre.Family": "genre_Family",
    "genre.Fantasy": "genre_Fantasy",
    "genre.History": "genre_History",
    "genre.Horror": "genre_Horror",
    "genre.Music": "genre_Music",
    "genre.Musical": "genre_Musical",
    "genre.Mystery": "genre_Mystery",
    "genre.News": "genre_News",
    "genre.Romance": "genre_Romance",
    "genre.Sci.Fi": "genre_Sci_Fi",
    "genre.Short": "genre_Short",
    "genre.Sport": "genre_Sport",
    "genre.Thriller": "genre_Thriller",
    "genre.War": "genre_War",
    "genre.Western": "genre_Western",
}


def load_and_inspect_data(force_reload: bool = False) -> pd.DataFrame:
    """Load movie_data.csv, preprocess, print inspection report, export DATA.csv."""
    if DATA_CSV.exists() and not force_reload:
        log.info(f"DATA.csv already exists -- loading from {DATA_CSV}")
        df = pd.read_csv(DATA_CSV)
        _print_inspection_report(df, source="DATA.csv")
        return df

    log.info(f"Loading source CSV from {SOURCE_CSV}")
    df = pd.read_csv(str(SOURCE_CSV))
    log.info(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # -- Rename dot-columns to underscore ------------------------------------
    df = df.rename(columns=_RENAME)

    # -- Derive log_bom_opening_theaters ------------------------------------
    # No zeros in bom_opening_theaters (confirmed: min=1), so log() matches R
    df["log_bom_opening_theaters"] = np.log(df["bom_opening_theaters"])

    # -- Encode bool columns as 0/1 integers --------------------------------
    for col in ["FLead", "major"]:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    _print_inspection_report(df, source=str(SOURCE_CSV))

    df.to_csv(DATA_CSV, index=False)
    log.info(f"DATA.csv written to {DATA_CSV} ({len(df):,} rows x {len(df.columns)} cols)")
    return df


def _print_inspection_report(df: pd.DataFrame, source: str) -> None:
    print("\n" + "=" * 72)
    print("DATA INSPECTION REPORT")
    print(f"Source: {source}")
    print("=" * 72)
    print(f"Shape:  {df.shape[0]:,} rows x {df.shape[1]} columns")
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

    key_vars = (
        [Config.DEPENDENT_VAR, Config.FOCAL_IV]
        + Config.CONTROLS
        + [Config.MAR_CONTROL]
    )
    present = [v for v in key_vars if v in df.columns]
    absent = [v for v in key_vars if v not in df.columns]

    print("-- Key variable presence check --")
    for v in present:
        stats = df[v].describe()
        print(
            f"  OK {v:<43} "
            f"mean={stats['mean']:>10.4f}  "
            f"std={stats['std']:>10.4f}  "
            f"min={stats['min']:>10.4f}  "
            f"max={stats['max']:>10.4f}  "
            f"missing={df[v].isnull().sum()}"
        )
    if absent:
        print(f"\n  ABSENT columns (CHECK IMMEDIATELY): {absent}")

    print()
    print("-- FE variable check --")
    for fe in [Config.FE_YEAR_VAR, Config.FE_MONTH_VAR]:
        if fe in df.columns:
            n_levels = df[fe].nunique()
            print(f"  OK {fe:<43} {n_levels:>6} unique levels  dtype={df[fe].dtype}")
        else:
            print(f"  MISSING: {fe} NOT FOUND")

    print()
    print("-- Genre dummy check --")
    for g in Config.GENRE_DUMMIES:
        if g in df.columns:
            print(f"  OK {g}")
        else:
            print(f"  MISSING: {g} NOT FOUND")

    print("=" * 72 + "\n")


# ============================================================================
# STEP 2 -- BASELINE REGRESSION
# ============================================================================

def _build_design_matrix(df: pd.DataFrame):
    """
    Build y vector and X matrix exactly matching R's m.pooled.sd:
      sd_pooled ~ FLead + mean_pooled + kim_violence_gore + kim_sex_nudity
                + kim_language + major + log(bom_opening_theaters)
                + genres.count + as.factor(bom_year) + as.factor(bom_open_month)
                + [22 genre dummies]

    - Year and month FEs: treatment coding (drop first level), matching R's
      as.factor() default.
    - Genre dummies: included as-is (already 0/1 in CSV).
    - Intercept: added via sm.add_constant (matches R's implicit intercept).
    """
    import statsmodels.api as sm

    y = df[Config.DEPENDENT_VAR].astype(float)

    # Continuous and binary predictors
    cont_cols = [Config.FOCAL_IV] + Config.CONTROLS
    X_cont = df[cont_cols].astype(float)

    # Year FEs (treatment coding: drop lowest year)
    year_dummies = pd.get_dummies(
        df[Config.FE_YEAR_VAR], prefix="yr", drop_first=True, dtype=float
    )
    # Month FEs (treatment coding: drop lowest month)
    month_dummies = pd.get_dummies(
        df[Config.FE_MONTH_VAR], prefix="mo", drop_first=True, dtype=float
    )

    # Genre dummies (already binary -- include all 22 as-is, no drop_first needed
    # since they are not mutually exclusive -- a film can have multiple genres)
    genre_cols = [g for g in Config.GENRE_DUMMIES if g in df.columns]
    X_genre = df[genre_cols].astype(float)

    X = pd.concat([X_cont, year_dummies, month_dummies, X_genre], axis=1)
    X = sm.add_constant(X, has_constant="raise")

    return y, X


def run_baseline_regression(df: pd.DataFrame) -> object:
    """
    Run the preferred main regression matching movies.R m.pooled.sd.
    Uses statsmodels OLS (no FE absorption needed -- year/month are dummy-expanded,
    genre dummies are pre-built binary columns, no partial-nesting collinearity).
    Standard OLS SEs (no clustering in R code).
    """
    import statsmodels.api as sm

    log.info("Running baseline regression (m.pooled.sd) with statsmodels OLS...")

    reg_vars = (
        [Config.DEPENDENT_VAR, Config.FOCAL_IV]
        + Config.CONTROLS
        + Config.GENRE_DUMMIES
        + [Config.FE_YEAR_VAR, Config.FE_MONTH_VAR]
    )
    reg_vars_present = [v for v in reg_vars if v in df.columns]
    df_reg = df[reg_vars_present].dropna()
    n_dropped = len(df) - len(df_reg)
    if n_dropped > 0:
        log.warning(f"Dropped {n_dropped:,} rows with missing values in regression variables")
    log.info(f"Regression sample: N = {len(df_reg):,}")

    y, X = _build_design_matrix(df_reg)
    fit = sm.OLS(y, X).fit()
    log.info("statsmodels OLS complete")

    _print_baseline_table(fit, n_obs=len(df_reg))
    return fit


def _run_simulation_regression(df: pd.DataFrame) -> object:
    """
    Re-run the preferred regression on an (imputed) dataset.
    Used inside the simulation loop.
    """
    import statsmodels.api as sm

    reg_vars = (
        [Config.DEPENDENT_VAR, Config.FOCAL_IV]
        + Config.CONTROLS
        + Config.GENRE_DUMMIES
        + [Config.FE_YEAR_VAR, Config.FE_MONTH_VAR]
    )
    reg_vars_present = [v for v in reg_vars if v in df.columns]
    df_clean = df[reg_vars_present].dropna()
    y, X = _build_design_matrix(df_clean)
    fit = sm.OLS(y, X).fit()
    return fit


def _print_baseline_table(fit, n_obs: int) -> None:
    sep = "=" * 72
    dash = "-" * 72
    print(sep)
    print("BASELINE REGRESSION -- PAPER TABLE 2, MODEL m.pooled.sd (statsmodels)")
    print(f"N = {n_obs:,}")
    print(sep)
    print(f"{'Variable':<45} {'Coef':>10} {'SE':>10} {'p-value':>10} {'Sig':>4}")
    print(dash)

    focal = Config.FOCAL_IV
    display_vars = [focal] + Config.CONTROLS
    for var in display_vars:
        if var in fit.params.index:
            coef = float(fit.params[var])
            se = float(fit.bse[var])
            pv = float(fit.pvalues[var])
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            print(f"  {var:<43} {coef:>10.4f} {se:>10.4f} {pv:>10.4f} {sig:>4}")

    print(dash)
    try:
        r2 = fit.rsquared
        r2_adj = fit.rsquared_adj
        print(f"  R2        = {r2:.4f}")
        print(f"  R2 adj    = {r2_adj:.4f}")
    except Exception:
        pass

    print(sep)
    print()
    print("COMPARE TO PUBLISHED TABLE 2, MODEL m.pooled.sd AND FILL confignotes.txt")
    print("FLead coef ~ 0.047 (published) -- within 4th decimal -> baseline MATCHED")
    print("Deviation > 5 pct on FLead -> stop and debug")
    print()


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Paper 0017 simulation script")
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
        help="Force reload from source CSV even if DATA.csv already exists",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"Paper 0017 -- Mode: {args.mode.upper()}")
    log.info("=" * 60)

    # -- Phase 1: Data loading and baseline ----------------------------------
    df = load_and_inspect_data(force_reload=args.reload)
    result = run_baseline_regression(df)

    if args.mode == "baseline":
        log.info("Baseline mode complete. Review output above and update confignotes.txt.")
        log.info("DO NOT proceed to simulation until baseline is confirmed matched.")
        return

    # -- Phase 2+: Simulation ------------------------------------------------
    log.info("Simulation not yet implemented. Run with --mode baseline first.")
    log.info("Once baseline is confirmed, simulation infrastructure will be added here.")


if __name__ == "__main__":
    main()
