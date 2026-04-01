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
import hashlib
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import t as t_dist
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -- Paths -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PAPER_DIR = SCRIPT_DIR.parent          # paper root (one level above scripts/)
SOURCE_CSV = (
    PAPER_DIR.parent.parent
    / "source_artifacts"
    / "movie_data.csv"
)
DATA_CSV = PAPER_DIR / "DATA.csv"
LOG_FILE = PAPER_DIR / "logs" / "simulation_0017.log"

# -- Logging -----------------------------------------------------------------
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
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
    RANDOM_SEED = 42
    ADD_RESIDUAL_NOISE = True
    DL_EPOCHS = 30
    DL_PATIENCE = 5
    MICE_LGBM_N_ESTIMATORS = 30
    MICE_LGBM_MAX_DEPTH = 4
    MICE_LGBM_LEARNING_RATE = 0.05
    MICE_LGBM_NUM_LEAVES = 10

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

    MECHANISMS = ["MCAR", "MAR", "NMAR"]
    METHODS = ["LD", "Mean", "Reg", "Iter", "RF", "DL", "MILGBM"]

    # -- Output --------------------------------------------------------------
    OUTPUT_DIR = PAPER_DIR
    REPORT_WORKBOOK = PAPER_DIR / "full_run" / "Stroube2024Report_0017.xlsx"
    SMOKE_WORKBOOK = PAPER_DIR / "smoke" / "SMOKE_Stroube2024Report_0017.xlsx"
    REGRESSION_TXT_DIR = PAPER_DIR / "regression_outputs"
    PROGRESS_LOG = PAPER_DIR / "logs" / "simulation_0017_progress.log"


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
        if col in df.columns and df[col].dtype == bool:
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

    print("-- Missing value rates (non-zero only) --")
    miss = df.isnull().mean()
    miss_nonzero = miss[miss > 0].sort_values(ascending=False)
    if miss_nonzero.empty:
        print("  No missing values detected.")
    else:
        for col, rate in miss_nonzero.items():
            print(f"  {col:<45} {rate:.4f} ({int(rate * len(df)):,} missing)")

    key_vars = [Config.DEPENDENT_VAR, Config.FOCAL_IV] + Config.CONTROLS
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

    print("-- FE variable check --")
    for fe in [Config.FE_YEAR_VAR, Config.FE_MONTH_VAR]:
        if fe in df.columns:
            n_levels = df[fe].nunique()
            print(f"  OK {fe:<43} {n_levels:>6} unique levels  dtype={df[fe].dtype}")
        else:
            print(f"  MISSING: {fe} NOT FOUND")

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

    # Genre dummies (already binary -- include all 22 as-is)
    genre_cols = [g for g in Config.GENRE_DUMMIES if g in df.columns]
    X_genre = df[genre_cols].astype(float)

    X = pd.concat([X_cont, year_dummies, month_dummies, X_genre], axis=1)
    X = sm.add_constant(X, has_constant="raise")

    return y, X


def run_baseline_regression(df: pd.DataFrame) -> object:
    """
    Run the preferred main regression matching movies.R m.pooled.sd.
    statsmodels OLS, standard SEs (no clustering in R code).
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
    if len(df_clean) < 50:
        raise ValueError(f"Too few observations after dropna: {len(df_clean)}")
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
# STEP 3 -- COEFFICIENT EXTRACTION
# ============================================================================

def _extract_focal_coef(fit) -> dict:
    """Extract focal IV coefficient info from statsmodels result or pooled MI object."""
    if fit is None:
        return None
    try:
        if getattr(fit, "_is_pooled_mi", False):
            coef = fit._pooled_coef
            se = fit._pooled_se
            pval = fit._pooled_pval
            nobs = fit._pooled_n
        else:
            var = Config.FOCAL_IV
            coef = float(fit.params[var])
            se = float(fit.bse[var])
            pval = float(fit.pvalues[var])
            nobs = int(fit.nobs)
        return {
            "coef": float(coef) if pd.notna(coef) else np.nan,
            "se": float(se) if pd.notna(se) else np.nan,
            "pval": float(pval) if pd.notna(pval) else np.nan,
            "nobs": int(nobs),
            "sign": int(np.sign(coef)) if pd.notna(coef) else 0,
            "sig": bool(float(pval) < Config.ALPHA) if pd.notna(pval) else False,
        }
    except Exception as e:
        log.debug(f"_extract_focal_coef failed: {e}")
        return None


def _save_iter_txt(fit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if getattr(fit, "_is_pooled_mi", False):
            content = (
                f"MI Pooled Result -- Paper 0017\n"
                f"Focal IV: {Config.FOCAL_IV}\n"
                f"Pooled Coef: {fit._pooled_coef:.6f}\n"
                f"Pooled SE:   {fit._pooled_se:.6f}\n"
                f"Pooled pval: {fit._pooled_pval:.6f}\n"
                f"N (mean):    {fit._pooled_n}\n"
                f"FMI:         {getattr(fit, '_fmi', float('nan')):.4f}\n"
                f"RE:          {getattr(fit, '_re', float('nan')):.4f}\n"
            )
        else:
            content = fit.summary().as_text()
    except Exception:
        content = "[Output capture failed]"
    with open(path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(content)


# ============================================================================
# STEP 4 -- MISSINGNESS SIMULATION
# ============================================================================
def simulate_missingness_single_col(
    df: pd.DataFrame,
    col: str,
    miss_prop: float,
    seed: int,
    mechanism: str = "MCAR",
    mar_control_col: str = None,
    strength: float = 1.5,
) -> pd.DataFrame:
    data = df.copy()
    rng = np.random.default_rng(seed)
    if col not in data.columns:
        return data
    data[col] = pd.to_numeric(data[col], errors="coerce")
    eligible = data.index[data[col].notna()].tolist()
    if not eligible:
        return data
    n_eligible = len(eligible)
    n_missing = int(np.floor(miss_prop * n_eligible))
    if n_missing <= 0:
        return data
    n_missing = min(n_missing, n_eligible)

    indices_to_nan = []

    if mechanism.upper() == "MCAR":
        indices_to_nan = rng.choice(eligible, size=n_missing, replace=False)

    elif mechanism.upper() == "MAR":
        if mar_control_col is None or mar_control_col not in data.columns:
            return simulate_missingness_single_col(df, col, miss_prop, seed, "MCAR", strength=strength)
        ctrl = pd.to_numeric(data.loc[eligible, mar_control_col], errors="coerce")
        if ctrl.isna().any():
            ctrl = ctrl.fillna(ctrl.mean())
        if ctrl.nunique() <= 1 or ctrl.max() == ctrl.min():
            return simulate_missingness_single_col(df, col, miss_prop, seed, "MCAR", strength=strength)
        norm_ctrl = (ctrl - ctrl.min()) / (ctrl.max() - ctrl.min())
        weights = np.exp(norm_ctrl.values * strength)
        if not np.all(np.isfinite(weights)) or weights.sum() == 0:
            return simulate_missingness_single_col(df, col, miss_prop, seed, "MCAR", strength=strength)
        probs = weights / weights.sum()
        try:
            indices_to_nan = rng.choice(eligible, size=n_missing, replace=False, p=probs)
        except ValueError:
            return simulate_missingness_single_col(df, col, miss_prop, seed, "MCAR", strength=strength)

    elif mechanism.upper() == "NMAR":
        vals = data.loc[eligible, col]
        mean_y, std_y = vals.mean(), vals.std()
        if pd.isna(std_y) or std_y < 1e-9:
            return simulate_missingness_single_col(df, col, miss_prop, seed, "MCAR", strength=strength)
        std_vals = (vals - mean_y) / std_y
        weights = np.exp(std_vals.values * strength)
        if not np.all(np.isfinite(weights)) or weights.sum() == 0:
            return simulate_missingness_single_col(df, col, miss_prop, seed, "MCAR", strength=strength)
        probs = weights / weights.sum()
        try:
            indices_to_nan = rng.choice(eligible, size=n_missing, replace=False, p=probs)
        except ValueError:
            return simulate_missingness_single_col(df, col, miss_prop, seed, "MCAR", strength=strength)
    else:
        return simulate_missingness_single_col(df, col, miss_prop, seed, "MCAR", strength=strength)

    if len(indices_to_nan) > 0:
        data.loc[indices_to_nan, col] = np.nan
    return data


# ============================================================================
# STEP 5 -- IMPUTATION PIPELINE
# ============================================================================
class ImputationPipeline:
    """Cross-section imputation pipeline (adapted from reference, panel logic removed)."""

    def __init__(
        self,
        df_with_na: pd.DataFrame,
        original_df_complete: pd.DataFrame,
        key_var: str,
        predictor_pool: list,
        iteration_num: int,
        n_imputations: int = 5,
        mice_iters: int = 5,
        random_seed: int = 42,
        add_noise: bool = True,
        lgbm_n_est: int = 30,
        lgbm_max_depth: int = 4,
        lgbm_lr: float = 0.05,
        lgbm_num_leaves: int = 10,
        dl_epochs: int = 30,
        dl_patience: int = 5,
    ):
        self.df = df_with_na.copy()
        self.original_df = original_df_complete.copy()
        self.key_var = key_var
        self.iteration_num = iteration_num
        self.n_imputations = n_imputations
        self.mice_iters = mice_iters
        self.random_seed = random_seed
        self.add_noise = add_noise
        self.lgbm_n_est = lgbm_n_est
        self.lgbm_max_depth = lgbm_max_depth
        self.lgbm_lr = lgbm_lr
        self.lgbm_num_leaves = lgbm_num_leaves
        self.dl_epochs = dl_epochs
        self.dl_patience = dl_patience

        # Numeric predictor columns present in the dataset
        self.numeric_cols = []
        for col in predictor_pool:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                if self.df[col].notna().any():
                    self.numeric_cols.append(col)

    # -- Helpers --

    def _impute_predictors_mean(self, df_in: pd.DataFrame, cols: list) -> pd.DataFrame:
        df_out = df_in.copy()
        for col in cols:
            if col in df_out.columns and df_out[col].isna().any():
                df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
                mv = df_out[col].mean()
                df_out[col] = df_out[col].fillna(mv if pd.notna(mv) else 0)
        return df_out

    def _residual_sd(self, model, X_tr, y_tr) -> float:
        try:
            resids = y_tr.values - model.predict(X_tr)
            nonzero = resids[np.abs(resids) > 1e-9]
            if len(nonzero) < 2:
                return 0.0
            sd = float(np.std(nonzero))
            return sd if np.isfinite(sd) else 0.0
        except Exception:
            return 0.0

    def _prep_train_pred(self, target_col: str, pred_cols: list):
        """Return X_train, y_train, X_pred and fallback_mean."""
        imp_df = self._impute_predictors_mean(self.df.copy(), pred_cols)
        missing_mask = self.df[target_col].isna()
        fallback = self.df[target_col].mean()
        fallback = 0 if pd.isna(fallback) else float(fallback)

        X_tr = imp_df.loc[~missing_mask, pred_cols]
        y_tr = self.df.loc[~missing_mask, target_col]
        X_pr = imp_df.loc[missing_mask, pred_cols]

        common = X_tr.dropna(how="any").index.intersection(y_tr.dropna().index)
        X_tr = X_tr.loc[common]
        y_tr = y_tr.loc[common]
        X_pr = X_pr.dropna(how="any")
        return X_tr, y_tr, X_pr, fallback, missing_mask

    # -- 7 Methods --

    def listwise_deletion(self) -> pd.DataFrame:
        if self.key_var not in self.df.columns:
            return self.df.copy()
        return self.df.dropna(subset=[self.key_var])

    def mean_imputation(self) -> pd.DataFrame:
        out = self.df.copy()
        if self.key_var in out.columns and out[self.key_var].isna().any():
            out[self.key_var] = pd.to_numeric(out[self.key_var], errors="coerce")
            mv = out[self.key_var].mean()
            out[self.key_var] = out[self.key_var].fillna(mv if pd.notna(mv) else 0)
        return out

    def regression_imputation(self) -> pd.DataFrame:
        target = self.key_var
        pred_cols = [c for c in self.numeric_cols if c != target]
        if not pred_cols:
            return self.mean_imputation()
        out = self.df.copy()
        X_tr, y_tr, X_pr, fallback, missing_mask = self._prep_train_pred(target, pred_cols)
        if X_tr.empty or y_tr.empty or X_tr.shape[0] < 5 or X_pr.empty:
            out.loc[missing_mask, target] = fallback
            return out
        try:
            model = LinearRegression(n_jobs=1)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_pr)
            if self.add_noise:
                sd = self._residual_sd(model, X_tr, y_tr)
                if sd > 0:
                    preds = preds + np.random.normal(0, sd, size=len(preds))
            out.loc[X_pr.index, target] = preds
            still_missing = out[target].isna() & missing_mask
            if still_missing.any():
                out.loc[still_missing, target] = fallback
        except Exception as e:
            log.debug(f"Reg impute error: {e}")
            out.loc[missing_mask, target] = fallback
        return out

    def stochastic_iterative_imputation(self) -> pd.DataFrame:
        target = self.key_var
        pred_cols = [c for c in self.numeric_cols if c != target]
        if not pred_cols:
            return self.mean_imputation()
        out = self.df.copy()
        out[target] = pd.to_numeric(out[target], errors="coerce")
        all_cols = [target] + [c for c in pred_cols if c in out.columns]
        all_cols = [c for c in all_cols if out[c].notna().any()]
        if target not in all_cols or len(all_cols) <= 1:
            return self.mean_imputation()
        sub = out[all_cols].copy()
        try:
            imp_seed = self.random_seed + self.iteration_num
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                random_state=imp_seed,
                max_iter=self.mice_iters,
                n_nearest_features=min(10, len(all_cols) - 1),
                sample_posterior=True,
                tol=1e-3,
                verbose=0,
                initial_strategy="mean",
            )
            filled = imputer.fit_transform(sub)
            out[all_cols] = filled
            if out[target].isna().any():
                mv = self.df[target].mean()
                out[target] = out[target].fillna(mv if pd.notna(mv) else 0)
        except Exception as e:
            log.debug(f"Iter impute error: {e}")
            return self.mean_imputation()
        return out

    def ml_imputation(self) -> pd.DataFrame:
        target = self.key_var
        pred_cols = [c for c in self.numeric_cols if c != target
                     and self.df[c].notna().sum() > 0.5 * len(self.df)]
        if not pred_cols:
            return self.mean_imputation()
        out = self.df.copy()
        X_tr, y_tr, X_pr, fallback, missing_mask = self._prep_train_pred(target, pred_cols)
        if X_tr.empty or X_tr.shape[0] < 5 or X_pr.empty:
            out.loc[missing_mask, target] = fallback
            return out
        try:
            model_seed = self.random_seed + self.iteration_num
            model = RandomForestRegressor(
                n_estimators=self.lgbm_n_est,
                max_depth=self.lgbm_max_depth,
                random_state=model_seed,
                n_jobs=1,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_pr)
            if self.add_noise:
                sd = self._residual_sd(model, X_tr, y_tr)
                if sd > 0:
                    preds = preds + np.random.normal(0, sd, size=len(preds))
            out.loc[X_pr.index, target] = preds
            still_missing = out[target].isna() & missing_mask
            if still_missing.any():
                out.loc[still_missing, target] = fallback
        except Exception as e:
            log.debug(f"RF impute error: {e}")
            out.loc[missing_mask, target] = fallback
        return out

    def deep_learning_imputation(self) -> pd.DataFrame:
        target = self.key_var
        pred_cols = [c for c in self.numeric_cols if c != target
                     and self.df[c].notna().sum() > 0.5 * len(self.df)]
        if not pred_cols:
            return self.mean_imputation()
        out = self.df.copy()
        X_tr, y_tr, X_pr, fallback, missing_mask = self._prep_train_pred(target, pred_cols)
        if X_tr.empty or X_tr.shape[0] < 10 or X_pr.empty or y_tr.isna().any():
            out.loc[missing_mask, target] = fallback
            return out
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)

            tf.random.set_seed(self.random_seed + self.iteration_num)
            scaler_x, scaler_y = StandardScaler(), StandardScaler()
            X_tr_s = scaler_x.fit_transform(X_tr)
            y_tr_s = scaler_y.fit_transform(y_tr.values.reshape(-1, 1))

            model = Sequential([
                Dense(32, activation="relu", input_shape=(X_tr_s.shape[1],)),
                Dropout(0.1),
                Dense(16, activation="relu"),
                Dense(1),
            ])
            model.compile(optimizer=Adam(learning_rate=0.005), loss="mse")
            val_split = 0.1 if len(X_tr_s) * 0.1 >= 1 else 0.0
            monitor = "val_loss" if val_split > 0 else "loss"
            cb = [EarlyStopping(monitor=monitor, patience=self.dl_patience, restore_best_weights=True, verbose=0)]
            model.fit(X_tr_s, y_tr_s, epochs=self.dl_epochs,
                      batch_size=min(32, len(X_tr_s)),
                      validation_split=val_split, callbacks=cb, verbose=0)

            if not X_pr.empty:
                X_pr_s = scaler_x.transform(X_pr)
                preds_s = model.predict(X_pr_s, verbose=0)
                preds = scaler_y.inverse_transform(preds_s).flatten()
                if self.add_noise:
                    y_pred_train_s = model.predict(X_tr_s, verbose=0)
                    y_pred_train = scaler_y.inverse_transform(y_pred_train_s).flatten()
                    resids = y_tr.values - y_pred_train
                    nonzero = resids[np.abs(resids) > 1e-9]
                    if len(nonzero) > 1:
                        sd = np.std(nonzero)
                        if np.isfinite(sd) and sd > 0:
                            preds = preds + np.random.normal(0, sd, size=len(preds))
                out.loc[X_pr.index, target] = preds

            still_missing = out[target].isna() & missing_mask
            if still_missing.any():
                out.loc[still_missing, target] = fallback
            del model
            tf.keras.backend.clear_session()
        except Exception as e:
            log.debug(f"DL impute error: {e}")
            out.loc[missing_mask, target] = fallback
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except Exception:
                pass
        return out

    def custom_multiple_imputation(self) -> list:
        """MICE with LightGBM for M=5 completed datasets."""
        import lightgbm as lgb
        target = self.key_var
        original_with_na = self.df.copy()
        pred_cols = [c for c in self.numeric_cols if c != target
                     and original_with_na[c].notna().sum() > 0.5 * len(original_with_na)]
        missing_mask = original_with_na[target].isna()
        fallback = original_with_na[target].mean()
        fallback = 0 if pd.isna(fallback) else float(fallback)

        if not missing_mask.any():
            return [original_with_na.copy() for _ in range(self.n_imputations)]

        all_datasets = []
        for m_idx in range(self.n_imputations):
            seed_m = self.random_seed + m_idx * 100 + self.iteration_num
            np.random.seed(seed_m)
            current_df = original_with_na.copy()
            # Initialize with mean
            current_df.loc[missing_mask, target] = fallback

            for mice_it in range(self.mice_iters):
                if not missing_mask.any():
                    break
                pred_df = self._impute_predictors_mean(current_df.copy(), pred_cols)
                X_tr = pred_df.loc[~missing_mask, pred_cols]
                y_tr = original_with_na.loc[~missing_mask, target]
                X_pr = pred_df.loc[missing_mask, pred_cols]
                common = X_tr.dropna(how="any").index.intersection(y_tr.dropna().index)
                X_tr_c = X_tr.loc[common]
                y_tr_c = y_tr.loc[common]
                X_pr_c = X_pr.dropna(how="any")
                if X_tr_c.empty or len(X_tr_c) < 5 or X_pr_c.empty:
                    break
                try:
                    model = lgb.LGBMRegressor(
                        n_estimators=self.lgbm_n_est,
                        max_depth=self.lgbm_max_depth,
                        learning_rate=self.lgbm_lr,
                        num_leaves=self.lgbm_num_leaves,
                        random_state=seed_m + mice_it,
                        verbosity=-1,
                        n_jobs=1,
                    )
                    model.fit(X_tr_c, y_tr_c)
                    preds = model.predict(X_pr_c)
                    if self.add_noise:
                        sd = self._residual_sd(model, X_tr_c, y_tr_c)
                        if sd > 0:
                            preds = preds + np.random.normal(0, sd, size=len(preds))
                    current_df.loc[X_pr_c.index, target] = preds
                    still = current_df[target].isna() & missing_mask
                    if still.any():
                        current_df.loc[still, target] = fallback
                except Exception as e:
                    log.debug(f"MICE LGB iteration {mice_it} failed: {e}")
                    current_df.loc[missing_mask, target] = fallback
                    break

            all_datasets.append(current_df)
        return all_datasets


# ============================================================================
# STEP 6 -- RUBIN'S RULES POOLING
# ============================================================================
def _pool_mi_results(fits: list, key_var_coefs: list = None):
    """Pool M statsmodels OLS results using Rubin's Rules."""
    valid = [(f, c) for f, c in zip(fits, key_var_coefs or [None]*len(fits)) if f is not None]
    if not valid:
        return None
    M = len(valid)
    coefs = np.array([c["coef"] for _, c in valid if c is not None and pd.notna(c.get("coef", np.nan))])
    ses = np.array([c["se"] for _, c in valid if c is not None and pd.notna(c.get("se", np.nan))])
    nobs_list = [c["nobs"] for _, c in valid if c is not None and "nobs" in c]
    if len(coefs) == 0:
        return None
    M_v = len(coefs)
    pooled_coef = float(np.mean(coefs))
    within_var = float(np.mean(ses ** 2))
    between_var = float(np.var(coefs, ddof=1)) if M_v > 1 else 0.0
    total_var = within_var + (1 + 1.0 / M_v) * between_var
    total_se = float(np.sqrt(max(total_var, 1e-12)))
    if total_se > 0:
        t_stat = abs(pooled_coef / total_se)
        df_rubin = max(M_v - 1, 1)
        pooled_pval = float(2 * (1 - t_dist.cdf(t_stat, df=df_rubin)))
    else:
        pooled_pval = np.nan
    pooled_n = int(np.mean(nobs_list)) if nobs_list else 0
    fmi = between_var / total_var if total_var > 0 else np.nan
    re = 1.0 / (1.0 + fmi / M_v) if pd.notna(fmi) else np.nan
    mcse_se_ratio = 1.0 / np.sqrt(2 * M_v) if M_v > 0 else np.nan

    result = SimpleNamespace()
    result._is_pooled_mi = True
    result._pooled_coef = pooled_coef
    result._pooled_se = total_se
    result._pooled_pval = pooled_pval
    result._pooled_n = pooled_n
    result._fmi = fmi
    result._re = re
    result._mcse_se_ratio = mcse_se_ratio
    result._per_m_coefs = coefs.tolist()
    return result


# ============================================================================
# STEP 7 -- WILSON CI
# ============================================================================
def _wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (0.0, 0.0)
    z = sp_stats.norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# ============================================================================
# STEP 8 -- RESUMABILITY
# ============================================================================
def _load_done_set(log_path: Path) -> set:
    done = set()
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("ERROR:"):
                    done.add(line)
    return done


def _mark_done(log_path: Path, key: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(key + "\n")


def _combo_key(mechanism, pct_str, key_var, method, iteration):
    return f"{mechanism}|{pct_str}|{key_var}|{method}|{iteration}"


def _make_seed(mechanism, prop, key_var, iteration):
    s = f"{mechanism}{prop:.4f}{key_var}{iteration}"
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)


# ============================================================================
# STEP 9 -- STABILITY ACCUMULATOR
# ============================================================================
def _new_stability():
    return defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: {"both_same": 0, "ss": 0, "total": 0})
            )
        )
    )


def _update_stability(stability, key_var, mechanism, pct_str, method, coef_info, baseline_coef):
    cell = stability[key_var][mechanism][pct_str][method]
    cell["total"] += 1
    if coef_info is None or baseline_coef is None:
        return
    cell["sum_nobs"] = cell.get("sum_nobs", 0) + coef_info.get("nobs", 0)
    sign_same = coef_info["sign"] == baseline_coef["sign"]
    sig_same = coef_info["sig"] == baseline_coef["sig"]
    if sign_same and sig_same:
        cell["both_same"] += 1
    elif sign_same and not sig_same:
        cell["ss"] += 1


# ============================================================================
# STEP 10 -- OVER-IMPUTATION METRICS
# ============================================================================
def _over_imputation_metrics(df_original, df_imputed, key_var, missing_mask):
    try:
        true_vals = df_original.loc[missing_mask, key_var].dropna()
        imp_vals = df_imputed.loc[missing_mask, key_var]
        if isinstance(imp_vals, pd.DataFrame):
            imp_vals = imp_vals.iloc[:, 0]
        imp_vals = imp_vals.dropna()
        common = true_vals.index.intersection(imp_vals.index)
        if len(common) < 2:
            return {"rmse": np.nan, "mae": np.nan, "coverage": np.nan, "smd": np.nan}
        tv = true_vals.loc[common].values
        iv = imp_vals.loc[common].values
        rmse = float(np.sqrt(np.mean((iv - tv) ** 2)))
        mae = float(np.mean(np.abs(iv - tv)))
        std_true = float(df_original[key_var].std())
        coverage = float(np.mean(np.abs(iv - tv) < 1.96 * std_true)) if std_true > 0 else np.nan
        global_mean = float(df_original[key_var].mean())
        smd = float(abs(iv.mean() - global_mean) / std_true) if std_true > 0 else np.nan
        return {"rmse": rmse, "mae": mae, "coverage": coverage, "smd": smd}
    except Exception:
        return {"rmse": np.nan, "mae": np.nan, "coverage": np.nan, "smd": np.nan}


# ============================================================================
# STEP 11 -- EXCEL WORKBOOK WRITER
# ============================================================================
def _stability_to_df(stability_data, key_var, mechanism):
    """Convert stability accumulator to summary DataFrame for one mechanism."""
    levels = Config.MISSINGNESS_LEVELS
    pct_strs = [f"{int(p*100)}pct" for p in levels]
    rows = []
    for method in Config.METHODS:
        row = {"Method": method}
        for pct_str in pct_strs:
            cell = stability_data.get(key_var, {}).get(mechanism, {}).get(pct_str, {}).get(method, {})
            n = cell.get("total", 0)
            bs = cell.get("both_same", 0)
            ss = cell.get("ss", 0)
            b_prop = bs / n if n > 0 else np.nan
            lo, hi = _wilson_ci(bs, n) if n > 0 else (np.nan, np.nan)
            ss_prop = ss / n if n > 0 else np.nan
            row[f"B_{pct_str}"] = round(b_prop * 100, 1) if pd.notna(b_prop) else ""
            row[f"B_lo_{pct_str}"] = round(lo * 100, 1) if pd.notna(lo) else ""
            row[f"B_hi_{pct_str}"] = round(hi * 100, 1) if pd.notna(hi) else ""
            row[f"SS_{pct_str}"] = round(ss_prop * 100, 1) if pd.notna(ss_prop) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def write_excel_report(
    df: pd.DataFrame,
    stability: dict,
    baseline_fit,
    sim_data: dict,
    mode: str,
    output_path: Path,
) -> None:
    """Write the 17-sheet Excel report."""
    import openpyxl  # noqa: F401
    is_smoke = mode == "smoke"
    smoke_note = "[SMOKE TEST -- LIMITED DATA]" if is_smoke else ""
    key_vars_run = Config.KEY_VARIABLES[:1] if is_smoke else Config.KEY_VARIABLES
    levels_run = Config.SMOKE_MISSINGNESS_LEVELS if is_smoke else Config.MISSINGNESS_LEVELS

    log.info(f"Writing Excel workbook to {output_path} ...")

    # ---- Sheet 1: Baseline_Descriptives ----
    desc_cols = [Config.DEPENDENT_VAR, Config.FOCAL_IV] + Config.CONTROLS
    desc_cols = [c for c in desc_cols if c in df.columns]
    desc_df = df[desc_cols].describe().T
    desc_df.index.name = "Variable"

    # ---- Sheet 2: Baseline_Correlations ----
    corr_df = df[desc_cols].corr()

    # ---- Sheet 3: Baseline_Regression ----
    try:
        focal = Config.FOCAL_IV
        display_vars = [focal] + Config.CONTROLS
        reg_rows = []
        for var in display_vars:
            if var in baseline_fit.params.index:
                reg_rows.append({
                    "Variable": var,
                    "Coef": round(float(baseline_fit.params[var]), 6),
                    "SE": round(float(baseline_fit.bse[var]), 6),
                    "pvalue": round(float(baseline_fit.pvalues[var]), 6),
                    "R2": round(float(baseline_fit.rsquared), 4) if var == focal else "",
                    "N": int(baseline_fit.nobs) if var == focal else "",
                })
        reg_df = pd.DataFrame(reg_rows)
    except Exception:
        reg_df = pd.DataFrame({"Variable": ["statsmodels result not available"]})

    # ---- Sheets 4-6: Mean_Stability_* ----
    stab_sheets = {}
    for mech in Config.MECHANISMS:
        dfs = []
        for kv in key_vars_run:
            df_kv = _stability_to_df(stability, kv, mech)
            df_kv.insert(0, "KeyVar", kv)
            dfs.append(df_kv)
        stab_sheets[f"Mean_Stability_{mech}"] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # ---- Sheet 7: Model_Comparison ----
    coef_records = sim_data.get("coef_records", [])
    if coef_records:
        comp_df = pd.DataFrame(coef_records)
        baseline_coef_val = sim_data.get("baseline_coef_val", np.nan)
        if pd.notna(baseline_coef_val) and "coef" in comp_df.columns:
            comp_df["coef_dev"] = (comp_df["coef"] - baseline_coef_val).abs()
        baseline_se_val = sim_data.get("baseline_se_val", np.nan)
        if pd.notna(baseline_se_val) and "se" in comp_df.columns:
            comp_df["rel_se"] = comp_df["se"] / baseline_se_val
        if "coef_dev" in comp_df.columns:
            agg_dict = {"rmse": ("coef_dev", lambda x: np.sqrt(np.mean(x**2))), "n_iters": ("coef", "count")}
            if "rel_se" in comp_df.columns:
                agg_dict["avg_rel_se"] = ("rel_se", "mean")
            model_comp = comp_df.groupby(["method", "mechanism", "pct_str"]).agg(**agg_dict).reset_index()
        else:
            model_comp = comp_df.groupby(["method", "mechanism", "pct_str"]).agg(n_iters=("coef", "count")).reset_index()
    else:
        model_comp = pd.DataFrame({"Note": [f"No data {smoke_note}"]})

    # ---- Sheet 8: Stats_Features ----
    imp_stats = sim_data.get("imputed_stats", {})
    stats_rows = []
    for kv in key_vars_run:
        for mech in Config.MECHANISMS:
            for pct_str in [f"{int(p*100)}pct" for p in levels_run]:
                for method in Config.METHODS:
                    vals = imp_stats.get(kv, {}).get(mech, {}).get(pct_str, {}).get(method, [])
                    if vals:
                        vars_list = [v["var"] for v in vals if pd.notna(v.get("var", np.nan))]
                        skews_list = [v["skew"] for v in vals if pd.notna(v.get("skew", np.nan))]
                        stats_rows.append({
                            "KeyVar": kv, "Mechanism": mech, "Proportion": pct_str, "Method": method,
                            "AvgVariance": np.mean(vars_list) if vars_list else np.nan,
                            "AvgSkewness": np.mean(skews_list) if skews_list else np.nan,
                            "N": len(vals),
                        })
    stats_df = pd.DataFrame(stats_rows) if stats_rows else pd.DataFrame({"Note": [f"No data {smoke_note}"]})

    # ---- Sheet 9: Coef_Stability_Summary ----
    summary_rows = []
    for kv in key_vars_run:
        for mech in Config.MECHANISMS:
            for method in Config.METHODS:
                for pct_str in [f"{int(p*100)}pct" for p in levels_run]:
                    cell = stability.get(kv, {}).get(mech, {}).get(pct_str, {}).get(method, {})
                    n = cell.get("total", 0)
                    bs = cell.get("both_same", 0)
                    ss = cell.get("ss", 0)
                    b_prop = bs / n if n > 0 else np.nan
                    ss_prop = ss / n if n > 0 else np.nan
                    lo, hi = _wilson_ci(bs, n) if n > 0 else (np.nan, np.nan)
                    summary_rows.append({
                        "KeyVar": kv, "Mechanism": mech, "Proportion": pct_str, "Method": method,
                        "N_iters": n, "BothSame": bs, "SignSameSigChanged": ss,
                        "B_prop": round(b_prop * 100, 1) if pd.notna(b_prop) else np.nan,
                        "SS_prop": round(ss_prop * 100, 1) if pd.notna(ss_prop) else np.nan,
                        "B_CI_lo": round(lo * 100, 1) if pd.notna(lo) else np.nan,
                        "B_CI_hi": round(hi * 100, 1) if pd.notna(hi) else np.nan,
                        "Mean_N_obs": round(cell.get("sum_nobs", 0) / n, 1) if n > 0 else np.nan,
                    })
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame({"Note": [f"No data {smoke_note}"]})

    # ---- Sheet 10: Benchmark_Methods ----
    if summary_rows:
        bench_df = pd.DataFrame(summary_rows)
        bench_df = bench_df.dropna(subset=["B_prop"]).groupby("Method")["B_prop"].agg(
            ["mean", "min", "max", "count"]
        ).round(1).reset_index()
        bench_df.columns = ["Method", "AvgB_pct", "MinB_pct", "MaxB_pct", "N_cells"]
    else:
        bench_df = pd.DataFrame({"Note": [f"No data {smoke_note}"]})

    # ---- Sheet 11: MI_Diagnostics ----
    mi_diag = sim_data.get("mi_diagnostics", [])
    mi_diag_df = pd.DataFrame(mi_diag) if mi_diag else pd.DataFrame({"Note": [f"No data {smoke_note}"]})

    # ---- Sheet 12: MI_Trace ----
    mi_trace = sim_data.get("mi_traces", [])
    mi_trace_df = pd.DataFrame(mi_trace) if mi_trace else pd.DataFrame({"Note": [f"No data {smoke_note}"]})

    # ---- Sheet 13: MI_Overimputation ----
    oi_records = sim_data.get("over_imputation", [])
    oi_df = pd.DataFrame(oi_records) if oi_records else pd.DataFrame({"Note": [f"No data {smoke_note}"]})

    # ---- Sheet 14: MI_Distribution ----
    dist_records = sim_data.get("dist_compare", [])
    dist_df = pd.DataFrame(dist_records) if dist_records else pd.DataFrame({"Note": [f"No data {smoke_note}"]})

    # ---- Sheet 15: Missingness_Patterns ----
    miss_records = sim_data.get("missingness_patterns", [])
    miss_df = pd.DataFrame(miss_records) if miss_records else pd.DataFrame({"Note": ["No data collected"]})

    # ---- Sheet 16: NMAR_Residual ----
    nmar_res = sim_data.get("nmar_residual", [])
    nmar_res_df = pd.DataFrame(nmar_res) if nmar_res else pd.DataFrame({"Note": [f"No data {smoke_note}"]})

    # ---- Sheet 17: NMAR_Delta ----
    nmar_delta = sim_data.get("nmar_delta", [])
    nmar_delta_df = pd.DataFrame(nmar_delta) if nmar_delta else pd.DataFrame({"Note": [f"No data {smoke_note}"]})

    # ---- Write all sheets ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        desc_df.to_excel(writer, sheet_name="Baseline_Descriptives")
        corr_df.to_excel(writer, sheet_name="Baseline_Correlations")
        reg_df.to_excel(writer, sheet_name="Baseline_Regression", index=False)
        for sheet_name, df_s in stab_sheets.items():
            df_s.to_excel(writer, sheet_name=sheet_name, index=False)
        model_comp.to_excel(writer, sheet_name="Model_Comparison", index=False)
        stats_df.to_excel(writer, sheet_name="Stats_Features", index=False)
        summary_df.to_excel(writer, sheet_name="Coef_Stability_Summary", index=False)
        bench_df.to_excel(writer, sheet_name="Benchmark_Methods", index=False)
        mi_diag_df.to_excel(writer, sheet_name="MI_Diagnostics", index=False)
        mi_trace_df.to_excel(writer, sheet_name="MI_Trace", index=False)
        oi_df.to_excel(writer, sheet_name="MI_Overimputation", index=False)
        dist_df.to_excel(writer, sheet_name="MI_Distribution", index=False)
        miss_df.to_excel(writer, sheet_name="Missingness_Patterns", index=False)
        nmar_res_df.to_excel(writer, sheet_name="NMAR_Residual", index=False)
        nmar_delta_df.to_excel(writer, sheet_name="NMAR_Delta", index=False)

    log.info(f"Workbook written: {output_path}")


# ============================================================================
# STEP 12 -- COMPUTE NMAR DIAGNOSTICS (post-hoc)
# ============================================================================
def _compute_nmar_residual(df: pd.DataFrame, baseline_fit, key_vars: list) -> list:
    """Logistic regression of missingness indicator on baseline residuals."""
    from sklearn.linear_model import LogisticRegression
    records = []
    try:
        # statsmodels exposes residuals directly
        resids = baseline_fit.resid.values
        reg_vars = (
            [Config.DEPENDENT_VAR, Config.FOCAL_IV]
            + Config.CONTROLS
            + Config.GENRE_DUMMIES
            + [Config.FE_YEAR_VAR, Config.FE_MONTH_VAR]
        )
        reg_vars_present = [v for v in reg_vars if v in df.columns]
        df_reg = df[reg_vars_present].dropna().reset_index(drop=True)

        for kv in key_vars:
            for prop in [0.20]:
                seed = _make_seed("NMAR_diag", prop, kv, 0)
                df_miss = simulate_missingness_single_col(
                    df, kv, prop, seed, "NMAR", Config.MAR_CONTROL, Config.MAR_NMAR_STRENGTH
                )
                # Align missingness indicator with regression sample
                miss_indicator = df_miss[kv].isna().astype(int)
                # Only use rows in df_reg (the baseline regression sample)
                common_idx = df_reg.index
                n_common = min(len(common_idx), len(resids), len(miss_indicator))
                if n_common < 20:
                    continue
                X_logit = pd.DataFrame({"resid": resids[:n_common]})
                y_logit = miss_indicator.iloc[:n_common].values
                if y_logit.sum() == 0 or y_logit.sum() == len(y_logit):
                    continue
                try:
                    clf = LogisticRegression(max_iter=200)
                    clf.fit(X_logit, y_logit)
                    coef = float(clf.coef_[0][0])
                    records.append({"KeyVar": kv, "Proportion": f"{int(prop*100)}pct",
                                    "LogitCoef_resid": round(coef, 4),
                                    "Interpretation": "Positive=NMAR (high values more likely missing)"})
                except Exception:
                    pass
    except Exception as e:
        log.debug(f"NMAR residual analysis failed: {e}")
    return records


def _compute_nmar_delta(df: pd.DataFrame, key_vars: list) -> list:
    """Sensitivity: vary NMAR strength and track focal coefficient under LD."""
    records = []
    deltas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    prop = 0.20
    for kv in key_vars:
        for delta in deltas:
            seed = _make_seed("NMAR_delta", prop, kv, int(delta * 10))
            mech = "MCAR" if delta == 0 else "NMAR"
            df_miss = simulate_missingness_single_col(
                df, kv, prop, seed, mech, Config.MAR_CONTROL, delta if delta > 0 else 1.5
            )
            try:
                pipeline = ImputationPipeline(
                    df_miss, df, kv, Config.PREDICTOR_POOL, 0,
                    n_imputations=1, random_seed=seed,
                )
                df_ld = pipeline.listwise_deletion()
                fit = _run_simulation_regression(df_ld)
                ci = _extract_focal_coef(fit)
                if ci:
                    records.append({"KeyVar": kv, "Delta": delta, "Prop": prop,
                                    "FocalCoef": round(ci["coef"], 4),
                                    "SE": round(ci["se"], 4),
                                    "pval": round(ci["pval"], 4),
                                    "Nobs": ci["nobs"]})
            except Exception:
                pass
    return records


# ============================================================================
# STEP 13 -- MAIN SIMULATION LOOP
# ============================================================================
def run_simulation(df: pd.DataFrame, baseline_fit, mode: str) -> None:
    if mode == "smoke":
        levels = Config.SMOKE_MISSINGNESS_LEVELS
        n_iters = Config.SMOKE_ITERATIONS
        key_vars = Config.KEY_VARIABLES[:1]
        wb_path = Config.SMOKE_WORKBOOK
        log.info(f"SMOKE TEST: {len(Config.MECHANISMS)} mechs x {len(levels)} levels x "
                 f"{len(key_vars)} var x {n_iters} iters x {len(Config.METHODS)} methods = "
                 f"{len(Config.MECHANISMS)*len(levels)*len(key_vars)*n_iters*len(Config.METHODS)} runs")
    else:
        levels = Config.MISSINGNESS_LEVELS
        n_iters = Config.NUM_ITERATIONS_PER_SCENARIO
        key_vars = Config.KEY_VARIABLES
        wb_path = Config.REPORT_WORKBOOK
        log.info(f"FULL RUN: {len(Config.MECHANISMS)} x {len(levels)} x {len(key_vars)} x "
                 f"{n_iters} x {len(Config.METHODS)} = "
                 f"{len(Config.MECHANISMS)*len(levels)*len(key_vars)*n_iters*len(Config.METHODS)} runs")

    done_set = _load_done_set(Config.PROGRESS_LOG)
    log.info(f"Resuming: {len(done_set)} combinations already complete")

    baseline_coef = _extract_focal_coef(baseline_fit)
    log.info(f"Baseline focal coef: {baseline_coef}")

    stability = _new_stability()
    sim_data = {
        "baseline_coef_val": baseline_coef["coef"] if baseline_coef else np.nan,
        "baseline_se_val": baseline_coef["se"] if baseline_coef else np.nan,
        "coef_records": [],
        "imputed_stats": defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))),
        "mi_diagnostics": [],
        "mi_traces": [],
        "over_imputation": [],
        "dist_compare": [],
        "missingness_patterns": [],
    }

    df_clean = df.copy()
    total_done = 0
    total_err = 0

    for mechanism in Config.MECHANISMS:
        for prop in levels:
            pct_str = f"{int(prop * 100)}pct"
            for key_var in key_vars:
                log.info(f"  {mechanism} | {pct_str} | {key_var}")
                for iteration in range(n_iters):
                    seed = _make_seed(mechanism, prop, key_var, iteration)
                    df_missing = simulate_missingness_single_col(
                        df_clean, key_var, prop, seed, mechanism,
                        Config.MAR_CONTROL, Config.MAR_NMAR_STRENGTH
                    )
                    # Record actual missingness rate
                    actual_prop = float(df_missing[key_var].isna().mean())
                    sim_data["missingness_patterns"].append({
                        "KeyVar": key_var, "Mechanism": mechanism, "Proportion": pct_str,
                        "TargetProp": prop, "ActualProp": round(actual_prop, 4),
                        "N_missing": int(df_missing[key_var].isna().sum()),
                        "Iteration": iteration,
                    })

                    pipeline = ImputationPipeline(
                        df_missing, df_clean, key_var, Config.PREDICTOR_POOL,
                        iteration,
                        n_imputations=Config.N_IMPUTATIONS,
                        mice_iters=Config.MICE_ITERATIONS,
                        random_seed=Config.RANDOM_SEED + iteration,
                        add_noise=Config.ADD_RESIDUAL_NOISE,
                        lgbm_n_est=Config.MICE_LGBM_N_ESTIMATORS,
                        lgbm_max_depth=Config.MICE_LGBM_MAX_DEPTH,
                        lgbm_lr=Config.MICE_LGBM_LEARNING_RATE,
                        lgbm_num_leaves=Config.MICE_LGBM_NUM_LEAVES,
                        dl_epochs=Config.DL_EPOCHS,
                        dl_patience=Config.DL_PATIENCE,
                    )

                    for method in Config.METHODS:
                        combo_key = _combo_key(mechanism, pct_str, key_var, method, iteration)
                        if combo_key in done_set:
                            total_done += 1
                            continue

                        txt_path = (
                            Config.REGRESSION_TXT_DIR
                            / mechanism
                            / pct_str
                            / key_var
                            / method
                            / f"iter{iteration}_model_{key_var}.txt"
                        )

                        try:
                            if method == "LD":
                                imputed = pipeline.listwise_deletion()
                                fit = _run_simulation_regression(imputed)
                            elif method == "Mean":
                                imputed = pipeline.mean_imputation()
                                fit = _run_simulation_regression(imputed)
                            elif method == "Reg":
                                imputed = pipeline.regression_imputation()
                                fit = _run_simulation_regression(imputed)
                            elif method == "Iter":
                                imputed = pipeline.stochastic_iterative_imputation()
                                fit = _run_simulation_regression(imputed)
                            elif method == "RF":
                                imputed = pipeline.ml_imputation()
                                fit = _run_simulation_regression(imputed)
                            elif method == "DL":
                                imputed = pipeline.deep_learning_imputation()
                                fit = _run_simulation_regression(imputed)
                            elif method == "MILGBM":
                                mi_datasets = pipeline.custom_multiple_imputation()
                                sub_fits = []
                                sub_coefs = []
                                for m_idx, mi_df in enumerate(mi_datasets):
                                    try:
                                        sub_fit = _run_simulation_regression(mi_df)
                                        sub_ci = _extract_focal_coef(sub_fit)
                                        sub_fits.append(sub_fit)
                                        sub_coefs.append(sub_ci)
                                        if sub_ci:
                                            sim_data["mi_traces"].append({
                                                "KeyVar": key_var, "Mechanism": mechanism,
                                                "Proportion": pct_str, "Iteration": iteration,
                                                "M": m_idx, "Coef": sub_ci["coef"],
                                            })
                                    except Exception:
                                        sub_fits.append(None)
                                        sub_coefs.append(None)
                                pooled = _pool_mi_results(sub_fits, sub_coefs)
                                fit = pooled
                                imputed = mi_datasets[0] if mi_datasets else df_missing
                                if pooled is not None:
                                    sim_data["mi_diagnostics"].append({
                                        "KeyVar": key_var, "Mechanism": mechanism,
                                        "Proportion": pct_str, "Iteration": iteration,
                                        "FMI": getattr(pooled, "_fmi", np.nan),
                                        "RE": getattr(pooled, "_re", np.nan),
                                        "MCSE_SE": getattr(pooled, "_mcse_se_ratio", np.nan),
                                        "M_valid": len([c for c in sub_coefs if c is not None]),
                                    })
                            else:
                                raise ValueError(f"Unknown method: {method}")

                            _save_iter_txt(fit, txt_path)
                            coef_info = _extract_focal_coef(fit)
                            _update_stability(stability, key_var, mechanism, pct_str, method, coef_info, baseline_coef)

                            if coef_info:
                                sim_data["coef_records"].append({
                                    "key_var": key_var, "mechanism": mechanism,
                                    "pct": prop, "pct_str": pct_str,
                                    "method": method, "iter": iteration,
                                    "coef": coef_info["coef"],
                                    "se": coef_info["se"],
                                    "pval": coef_info["pval"],
                                    "nobs": coef_info["nobs"],
                                })

                                try:
                                    if hasattr(imputed, "__len__") and not isinstance(imputed, list):
                                        imp_col = imputed[key_var] if key_var in imputed.columns else pd.Series()
                                        imp_col = imp_col.dropna()
                                        if len(imp_col) > 1:
                                            sim_data["imputed_stats"][key_var][mechanism][pct_str][method].append({
                                                "var": float(imp_col.var()),
                                                "skew": float(sp_stats.skew(imp_col.values)),
                                            })
                                except Exception:
                                    pass

                                missing_mask_bool = df_missing[key_var].isna()
                                oi = _over_imputation_metrics(df_clean, imputed, key_var, missing_mask_bool)
                                sim_data["over_imputation"].append({
                                    "KeyVar": key_var, "Mechanism": mechanism,
                                    "Proportion": pct_str, "Method": method, "Iteration": iteration,
                                    **oi,
                                })

                                try:
                                    orig_col = df_clean[key_var].dropna()
                                    if hasattr(imputed, "__len__") and not isinstance(imputed, list):
                                        imp_col_all = imputed[key_var].dropna() if key_var in imputed.columns else pd.Series()
                                        if len(imp_col_all) > 10 and len(orig_col) > 10:
                                            ks_stat, ks_pval = sp_stats.ks_2samp(orig_col.values, imp_col_all.values)
                                            mean_diff = float(imp_col_all.mean() - orig_col.mean())
                                            std_ratio = float(imp_col_all.std() / orig_col.std()) if orig_col.std() > 0 else np.nan
                                            sim_data["dist_compare"].append({
                                                "KeyVar": key_var, "Mechanism": mechanism,
                                                "Proportion": pct_str, "Method": method, "Iteration": iteration,
                                                "KS_stat": round(ks_stat, 4), "KS_pval": round(ks_pval, 4),
                                                "MeanDiff": round(mean_diff, 4),
                                                "StdRatio": round(std_ratio, 4) if pd.notna(std_ratio) else np.nan,
                                            })
                                except Exception:
                                    pass

                            _mark_done(Config.PROGRESS_LOG, combo_key)
                            total_done += 1

                            if total_done % 50 == 0:
                                log.info(f"    Progress: {total_done} done, {total_err} errors")

                        except Exception as e:
                            total_err += 1
                            log.warning(f"ERROR: {combo_key}: {e}")
                            with open(Config.PROGRESS_LOG, "a", encoding="utf-8") as fh:
                                fh.write(f"ERROR:{combo_key}:{str(e)[:200]}\n")

    log.info(f"Simulation complete: {total_done} done, {total_err} errors")

    log.info("Computing NMAR diagnostics...")
    sim_data["nmar_residual"] = _compute_nmar_residual(df_clean, baseline_fit, key_vars)
    sim_data["nmar_delta"] = _compute_nmar_delta(df_clean, key_vars)

    write_excel_report(df_clean, stability, baseline_fit, sim_data, mode, wb_path)
    log.info(f"{'Smoke test' if mode == 'smoke' else 'Full simulation'} complete.")


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

    # Namespace progress log by mode so smoke and full runs cannot contaminate each other
    Config.PROGRESS_LOG = PAPER_DIR / "logs" / f"simulation_0017_progress_{args.mode}.log"

    log.info("=" * 60)
    log.info(f"Paper 0017 -- Mode: {args.mode.upper()}")
    log.info("=" * 60)

    df = load_and_inspect_data(force_reload=args.reload)
    baseline_fit = run_baseline_regression(df)

    if args.mode == "baseline":
        log.info("Baseline mode complete. Review output above and update confignotes.txt.")
        log.info("DO NOT proceed to simulation until baseline is confirmed matched.")
        return

    run_simulation(df, baseline_fit, mode=args.mode)


if __name__ == "__main__":
    main()
