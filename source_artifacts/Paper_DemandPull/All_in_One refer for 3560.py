import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning as smConvergenceWarning, PerfectSeparationError
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, norm, t as t_dist, ks_2samp
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn import __version__ as sklearn_version
# from packaging import version # Not strictly needed if not using version.parse directly
import math
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam # Changed from legacy
from tensorflow.keras.callbacks import EarlyStopping
from linearmodels.panel import PanelOLS
from linearmodels.panel.results import PanelEffectsResults      # For type checking
from patsy import dmatrices                                     # Formula handling / clean_formula_vars

import os
# !!! IMPORTANT: SET YOUR R_HOME PATH HERE if not set globally !!!
# If R_HOME is not set, rpy2 will try to find R automatically.
# If it fails, uncomment and set the line below.
# os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.1" # Example for Windows
# os.environ["R_HOME"] = r"/usr/lib/R" # Example for Linux

import logging
from pathlib import Path
import warnings
from typing import List, Dict, Tuple, Optional, Any, Union
from tabulate import tabulate
import re
import collections
from pandas.api.types import CategoricalDtype, is_numeric_dtype, is_object_dtype, is_string_dtype, is_integer_dtype, is_float_dtype
from tqdm import tqdm
import shutil
import openpyxl # Added for Excel writing

# --- Parallel Processing Imports ---
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle # For potentially serializing complex objects if needed, though direct pass is preferred
import glob
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import time

# --- rpy2 setup ---
R_OK = False
fixest_r = None
broom_r = None
base_r = None
stats_r = None

try:
    from rpy2.robjects.packages import importr
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
    
    rpy2_logger.setLevel(logging.ERROR) 
    
    # Test basic R connectivity first
    try:
        robjects.r('x <- 1')
        result = robjects.r('x')
        robjects.r('rm(x)')
        logging.info(f"R basic connectivity test successful: x = {result[0]}")
    except Exception as e_test:
        logging.error(f"R basic connectivity test failed: {e_test}")
        raise e_test
    
    # Import basic R packages
    base_r = importr('base')
    stats_r = importr('stats')
    
    # Try to import optional packages
    try:
        fixest_r = importr('fixest')
        # Set fixest to use 1 thread for stability in parallel Python
        robjects.r('setFixest_nthreads(1)')
        logging.info("fixest package imported successfully")
    except Exception as e_fixest:
        logging.warning(f"fixest import failed: {e_fixest}")
        fixest_r = None

    try:
        broom_r = importr('broom')
        logging.info("broom package imported successfully")
    except Exception as e_broom:
        logging.warning(f"broom import failed: {e_broom}")
        broom_r = None

    # Check if we have at least basic R functionality
    if base_r is not None and stats_r is not None:
        R_OK = True
        logging.info("Successfully imported basic R packages via rpy2")
        if fixest_r is not None:
            logging.info("fixest package available for advanced panel models")
        else:
            logging.warning("fixest package not available - advanced panel models will use Python fallback")
    else:
        R_OK = False
        logging.error("Failed to import basic R packages")
        
except Exception as e:
    logging.error(f"Failed to import R packages via rpy2: {e}. R regression functionality will be disabled. Falling back to Python models if possible.")
    R_OK = False
# --- end rpy2 setup ---

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=SklearnConvergenceWarning)
warnings.filterwarnings("ignore", message="kurtosistest only valid for n>=20")
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
warnings.filterwarnings('ignore', category=smConvergenceWarning)
warnings.simplefilter('ignore', PerfectSeparationError)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('lightgbm').setLevel(logging.WARNING)
logging.getLogger("linearmodels").setLevel(logging.WARNING) 

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration Class (Adapted for Santamaria et al. 2024) ---
class Config:
    # Files and Directories
    ORIGINAL_DATA_FILE: str = "DATA.csv" 
    MCAR_BASE_DIR: str = "MCAR_Data_Santamaria2024"
    MAR_BASE_DIR: str = "MAR_Data_Santamaria2024"
    NMAR_BASE_DIR: str = "NMAR_Data_Santamaria2024"
    OUTPUT_EXCEL_FILE: str = "Santamaria2024_Imputation_Analysis_Report.xlsx" # Changed from HTML to Excel
    REGRESSION_OUTPUT_DIR_TXT: str = "regression_txt_outputs_Santamaria2024"
    CLEANUP_IMPUTED_FILES: bool = True

    # Simulation Parameters
    MISSINGNESS_LEVELS: List[float] = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    SIMULATION_SEED: int = 456
    NUM_ITERATIONS_PER_SCENARIO: int = 30
    
    POTENTIAL_COVARIATES_NUMERIC: List[str] = [
        "Age", "TeamSize", "WorkExp_numeric", 
        "InitialRevenueLikert_Post0", "InitialCustomersLikert_Post0", 
        "Education_numeric", "Round_dummy", 
    ]
    POTENTIAL_COVARIATES_TO_DUMMIFY_FOR_IMPUTATION: List[str] = [ 
        "Gender", "Ethnicity", "STEM_status", "Business_status", "Working_status",
        "EntrepExperience_status", "STEMEXP_status", "Registered_status", "Sector"
    ]
    
    KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS: List[str] = [
        "Age", "TeamSize", "WorkExp_numeric", "InitialCustomersLikert_Post0"
    ]
    MAR_CONTROL_COL: str = "Age" 
    MAR_STRENGTH_FACTOR: float = 1.5
    NMAR_STRENGTH_FACTOR: float = 1.5

    ID_COLUMN_ORIGINAL: str = "SN" 
    ID_COLUMN_TIME: str = "Post" 
    ID_COLUMN: str = "SN_Post_UniqueID" 

    NUMERICAL_COLS_FOR_IMPUTATION: List[str] = [] 
    CATEGORICAL_COLS_RAW: List[str] = [ 
        "Gender", "Ethnicity", "HighestEducationAttained", "Fieldofstudy",
        "Studying", "Working", "EntrepExperience", "WorkExp", "IndustryExperience",
        "Registered", "Sector", "Round"
    ] 

    N_IMPUTATIONS: int = 5
    MICE_ITERATIONS: int = 5
    RANDOM_SEED_IMPUTATION: int = 42
    ADD_RESIDUAL_NOISE: bool = True
    DL_EPOCHS: int = 30
    DL_PATIENCE: int = 5
    MICE_LGBM_N_ESTIMATORS: int = 30
    MICE_LGBM_MAX_DEPTH: int = 4
    MICE_LGBM_LEARNING_RATE: float = 0.05
    MICE_LGBM_NUM_LEAVES: int = 10
    MICE_LGBM_VERBOSITY: int = -1

    ALPHA: float = 0.05
    COLS_DESCRIPTIVE: List[str] = ["CustomersLikert", "Treatment", "Post", "Age", "TeamSize"]
    COLS_CORRELATION: List[str] = COLS_DESCRIPTIVE

    MODEL_FORMULAS: Dict[str, str] = {
        "santamaria_M4_customers_likert": "CustomersLikert ~ Post * Treatment"
    }
    MODEL_FAMILIES: Dict[str, Any] = {} 
    IMPUTATION_METHODS_TO_COMPARE: List[str] = [
        "listwise_deletion", "mean_imputation", "regression_imputation",
        "stochastic_iterative_imputation", "ml_imputation", "deep_learning_imputation",
        "custom_multiple_imputation",
    ]
    METHOD_DISPLAY_NAMES: Dict[str, str] = {
        "listwise_deletion": "Listwise Deletion", "mean_imputation": "Mean",
        "regression_imputation": "Regression (+Noise)",
        "stochastic_iterative_imputation": "Iterative (+Sample)",
        "ml_imputation": "ML (RF +Noise)", "deep_learning_imputation": "DL (MLP +Noise)",
        "custom_multiple_imputation": "MI (Custom LGBM MICE)",
    }
    MODEL_FIXED_EFFECTS: Dict[str, Optional[List[str]]] = {
        "santamaria_M4_customers_likert": ["SN"] 
    }
    MODEL_CLUSTER_SE: Dict[str, Optional[str]] = {
        "santamaria_M4_customers_likert": "SN" 
    }
    MODEL_WEIGHTS_COL: Dict[str, Optional[str]] = {
        "santamaria_M4_customers_likert": None 
    }
    MODEL_ABSORBED_IV_BY_FE: Dict[str, List[str]] = {
        "santamaria_M4_customers_likert": ["Treatment"] 
    }
    MODEL_USE_PANEL_ESTIMATOR: Dict[str, bool] = {
        "santamaria_M4_customers_likert": True 
    }

    MAIN_INTERACTION_TERM_FOR_TRACKING = "Post:Treatment" 
    KEY_VARS_AND_THEIR_MODEL_COEFS: Dict[str, Dict[str, List[str]]] = {} 
    KEY_VARS_FOR_STATS_TABLE: List[str] = ["Age", "TeamSize"] 

    MAX_WORKERS: int = max(1, cpu_count() // 2 if cpu_count() else 1)
    CHUNK_SIZE: int = 1 
    USE_PARALLEL: bool = True 

    def __init__(self):
        # Initialize NUMERICAL_COLS_FOR_IMPUTATION properly
        Config.NUMERICAL_COLS_FOR_IMPUTATION = list(Config.POTENTIAL_COVARIATES_NUMERIC)
        all_potential_predictors = Config.POTENTIAL_COVARIATES_NUMERIC + \
                                   Config.POTENTIAL_COVARIATES_TO_DUMMIFY_FOR_IMPUTATION + \
                                   Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS
        # Remove duplicates and sort
        Config.NUMERICAL_COLS_FOR_IMPUTATION = sorted(list(set(all_potential_predictors)))
        
        # Initialize KEY_VARS_AND_THEIR_MODEL_COEFS
        for kvar in Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS:
            tracked_coefs = [Config.MAIN_INTERACTION_TERM_FOR_TRACKING]
            Config.KEY_VARS_AND_THEIR_MODEL_COEFS[kvar] = {
                "santamaria_M4_customers_likert": tracked_coefs
            }

    @classmethod
    def get_data_path(cls, mechanism: str, missingness_level: float, type: str = "simulated", 
                       method_name: Optional[str] = None, iteration: Optional[int] = None, 
                       key_var_imputed_for_path: Optional[str] = None) -> str:
        if mechanism.upper() == "MCAR": base_dir = cls.MCAR_BASE_DIR
        elif mechanism.upper() == "MAR": base_dir = cls.MAR_BASE_DIR
        elif mechanism.upper() == "NMAR": base_dir = cls.NMAR_BASE_DIR
        else: raise ValueError(f"Unknown mechanism: {mechanism}")

        level_dir_name = f"{int(missingness_level * 100)}pct_missing"
        path_parts = [base_dir]
        if key_var_imputed_for_path: path_parts.append(f"imputed_for_{key_var_imputed_for_path}")
        path_parts.append(level_dir_name)
        if iteration is not None: path_parts.append(f"iter_{iteration}")
        
        current_level_dir = os.path.join(*path_parts)
        
        if type == "simulated":
            return os.path.join(current_level_dir, "simulated_data_with_missing.csv")
        elif type == "imputed_dir":
            imputed_base = os.path.join(current_level_dir, "imputed_data")
            return imputed_base
        elif type == "imputed_file" and method_name:
            imputed_base = os.path.join(current_level_dir, "imputed_data")
            return os.path.join(imputed_base, f"{method_name}.csv")
        raise ValueError("Invalid type or missing method_name for get_data_path")


# --- Utility Functions (Residual SD calculation) ---
def calculate_residual_sd(model, X_train, y_train) -> float:
    if X_train.empty or y_train.empty: return 0.0
    try:
        y_pred_train = model.predict(X_train)
        residuals = y_train - y_pred_train
        non_zero_residuals = residuals[np.abs(residuals) > 1e-9]
        if len(non_zero_residuals) < 2: return 0.0
        resid_sd = np.std(non_zero_residuals)
        if pd.isna(resid_sd) or resid_sd > np.std(y_train) * 5: 
             return min(np.std(y_train)*0.1, resid_sd if pd.notna(resid_sd) and resid_sd > 0 else 0.0) # Capped
        return resid_sd
    except Exception as e: logger.error(f"Error calculating residual SD: {e}"); return 0.0

# --- Data Simulation Function (MCAR, MAR, NMAR for a single specified column) ---
def simulate_missingness_single_col(
    df: pd.DataFrame,
    col_to_make_missing: str,
    miss_prop: float,
    seed: int,
    mechanism: str = "MCAR",
    mar_control_col: Optional[str] = None,
    mar_strength: float = 1.0,
    nmar_strength: float = 1.0
) -> pd.DataFrame:
    data_sim = df.copy()
    rng = np.random.default_rng(seed)

    if col_to_make_missing not in data_sim.columns:
        logger.warning(f"Simulate Missingness (Single Col): Column '{col_to_make_missing}' not found. Skipping.")
        return data_sim
    
    if not pd.api.types.is_numeric_dtype(data_sim[col_to_make_missing]):
        original_sum_na = data_sim[col_to_make_missing].isna().sum()
        data_sim[col_to_make_missing] = pd.to_numeric(data_sim[col_to_make_missing], errors='coerce')
        coerced_sum_na = data_sim[col_to_make_missing].isna().sum()
        if coerced_sum_na > original_sum_na:
            logger.warning(f"Simulate Missingness (Single Col): Coercing '{col_to_make_missing}' to numeric introduced {coerced_sum_na - original_sum_na} new NAs.")

    if not data_sim[col_to_make_missing].notna().any():
        logger.warning(f"Simulate Missingness (Single Col): Column '{col_to_make_missing}' has no non-NA values. Skipping.")
        return data_sim

    eligible_indices = data_sim.index[data_sim[col_to_make_missing].notna()].tolist()
    if not eligible_indices: 
        logger.warning(f"Simulate Missingness (Single Col): No eligible (non-NA) indices in '{col_to_make_missing}'. Skipping.")
        return data_sim

    n_eligible_cells = len(eligible_indices)
    n_to_make_missing = int(np.floor(miss_prop * n_eligible_cells))

    if n_to_make_missing <= 0: return data_sim
    n_to_make_missing = min(n_to_make_missing, n_eligible_cells)
    
    indices_to_nan_list = []

    if mechanism.upper() == "MCAR":
        indices_to_nan_list = rng.choice(eligible_indices, size=n_to_make_missing, replace=False)
    
    elif mechanism.upper() == "MAR":
        if mar_control_col is None or mar_control_col not in data_sim.columns:
            logger.error(f"MAR Error (Single Col): mar_control_col '{mar_control_col}' not found. Falling back to MCAR.")
            return simulate_missingness_single_col(df, col_to_make_missing, miss_prop, seed, "MCAR", mar_strength=mar_strength, nmar_strength=nmar_strength)
        
        control_values_series = pd.to_numeric(data_sim.loc[eligible_indices, mar_control_col], errors='coerce')
        if control_values_series.isna().any():
            mean_val = control_values_series.mean()
            if pd.notna(mean_val): control_values_series.fillna(mean_val, inplace=True)
            else: 
                  logger.warning(f"MAR Warning (Single Col): mar_control_col '{mar_control_col}' all NA for eligible indices. Fallback MCAR.")
                  return simulate_missingness_single_col(df, col_to_make_missing, miss_prop, seed, "MCAR", mar_strength=mar_strength, nmar_strength=nmar_strength)

        if control_values_series.nunique() <= 1:
             logger.warning(f"MAR Warning (Single Col): mar_control_col '{mar_control_col}' no variance for eligible indices. Fallback MCAR.")
             return simulate_missingness_single_col(df, col_to_make_missing, miss_prop, seed, "MCAR", mar_strength=mar_strength, nmar_strength=nmar_strength)
        
        min_val, max_val = control_values_series.min(), control_values_series.max()
        # Fix division by zero bug
        if max_val == min_val:
            logger.warning(f"MAR Warning (Single Col): mar_control_col '{mar_control_col}' has no variance after min/max calculation. Fallback MCAR.")
            return simulate_missingness_single_col(df, col_to_make_missing, miss_prop, seed, "MCAR", mar_strength=mar_strength, nmar_strength=nmar_strength)
            
        normalized_control = (control_values_series - min_val) / (max_val - min_val)
        weights = np.exp(normalized_control * mar_strength)
        
        # Fix infinite weights and zero sum bugs
        if not np.all(np.isfinite(weights)) or np.sum(weights) == 0:
            logger.warning(f"MAR Warning (Single Col): Invalid weights from mar_control_col '{mar_control_col}'. Fallback MCAR.")
            return simulate_missingness_single_col(df, col_to_make_missing, miss_prop, seed, "MCAR", mar_strength=mar_strength, nmar_strength=nmar_strength)
        probabilities = weights / np.sum(weights)
        
        if np.isnan(probabilities).any() or len(probabilities) != len(eligible_indices):
            logger.warning(f"MAR Warning (Single Col): Probabilities issue. Fallback MCAR.")
            return simulate_missingness_single_col(df, col_to_make_missing, miss_prop, seed, "MCAR", mar_strength=mar_strength, nmar_strength=nmar_strength)
        try:
            indices_to_nan_list = rng.choice(eligible_indices, size=n_to_make_missing, replace=False, p=probabilities)
        except ValueError as e:
            logger.error(f"MAR Error (Single Col) during weighted choice: {e}. Fallback MCAR.")
            return simulate_missingness_single_col(df, col_to_make_missing, miss_prop, seed, "MCAR", mar_strength=mar_strength, nmar_strength=nmar_strength)

    elif mechanism.upper() == "NMAR":
        values_for_nmarmiss = data_sim.loc[eligible_indices, col_to_make_missing].copy()
        
        mean_y, std_y = values_for_nmarmiss.mean(), values_for_nmarmiss.std()
        if pd.isna(std_y) or std_y < 1e-9:
            logger.warning(f"NMAR Warning (Single Col): Target col '{col_to_make_missing}' no variance. Fallback MCAR.")
            return simulate_missingness_single_col(df, col_to_make_missing, miss_prop, seed, "MCAR", mar_strength=mar_strength, nmar_strength=nmar_strength)
            
        standardized_values = (values_for_nmarmiss - mean_y) / std_y
        weights = np.exp(standardized_values * nmar_strength)
        
        # Fix infinite weights and zero sum bugs
        if not np.all(np.isfinite(weights)) or np.sum(weights) == 0:
            logger.warning(f"NMAR Warning (Single Col): Invalid weights from target col '{col_to_make_missing}'. Fallback MCAR.")
            return simulate_missingness_single_col(df, col_to_make_missing, miss_prop, seed, "MCAR", mar_strength=mar_strength, nmar_strength=nmar_strength)
        probabilities = weights / np.sum(weights)

        if np.isnan(probabilities).any() or len(probabilities) != len(eligible_indices):
            logger.warning(f"NMAR Warning (Single Col): Probabilities issue. Fallback MCAR.")
            return simulate_missingness_single_col(df, col_to_make_missing, miss_prop, seed, "MCAR", mar_strength=mar_strength, nmar_strength=nmar_strength)
        try:
            indices_to_nan_list = rng.choice(eligible_indices, size=n_to_make_missing, replace=False, p=probabilities)
        except ValueError as e:
            logger.error(f"NMAR Error (Single Col) during weighted choice: {e}. Fallback MCAR.")
            return simulate_missingness_single_col(df, col_to_make_missing, miss_prop, seed, "MCAR", mar_strength=mar_strength, nmar_strength=nmar_strength)
    
    else:
        logger.error(f"Unknown mechanism: {mechanism}. Defaulting to MCAR for column {col_to_make_missing}.")
        return simulate_missingness_single_col(df, col_to_make_missing, miss_prop, seed, "MCAR", mar_strength=mar_strength, nmar_strength=nmar_strength)

    if len(indices_to_nan_list) > 0:
        data_sim.loc[indices_to_nan_list, col_to_make_missing] = np.nan
    
    return data_sim


# --- Correlation Matrix Function ---
def corstars_py(df: pd.DataFrame, cols: List[str], method: str = 'pearson', remove_triangle: Optional[str] = 'lower') -> pd.DataFrame:
    numeric_df = df[cols].select_dtypes(include=np.number)
    if numeric_df.empty:
        logger.warning("corstars_py: No numeric columns found for correlation.")
        return pd.DataFrame() 

    corr_matrix = numeric_df.corr(method=method)
    p_matrix = pd.DataFrame(np.nan, index=corr_matrix.index, columns=corr_matrix.columns)

    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2:
                data1 = numeric_df[col1].dropna()
                data2 = numeric_df[col2].dropna()
                common_index = data1.index.intersection(data2.index)
                if len(common_index) >= 3: 
                    try:
                        stat, p_val = pearsonr(data1.loc[common_index], data2.loc[common_index])
                        p_matrix.loc[col1, col2] = p_val
                    except Exception as e_corr:
                        logger.debug(f"corstars_py: Could not compute p-value for {col1} and {col2}: {e_corr}")
                        p_matrix.loc[col1, col2] = np.nan
                else:
                    p_matrix.loc[col1, col2] = np.nan
            else:
                p_matrix.loc[col1, col1] = 0.0 

    stars_matrix = pd.DataFrame('', index=corr_matrix.index, columns=corr_matrix.columns)
    stars_matrix[p_matrix < 0.001] = '***'
    stars_matrix[(p_matrix >= 0.001) & (p_matrix < 0.01)] = '**'
    stars_matrix[(p_matrix >= 0.01) & (p_matrix < 0.05)] = '*'
    
    result_matrix = corr_matrix.round(3).astype(str) + stars_matrix
    for col in result_matrix.columns:
        result_matrix.loc[col, col] = '1' 

    if remove_triangle == 'lower':
        mask = np.triu(np.ones_like(result_matrix, dtype=bool), k=0) 
        result_matrix_masked = result_matrix.where(mask, '')
        for i, col_diag in enumerate(result_matrix.columns):
             result_matrix_masked.iloc[i, i] = '1'

    elif remove_triangle == 'upper':
        mask = np.tril(np.ones_like(result_matrix, dtype=bool), k=0) 
        result_matrix_masked = result_matrix.where(mask, '')
        for i, col_diag in enumerate(result_matrix.columns):
             result_matrix_masked.iloc[i, i] = '1'
    else: 
        result_matrix_masked = result_matrix
        
    return result_matrix_masked

# --- Utility functions for data cleaning and neural networks ---
def clean_coef_name_for_html(name_raw): # Renamed for clarity, used by Excel table too
    name_str = str(name_raw)
    name_str = re.sub(r'C\((.*?)\)\[T\.(.*?)\]', r'\1[\2]', name_str) 
    name_str = re.sub(r'C\((.*?)\)', r'\1', name_str) 
    name_str = re.sub(r':C\((.*?)\)\[T\.(.*?)\]', r':\1[\2]', name_str) 
    name_str = re.sub(r'\[T\.(\d+)\.0\]', r'[\1]', name_str) 
    name_str = re.sub(r'\[T\.True\]', r'[1]', name_str)
    name_str = re.sub(r'\[T\.False\]', r'[0]', name_str)
    return name_str.strip()

def clean_coef_name_comp(name_raw_comp): 
    return clean_coef_name_for_html(name_raw_comp)

def create_mlp(input_dim):
    m = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,), kernel_initializer='glorot_uniform'), 
        Dropout(0.1), 
        Dense(16, activation='relu', kernel_initializer='glorot_uniform'), 
        Dense(1, kernel_initializer='glorot_uniform')
    ])
    m.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
    return m

def clean_formula_vars(formula: str) -> List[str]:
    formula_plain = re.sub(r'[A-Za-z0-9_]+\((.*?)\)', r'\1', formula) 
    formula_plain = re.sub(r'C\((.*?)\)', r'\1', formula_plain) 
    formula_plain = formula_plain.replace("EntityEffects", "").replace("TimeEffects", "")
    parts = formula_plain.split('~')
    dep_var = parts[0].strip()
    ind_vars_str = parts[1] if len(parts) > 1 else ''
    raw_terms = re.split(r'\s*[\+\-\*]\s*', ind_vars_str) 
    varnames = [dep_var]
    for term in raw_terms:
        if term.strip(): 
            sub_terms = re.split(r'\s*[:]\s*', term.strip())
            for sub_term in sub_terms:
                cleaned_sub_term = sub_term.strip()
                if cleaned_sub_term and cleaned_sub_term != "1": 
                    varnames.append(cleaned_sub_term)
    return list(set(filter(None, varnames))) 


# --- MiniSummary class for RFeolsResult (placeholder for summary structure) ---
class MiniSummary:
    pass

# --- CovTypeContainer and CovClassName classes (for summary compatibility) ---
class CovClassName: 
    pass

class CovTypeContainer:
    def __init__(self, type_name_str): 
        self.type_name_str = type_name_str
    def __str__(self): 
        return self.type_name_str


# --- RFeolsResult Class to store results from R's feols ---
class RFeolsResult:
    def __init__(self, params_df, nobs, rsquared, rsquared_adj, rsquared_within,
                 fixed_effects_cols_used, weights_col_used, cluster_col_used,
                 formula_str, model_key_name_for_config, vcov_matrix):
        
        if 'term' not in params_df.columns:
            raise ValueError("RFeolsResult: 'term' column missing in params_df from R.")
        
        params_df_indexed = params_df.set_index('term')

        self.params = params_df_indexed['estimate']
        self.bse = params_df_indexed['std.error']
        self.pvalues = params_df_indexed['p.value']
        
        self.conf_low = params_df_indexed.get('conf.low', pd.Series(np.nan, index=self.params.index))
        self.conf_high = params_df_indexed.get('conf.high', pd.Series(np.nan, index=self.params.index))
        
        self.nobs = nobs
        self.rsquared = rsquared
        self.rsquared_adj = rsquared_adj
        self.rsquared_within = rsquared_within 

        self.fixed_effects_cols_used = fixed_effects_cols_used
        self.weights_col_used = weights_col_used
        self.cluster_col_used = cluster_col_used
        
        if isinstance(vcov_matrix, np.ndarray):
            self.cov = pd.DataFrame(vcov_matrix, index=self.params.index, columns=self.params.index)
        elif isinstance(vcov_matrix, pd.DataFrame):
            self.cov = vcov_matrix.reindex(index=self.params.index, columns=self.params.index)
        else:
            logger.warning(f"RFeolsResult: vcov_matrix is not ndarray or DataFrame. Type: {type(vcov_matrix)}")
            nan_fill = np.full((len(self.params.index), len(self.params.index)), np.nan)
            self.cov = pd.DataFrame(nan_fill, index=self.params.index, columns=self.params.index)

        self.is_r_feols = True
        self.is_linearmodels = False 
        self.model_key_name_for_config = model_key_name_for_config 
        self.original_formula_str = formula_str 
        
        dep_var = formula_str.split("~")[0].strip()
        fe_names_str = f"Yes ({', '.join(fixed_effects_cols_used)})" if fixed_effects_cols_used else "No"
        
        _summary_info_data = {
            'Model:': ['R feols (fixest)'],
            'Dep. Variable:': [dep_var],
            'Observations:': [str(int(nobs)) if pd.notna(nobs) else 'N/A'],
            'R-squared:': [f"{rsquared:.4f}" if pd.notna(rsquared) else 'N/A'],
            'Adj. R-squared:': [f"{rsquared_adj:.4f}" if pd.notna(rsquared_adj) else 'N/A'],
            'Within R-sq.:': [f"{rsquared_within:.4f}" if pd.notna(rsquared_within) else 'N/A'], 
            'Fixed Effects:': [fe_names_str],
            'Clustering:': [cluster_col_used if cluster_col_used else 'No'],
            'Weights:': [weights_col_used if weights_col_used else 'No'],
        }
        _summary_table0 = pd.DataFrame.from_dict(_summary_info_data, orient='index', columns=['Value'])
        
        t_values = params_df_indexed.get('statistic', self.params / self.bse)

        _summary_table1 = pd.DataFrame({
            'Parameter': self.params.index,
            'Estimate': self.params.values,
            'Std. Err.': self.bse.values,
            't-value': t_values.values,
            'P>|t|': self.pvalues.values,
            '[0.025': self.conf_low.values, 
            '0.975]': self.conf_high.values
        }).set_index('Parameter')

        self.summary = MiniSummary() 
        self.summary.tables = [_summary_table0, _summary_table1] 

        self.entity_effects = bool(fixed_effects_cols_used and len(fixed_effects_cols_used) > 0)
        self.time_effects = False 
        
        self.cov_type_name = f"Clustered ({cluster_col_used})" if cluster_col_used else "Heteroskedastic (HC1 default in fixest without cluster)" 
        self.cov_type = CovTypeContainer(self.cov_type_name)
    
    def __getstate__(self):
        """自定义pickle序列化，排除任何R对象引用"""
        state = self.__dict__.copy()
        # 确保所有数据都是可序列化的
        return state
    
    def __setstate__(self, state):
        """自定义pickle反序列化"""
        self.__dict__.update(state)


    def summary2(self): 
        table_df = pd.DataFrame({
            'Coef.': self.params, 
            'Std.Err.': self.bse, 
            'P>|t|': self.pvalues, 
            '[0.025': self.conf_low, 
            '0.975]': self.conf_high
        })
        
        dep_var_name = self.original_formula_str.split("~")[0].strip()
        nobs_val_s2 = 'N/A'
        if pd.notna(self.nobs):
            try: nobs_val_s2 = str(int(self.nobs))
            except: nobs_val_s2 = str(self.nobs)

        info_data = {
            'Value': [
                "R feols (fixest)", 
                dep_var_name, 
                nobs_val_s2,
                'N/A' 
            ]
        }
        info_index = ['Imputation Method', 'Dep. Variable', 'No. Observations', 'Df Residuals']
        table_info_df = pd.DataFrame(info_data, index=info_index)
        
        SummaryContainer = collections.namedtuple("SummaryContainer", ["tables"])
        return SummaryContainer(tables=[table_info_df, table_df])


# --- MODIFIED safe_run_regression (to use R feols for Santamaria model) ---
def safe_run_regression(
    formula: str, 
    data: pd.DataFrame,
    model_key: str, 
    family: Optional[Any] = None,
) -> Optional[Any]:
    original_formula = formula 
    data_for_reg = data.copy()

    fixed_effects_cols = Config.MODEL_FIXED_EFFECTS.get(model_key)
    cluster_col = Config.MODEL_CLUSTER_SE.get(model_key)
    weights_col = Config.MODEL_WEIGHTS_COL.get(model_key)
    
    all_vars_needed_for_check = clean_formula_vars(original_formula) 
    if weights_col and weights_col not in all_vars_needed_for_check: all_vars_needed_for_check.append(weights_col)
    if fixed_effects_cols: all_vars_needed_for_check.extend(fe_col for fe_col in fixed_effects_cols if fe_col not in all_vars_needed_for_check)
    if cluster_col and cluster_col not in all_vars_needed_for_check: all_vars_needed_for_check.append(cluster_col)
    
    missing_vars_in_data = [v for v in all_vars_needed_for_check if v not in data_for_reg.columns]
    if missing_vars_in_data: 
        logger.warning(f"Reg Skip ({model_key}): Formula '{original_formula}' - Vars not in data: {missing_vars_in_data}")
        return None

    if fixed_effects_cols:
        for fe_c in fixed_effects_cols:
            if fe_c not in data_for_reg.columns:
                logger.error(f"Reg Error ({model_key}): FE column '{fe_c}' not found in data. Cannot proceed with panel model.")
                return None
    
    if cluster_col and cluster_col not in data_for_reg.columns:
        logger.warning(f"Reg Warning ({model_key}): Cluster column '{cluster_col}' not found. Proceeding without clustering if R/Python model allows.")
        cluster_col_exists = False
    else:
        cluster_col_exists = True if cluster_col else False

    if weights_col and weights_col not in data_for_reg.columns:
        logger.warning(f"Reg Warning ({model_key}): Weights column '{weights_col}' not found. Proceeding unweighted for this model.")
        weights_col = None

    is_panel_candidate = Config.MODEL_USE_PANEL_ESTIMATOR.get(model_key, False)

    if R_OK and is_panel_candidate: 
        logger.info(f"Attempting R feols (fixest) for {model_key}...")
        
        py_formula_parts = original_formula.split('~')
        dep_var_r = py_formula_parts[0].strip()
        indep_vars_r_str = py_formula_parts[1].strip()
        
        r_formula_str_for_feols = f"{dep_var_r} ~ {indep_vars_r_str}"
        actual_fixed_effects_cols_used_r = []

        if fixed_effects_cols:
            valid_fe_cols_r = [fe for fe in fixed_effects_cols if fe in data_for_reg.columns]
            if valid_fe_cols_r:
                r_fe_str_part = " + ".join(valid_fe_cols_r)
                r_formula_str_for_feols += f" | {r_fe_str_part}"
                actual_fixed_effects_cols_used_r = valid_fe_cols_r
            else:
                logger.warning(f"R feols ({model_key}): Specified FE cols {fixed_effects_cols} not found or all invalid. Running without FEs in formula.")
        
        if data_for_reg[dep_var_r].isnull().all(): 
            logger.error(f"R feols Error ({model_key}): Dependent variable '{dep_var_r}' is all NA."); return None
        for fe_c in actual_fixed_effects_cols_used_r:
             if data_for_reg[fe_c].isnull().all():
                logger.error(f"R feols Error ({model_key}): FE column '{fe_c}' is all NA."); return None

        try:
            r_data_for_feols = data_for_reg.copy()
            vars_to_check_bool = clean_formula_vars(r_formula_str_for_feols) 
            if cluster_col and cluster_col not in vars_to_check_bool: vars_to_check_bool.append(cluster_col)
            if weights_col and weights_col not in vars_to_check_bool: vars_to_check_bool.append(weights_col)

            for col_r_check in vars_to_check_bool:
                if col_r_check in r_data_for_feols.columns:
                    if pd.api.types.is_bool_dtype(r_data_for_feols[col_r_check]):
                        r_data_for_feols[col_r_check] = r_data_for_feols[col_r_check].astype(int)
                    if col_r_check in actual_fixed_effects_cols_used_r:
                        if pd.api.types.is_float_dtype(r_data_for_feols[col_r_check]) and r_data_for_feols[col_r_check].apply(lambda x: pd.isna(x) or x.is_integer()).all():
                             r_data_for_feols[col_r_check] = r_data_for_feols[col_r_check].astype('Int64') 

            
            r_data_name = "current_r_df_for_feols"
            with localconverter(robjects.default_converter + pandas2ri.converter + numpy2ri.converter):
                robjects.globalenv[r_data_name] = robjects.conversion.py2rpy(r_data_for_feols)
            
            feols_call_args = {"fml": robjects.Formula(r_formula_str_for_feols), "data": robjects.globalenv[r_data_name]}
            
            current_weights_col_r = None
            if weights_col and weights_col in data_for_reg.columns and data_for_reg[weights_col].notna().any():
                feols_call_args["weights"] = data_for_reg[weights_col].values 
                current_weights_col_r = weights_col
            elif weights_col: 
                logger.warning(f"R feols ({model_key}): Weights '{weights_col}' all NA or not found in data. Unweighted.")
            
            actual_cluster_col_used_r = None
            if cluster_col and cluster_col_exists and cluster_col in data_for_reg.columns and data_for_reg[cluster_col].notna().any():
                feols_call_args["cluster"] = robjects.Formula(f"~{cluster_col}")
                actual_cluster_col_used_r = cluster_col
            elif cluster_col: 
                logger.warning(f"R feols ({model_key}): Cluster '{cluster_col}' all NA or not found. Default SEs (HC1 if no FEs, Clustered by FE if FEs).")

            logger.info(f"R feols Call Args ({model_key}): formula='{r_formula_str_for_feols}', cluster='{actual_cluster_col_used_r}', weights='{current_weights_col_r}'")
            
            r_model_obj = fixest_r.feols(**feols_call_args)
            robjects.globalenv['current_r_model_obj_feols'] = r_model_obj 

            # 使用rpy2正确的方式调用R函数获取模型结果
            try:
                # 获取系数 - 使用robjects.r()调用R的coef函数
                coef_r = robjects.r('coef')(r_model_obj)
                # 获取标准误 - 使用fixest包的se函数
                se_r = robjects.r('se')(r_model_obj)
                # 获取p值 - 使用fixest包的pvalue函数
                pval_r = robjects.r('pvalue')(r_model_obj)
                
                # 获取变量名
                var_names_r = robjects.r('names')(coef_r)
                
                # 转换为Python对象
                with localconverter(robjects.default_converter + pandas2ri.converter + numpy2ri.converter):
                    coef_py = robjects.conversion.rpy2py(coef_r)
                    se_py = robjects.conversion.rpy2py(se_r)
                    pval_py = robjects.conversion.rpy2py(pval_r)
                    var_names_py = robjects.conversion.rpy2py(var_names_r)
                
                # 尝试获取置信区间，如果失败则使用NA
                try:
                    ci_r = robjects.r('confint')(r_model_obj)
                    ci_py = robjects.conversion.rpy2py(ci_r)
                    
                    # 检查置信区间的形状
                    if hasattr(ci_py, 'shape') and len(ci_py.shape) == 2:
                        conf_low = ci_py[:, 0]
                        conf_high = ci_py[:, 1]
                    else:
                        # 如果形状不对，创建NA数组
                        n_coefs = len(coef_py)
                        conf_low = np.full(n_coefs, np.nan)
                        conf_high = np.full(n_coefs, np.nan)
                except Exception as e_ci:
                    logger.warning(f"R feols ({model_key}): Could not extract confidence intervals: {e_ci}")
                    n_coefs = len(coef_py)
                    conf_low = np.full(n_coefs, np.nan)
                    conf_high = np.full(n_coefs, np.nan)
                
                # 创建类似broom::tidy输出的DataFrame
                tidy_py_df = pd.DataFrame({
                    'term': var_names_py,
                    'estimate': coef_py,
                    'std.error': se_py,
                    'p.value': pval_py,
                    'conf.low': conf_low,
                    'conf.high': conf_high
                })
                
            except Exception as e_coef:
                logger.error(f"R feols Error ({model_key}): Failed to extract coefficients: {e_coef}")
                return None
            
            if tidy_py_df.empty or 'estimate' not in tidy_py_df.columns: 
                logger.error(f"R feols Error ({model_key}): Failed to create coefficient DataFrame."); return None
            
            nobs_r = robjects.r('nobs')(r_model_obj)
            nobs_val = int(nobs_r[0]) if nobs_r else np.nan
            
            try: 
                rsq_val = robjects.r('r2')(r_model_obj, type="r2")[0] 
                rsq_adj_val = robjects.r('r2')(r_model_obj, type="ar2")[0] 
                rsq_within_val = np.nan
                if actual_fixed_effects_cols_used_r: 
                    rsq_within_val = robjects.r('r2')(r_model_obj, type="wr2")[0] 
            except Exception as e_rsq: 
                logger.warning(f"R feols ({model_key}): Could not retrieve R-sq values: {e_rsq}.")
                rsq_val, rsq_adj_val, rsq_within_val = np.nan, np.nan, np.nan
            
            vcov_r_matrix = stats_r.vcov(r_model_obj) 
            vcov_py_matrix = np.asarray(vcov_r_matrix)
            
            rsq_str = f"{rsq_val:.3f}" if pd.notna(rsq_val) else "NA"
            logger.info(f"R feols ({model_key}): Success. Nobs={nobs_val}, R2={rsq_str}")
            
            return RFeolsResult(params_df=tidy_py_df, nobs=nobs_val, 
                                rsquared=rsq_val, rsquared_adj=rsq_adj_val, rsquared_within=rsq_within_val,
                                fixed_effects_cols_used=actual_fixed_effects_cols_used_r, 
                                weights_col_used=current_weights_col_r, 
                                cluster_col_used=actual_cluster_col_used_r, 
                                formula_str=original_formula, 
                                model_key_name_for_config=model_key, 
                                vcov_matrix=vcov_py_matrix)
        except Exception as e_r_feols:
            logger.error(f"R feols Error ({model_key}) for '{r_formula_str_for_feols}': {e_r_feols}", exc_info=True)
            return None
        finally:
            if 'current_r_df_for_feols' in robjects.globalenv: robjects.r("rm(current_r_df_for_feols)")
            if 'current_r_model_obj_feols' in robjects.globalenv: robjects.r("rm(current_r_model_obj_feols)")
    
    elif is_panel_candidate and not R_OK:
        logger.warning(f"R feols configured for {model_key}, but rpy2 is not available. Attempting Python fallback...")
        # Fall through to Python statsmodels implementation
    elif is_panel_candidate and R_OK: 
        logger.warning(f"R feols path for {model_key} was not taken despite R_OK and panel_candidate. This is unexpected.")
        return None
        
    # Python fallback (either not panel candidate or R not available)
    if is_panel_candidate:
        logger.info(f"Attempting Python PanelOLS for {model_key} (R feols not available)...")
        # Try PanelOLS from linearmodels for panel data
        try:
            from linearmodels.panel import PanelOLS
            from linearmodels.panel.results import PanelEffectsResults
            
            # Prepare data for PanelOLS
            dep_var_name_panel = original_formula.split('~')[0].strip()
            if dep_var_name_panel not in data_for_reg.columns or data_for_reg[dep_var_name_panel].isnull().all():
                logger.error(f"PanelOLS Error ({model_key}): DV '{dep_var_name_panel}' not in data or all NA."); 
                return None
            
            # Create entity and time indices for PanelOLS
            entity_col = Config.ID_COLUMN_ORIGINAL  # SN
            time_col = Config.ID_COLUMN_TIME        # Post
            
            if entity_col not in data_for_reg.columns or time_col not in data_for_reg.columns:
                logger.error(f"PanelOLS Error ({model_key}): Required panel indices {entity_col}, {time_col} not found.")
                return None
            
            # Prepare formula for PanelOLS (remove entity effects from formula)
            indep_vars_panel = original_formula.split('~')[1].strip()
            
            # Create PanelOLS model
            panel_data = data_for_reg.set_index([entity_col, time_col])
            y_panel = panel_data[dep_var_name_panel]
            
            # Handle interaction terms and other variables
            if ' * ' in indep_vars_panel:
                # For interaction terms, we need to create them manually
                vars_list = []
                for var_part in indep_vars_panel.split(' * '):
                    var_part = var_part.strip()
                    if var_part in panel_data.columns:
                        vars_list.append(panel_data[var_part])
                    else:
                        logger.warning(f"PanelOLS Warning ({model_key}): Variable {var_part} not found in panel data")
                
                if len(vars_list) == 2:
                    # Create interaction term
                    interaction_term = vars_list[0] * vars_list[1]
                    interaction_term.name = f"{vars_list[0].name}_x_{vars_list[1].name}"
                    X_panel = panel_data[list(set([v.name for v in vars_list]))].copy()
                    X_panel[interaction_term.name] = interaction_term
                else:
                    X_panel = panel_data[[v.name for v in vars_list if v.name in panel_data.columns]]
            else:
                # Simple case - no interactions
                var_names = [v.strip() for v in indep_vars_panel.split('+')]
                available_vars = [v for v in var_names if v in panel_data.columns]
                if not available_vars:
                    logger.error(f"PanelOLS Error ({model_key}): No independent variables found in panel data")
                    return None
                X_panel = panel_data[available_vars]
            
            # Run PanelOLS
            model_panel = PanelOLS(y_panel, X_panel, entity_effects=True)
            results_panel = model_panel.fit(cov_type='clustered', cluster_entity=True)
            
            # Convert to compatible format
            params_panel = results_panel.params
            bse_panel = results_panel.std_errors
            pvalues_panel = results_panel.pvalues
            
            # Create a mock results object compatible with the rest of the code
            class PanelOLSResult:
                def __init__(self, params, bse, pvalues, nobs, rsquared, model_formula, method):
                    self.params = params
                    self.bse = bse
                    self.pvalues = pvalues
                    self.nobs = nobs
                    self.rsquared = rsquared
                    self.model_formula = model_formula
                    self.method = method
                    self.fittedvalues = None
                    self.resid = None
                    self.df_resid = nobs - len(params) if nobs else None
                
                def summary2(self):
                    # Create a simple summary
                    summary_data = {
                        'Coef.': self.params,
                        'Std.Err.': self.bse,
                        'P>|t|': self.pvalues
                    }
                    return pd.DataFrame(summary_data)
            
            panel_result = PanelOLSResult(
                params=params_panel,
                bse=bse_panel,
                pvalues=pvalues_panel,
                nobs=len(y_panel),
                rsquared=results_panel.rsquared,
                model_formula=original_formula,
                method="PanelOLS (Python fallback)"
            )
            
            logger.info(f"PanelOLS Success ({model_key}): Nobs={len(y_panel)}, R2={results_panel.rsquared:.3f}")
            return panel_result
            
        except Exception as e_panel:
            logger.error(f"PanelOLS Error ({model_key}): {e_panel}")
            logger.info(f"Falling back to standard statsmodels for {model_key}...")
    
    logger.info(f"Attempting statsmodels for {model_key}...")
    actual_fixed_effects_cols_used_sm = []
    current_formula_sm = original_formula 
    dep_var_name_sm_check = current_formula_sm.split('~')[0].strip()
    if dep_var_name_sm_check not in data_for_reg.columns or data_for_reg[dep_var_name_sm_check].isnull().all():
        logger.error(f"Statsmodels Reg Error ({model_key}): DV '{dep_var_name_sm_check}' not in data or all NA."); return None

    if fixed_effects_cols:
        fe_terms_to_add_sm = []
        for fe_col_sm in fixed_effects_cols:
            if fe_col_sm not in data_for_reg.columns: 
                logger.error(f"Statsmodels Reg Error ({model_key}): FE col '{fe_col_sm}' not found for C() term."); continue
            fe_terms_to_add_sm.append(f"C({fe_col_sm})")
            actual_fixed_effects_cols_used_sm.append(fe_col_sm)
        
        if fe_terms_to_add_sm: 
            current_formula_sm = f"{current_formula_sm} + {' + '.join(fe_terms_to_add_sm)}"
            logger.info(f"Statsmodels Reg Info ({model_key}): Added FEs {actual_fixed_effects_cols_used_sm} as C() terms. Formula: {current_formula_sm}")

    cat_vars_in_formula_sm = re.findall(r'C\((.*?)\)', current_formula_sm)
    for v_cat_raw_sm in cat_vars_in_formula_sm:
        v_cat_clean_sm = v_cat_raw_sm.strip()
        if v_cat_clean_sm in data_for_reg.columns and v_cat_clean_sm not in actual_fixed_effects_cols_used_sm: 
            if not isinstance(data_for_reg[v_cat_clean_sm].dtype, CategoricalDtype) and \
               data_for_reg[v_cat_clean_sm].dtype != 'object' and \
               data_for_reg[v_cat_clean_sm].dtype.name != 'category':
                try:
                    if pd.api.types.is_numeric_dtype(data_for_reg[v_cat_clean_sm]) and data_for_reg[v_cat_clean_sm].nunique() > 20: 
                         logger.warning(f"Statsmodels Reg Warn ({model_key}): Wrapping potentially continuous numeric var '{v_cat_clean_sm}' with C(). Check if intended.")
                except Exception as e_cat_convert:
                     logger.warning(f"Statsmodels Reg Warn ({model_key}): Cannot convert '{v_cat_clean_sm}' to category for C(): {e_cat_convert}")
    
    try: 
        vars_for_sm_na_check = clean_formula_vars(current_formula_sm) 
        current_weights_col_sm = None
        if weights_col and weights_col in data_for_reg.columns and data_for_reg[weights_col].notna().any():
            vars_for_sm_na_check.append(weights_col); current_weights_col_sm = weights_col
        elif weights_col: logger.warning(f"Statsmodels ({model_key}): Weights '{weights_col}' all NA or not found. Unweighted.")
        
        actual_cluster_col_used_sm = None
        if cluster_col and cluster_col_exists and cluster_col in data_for_reg.columns and data_for_reg[cluster_col].notna().any():
            vars_for_sm_na_check.append(cluster_col); actual_cluster_col_used_sm = cluster_col
        elif cluster_col: logger.warning(f"Statsmodels ({model_key}): Cluster '{cluster_col}' all NA or not found. Unclustered SEs.")

        cols_present_for_sm_na_check = [c for c in vars_for_sm_na_check if c in data_for_reg.columns]
        data_for_obs_check_sm = data_for_reg.dropna(subset=cols_present_for_sm_na_check, how='any')
        n_obs_sm = data_for_obs_check_sm.shape[0]

        if n_obs_sm == 0: 
            logger.warning(f"Statsmodels Reg Skip ({model_key}): No complete cases for '{current_formula_sm}'. Vars checked: {cols_present_for_sm_na_check}"); return None
        
        try:
            y_design, X_design = dmatrices(current_formula_sm, data=data_for_obs_check_sm, return_type='dataframe', NA_action='drop') 
        except Exception as e_dmat:
            logger.error(f"Statsmodels Reg Error ({model_key}): dmatrices failed for '{current_formula_sm}' after NA drop: {e_dmat}. Data shape: {data_for_obs_check_sm.shape}", exc_info=True)
            return None

        n_terms_approx_sm = X_design.shape[1]

        if n_obs_sm <= n_terms_approx_sm: 
             logger.warning(f"Statsmodels Reg Skip ({model_key}): Insufficient observations ({n_obs_sm}) for {n_terms_approx_sm} terms (or n_obs <= n_features). Formula: {current_formula_sm}"); return None

        dep_var_name_sm = current_formula_sm.split('~')[0].strip()
        if dep_var_name_sm in data_for_obs_check_sm.columns:
            if data_for_obs_check_sm[dep_var_name_sm].nunique() < 1: logger.warning(f"Statsmodels Reg Skip ({model_key}): DV '{dep_var_name_sm}' zero variance."); return None
            if data_for_obs_check_sm[dep_var_name_sm].nunique() < 2 and family is None: logger.warning(f"Statsmodels Reg Skip ({model_key}): DV '{dep_var_name_sm}' < 2 unique values for OLS."); return None
        
        logger.info(f"Statsmodels Reg Attempt ({model_key}): Formula '{current_formula_sm}', N_obs={n_obs_sm}, N_params_approx={n_terms_approx_sm}")
    except Exception as e_prep_sm: 
        logger.error(f"Statsmodels Reg Error ({model_key}): Prep failed for '{current_formula_sm}': {e_prep_sm}", exc_info=True); return None
        
        try: 
            model_fitting_weights_sm_series = data_for_reg[current_weights_col_sm] if current_weights_col_sm else None
            
            if family: model_sm = smf.glm(formula=current_formula_sm, data=data_for_reg, family=family, missing='drop')
            else: model_sm = smf.ols(formula=current_formula_sm, data=data_for_reg, missing='drop')
            
            results_sm = None
            if actual_cluster_col_used_sm:
                try:
                    cluster_groups_for_fit_sm = data_for_reg[actual_cluster_col_used_sm]
                    results_sm = model_sm.fit(cov_type='cluster', 
                                              cov_kwds={'groups': cluster_groups_for_fit_sm, 'debiased': True, 'use_correction': True}, 
                                              weights=model_fitting_weights_sm_series)
                    logger.info(f"Statsmodels Reg ({model_key}): Clustered SE by '{actual_cluster_col_used_sm}'.")
                except Exception as e_cluster_sm: 
                    logger.error(f"Statsmodels Reg ({model_key}): Error with clustered SEs: {e_cluster_sm}. Fitting unclustered.", exc_info=False) 
                    results_sm = model_sm.fit(weights=model_fitting_weights_sm_series)
                    actual_cluster_col_used_sm = None 
            else: 
                results_sm = model_sm.fit(weights=model_fitting_weights_sm_series)

            if hasattr(results_sm, 'mle_retvals') and not results_sm.mle_retvals.get('converged', True): 
                logger.warning(f"Statsmodels Reg Warn ({model_key}): Convergence issue for '{current_formula_sm}'.")
            
            results_sm.is_linearmodels = False; results_sm.is_r_feols = False
            results_sm.fixed_effects_cols_used = actual_fixed_effects_cols_used_sm 
            results_sm.model_key_name_for_config = model_key
            results_sm.weights_col_used = current_weights_col_sm
            results_sm.cluster_col_used = actual_cluster_col_used_sm
            results_sm.original_formula_str = original_formula 
            logger.info(f"Statsmodels Reg ({model_key}): Successfully fitted model.")
            return results_sm
        except PerfectSeparationError: logger.error(f"Statsmodels Reg Error ({model_key}): Perfect separation for '{current_formula_sm}'.")
        except np.linalg.LinAlgError as e_linalg_sm: logger.error(f"Statsmodels Reg Error ({model_key}): LinAlgError for '{current_formula_sm}': {e_linalg_sm}")
        except Exception as e_sm: logger.error(f"Statsmodels Reg Error ({model_key}) for '{current_formula_sm}': {e_sm}", exc_info=True)
        return None


# --- PooledRegressionResults Class (Unchanged) ---
class PooledRegressionResults:
    def __init__(self, params, bse, pvalues, nobs, df_resid, model_formula, method, 
                 fixed_effects_cols_used=None, weights_col_used=None, cluster_col_used=None, 
                 is_linearmodels=False, is_r_feols=False): 
        self.params = params; self.bse = bse; self.pvalues = pvalues
        self.nobs = nobs; self.df_resid = df_resid
        self.model_formula = model_formula; self.method = method
        self.fixed_effects_cols_used = fixed_effects_cols_used 
        self.weights_col_used = weights_col_used
        self.cluster_col_used = cluster_col_used 
        self.is_linearmodels = is_linearmodels 
        self.is_r_feols = is_r_feols 
        self.rsquared = np.nan; self.rsquared_adj = np.nan 
        self.entity_effects = False; self.time_effects = False 
        self.original_formula_str = model_formula 

        self.cov_type = CovTypeContainer(f"Clustered ({self.cluster_col_used})" if self.cluster_col_used else "Rubin's Rules (Unadjusted U_bar)")

    def summary2(self) -> collections.namedtuple: 
        table_df = pd.DataFrame(columns=['Coef.', 'Std.Err.', 'P>|t|', '[0.025', '0.975]'])
        try:
            _params = self.params if isinstance(self.params, (pd.Series, dict)) else pd.Series(self.params if self.params is not None else {})
            _bse = self.bse if isinstance(self.bse, (pd.Series, dict)) else pd.Series(self.bse if self.bse is not None else {})
            _pvalues = self.pvalues if isinstance(self.pvalues, (pd.Series, dict)) else pd.Series(self.pvalues if self.pvalues is not None else {})
            all_indices = _params.index.union(_bse.index).union(_pvalues.index)
            _params = _params.reindex(all_indices); _bse = _bse.reindex(all_indices); _pvalues = _pvalues.reindex(all_indices)
            current_table_df = pd.DataFrame({'Coef.': _params, 'Std.Err.': _bse, 'P>|t|': _pvalues})
            valid_df_resid = self.df_resid is not None and np.isfinite(self.df_resid) and self.df_resid > 0
            if valid_df_resid and isinstance(_bse, pd.Series) and not _bse.empty:
                _params_series = _params if isinstance(_params, pd.Series) else pd.Series(_params)
                try:
                    alpha = 0.05
                    if np.isscalar(self.df_resid) and self.df_resid > 0:
                        t_critical_value = t_dist.ppf(1 - alpha / 2, self.df_resid)
                        current_table_df['[0.025'] = _params_series - t_critical_value * _bse
                        current_table_df['0.975]'] = _params_series + t_critical_value * _bse
                    else: 
                        norm_critical_value = norm.ppf(1 - alpha / 2)
                        current_table_df['[0.025'] = _params_series - norm_critical_value * _bse
                        current_table_df['0.975]'] = _params_series + norm_critical_value * _bse
                except Exception as e: 
                    logger.warning(f"PooledResults.summary2: Error calculating CIs: {e}")
                    current_table_df['[0.025'] = np.nan; current_table_df['0.975]'] = np.nan
            else:
                current_table_df['[0.025'] = np.nan; current_table_df['0.975]'] = np.nan
            if not current_table_df.empty: table_df = current_table_df
        except Exception as e:
            logger.error(f"PooledResults.summary2: Failed to construct coefficient table_df: {e}", exc_info=True)
        dep_var_name = self.model_formula.split("~")[0].strip() if self.model_formula and "~" in self.model_formula else "N/A"
        nobs_val = str(int(self.nobs)) if pd.notna(self.nobs) else 'N/A'
        df_resid_val = str(round(float(self.df_resid), 2)) if pd.notna(self.df_resid) else 'N/A'
        info_data_list = [str(self.method if self.method is not None else "N/A"), dep_var_name, nobs_val, df_resid_val]
        info_index = ['Imputation Method', 'Dep. Variable', 'No. Observations', 'Df Residuals']
        try: table_info_df = pd.DataFrame({'Value': info_data_list}, index=info_index)
        except Exception as e: logger.error(f"PooledResults.summary2: Failed to construct table_info_df: {e}", exc_info=True); table_info_df = pd.DataFrame(columns=['Value'])
        SummaryContainer = collections.namedtuple("SummaryContainer", ["tables"])
        if not isinstance(table_df, pd.DataFrame): table_df = pd.DataFrame(columns=['Coef.', 'Std.Err.', 'P>|t|', '[0.025', '0.975]'])
        if not isinstance(table_info_df, pd.DataFrame): table_info_df = pd.DataFrame(columns=['Value'])
        return SummaryContainer(tables=[table_info_df, table_df])

# --- MODIFIED run_pooled_regression (Unchanged) ---
def run_pooled_regression(
    imputed_datasets: List[pd.DataFrame],
    formula: str, model_key: str,
    family: Optional[Any] = None, baseline_nobs: Optional[int] = None
) -> Optional[PooledRegressionResults]:
    if not imputed_datasets or len(imputed_datasets) == 0: logger.error("Pooled Reg Error: No imputed datasets."); return None
    num_imputations_M = len(imputed_datasets); params_list, cov_matrices_list, nobs_list = [], [], []; valid_model_fits_count = 0
    
    pooled_fixed_effects_cols = Config.MODEL_FIXED_EFFECTS.get(model_key)
    pooled_cluster_col = Config.MODEL_CLUSTER_SE.get(model_key)
    pooled_weights_col = Config.MODEL_WEIGHTS_COL.get(model_key)
    
    actual_fes_used_pooled = pooled_fixed_effects_cols 
    actual_weights_col_used_pooled = None 
    actual_cluster_col_used_pooled = None 

    model_was_linearmodels_pooled = False; model_was_r_feols_pooled = False

    for i, data_m_imputed in enumerate(imputed_datasets):
        if not isinstance(data_m_imputed, pd.DataFrame): logger.warning(f"Pooled Reg: Dataset {i} not DataFrame."); continue
        df_for_this_regression = data_m_imputed.copy()
        
        if df_for_this_regression.index.name == Config.ID_COLUMN:
            if Config.ID_COLUMN not in df_for_this_regression.columns: 
                 df_for_this_regression = df_for_this_regression.reset_index()
        
        for id_c in [Config.ID_COLUMN_ORIGINAL, Config.ID_COLUMN_TIME]:
            if id_c not in df_for_this_regression.columns and id_c in data_m_imputed.columns:
                df_for_this_regression[id_c] = data_m_imputed[id_c]
            elif id_c not in df_for_this_regression.columns:
                 logger.warning(f"Pooled Reg: Original ID col '{id_c}' missing in imputed dataset {i} for regression.")


        result_m_fit = safe_run_regression(formula, df_for_this_regression, model_key, family) 
        
        if result_m_fit and hasattr(result_m_fit, 'params'):
            cov_m = None
            is_lm_model_iter = getattr(result_m_fit, 'is_linearmodels', False)
            is_r_feols_model_iter = getattr(result_m_fit, 'is_r_feols', False)

            if is_r_feols_model_iter: 
                if hasattr(result_m_fit, 'cov') and isinstance(result_m_fit.cov, pd.DataFrame): cov_m = result_m_fit.cov
                else: logger.warning(f"Pooled Reg (R feols): Fit {i} missing .cov or not DF.")
                model_was_r_feols_pooled = True
            elif is_lm_model_iter:  
                if hasattr(result_m_fit, 'cov'): cov_m = result_m_fit.cov
                else: logger.warning(f"Pooled Reg (PanelOLS): Fit {i} missing .cov.")
                model_was_linearmodels_pooled = True  
            else: 
                if hasattr(result_m_fit, 'cov_params') and callable(result_m_fit.cov_params): cov_m = result_m_fit.cov_params()
                else: logger.warning(f"Pooled Reg (statsmodels): Fit {i} missing cov_params().")

            if cov_m is not None and isinstance(cov_m, pd.DataFrame) and not cov_m.empty:
                common_idx = result_m_fit.params.index.intersection(cov_m.index).intersection(cov_m.columns)
                if not result_m_fit.params.index.equals(common_idx) or len(common_idx) < len(result_m_fit.params):
                    logger.warning(f"Pooled Reg: Data {i}, formula '{formula}', param/cov_m index mismatch or reduction. Aligning to common: {common_idx.tolist()}")
                
                if not common_idx.empty:
                    params_list.append(result_m_fit.params.loc[common_idx])
                    cov_matrices_list.append(cov_m.loc[common_idx, common_idx])
                    nobs_list.append(result_m_fit.nobs if hasattr(result_m_fit, 'nobs') else np.nan)
                    valid_model_fits_count += 1
                    if valid_model_fits_count == 1: 
                        actual_fes_used_pooled = getattr(result_m_fit, 'fixed_effects_cols_used', pooled_fixed_effects_cols)
                        actual_weights_col_used_pooled = getattr(result_m_fit, 'weights_col_used', None)
                        actual_cluster_col_used_pooled = getattr(result_m_fit, 'cluster_col_used', None)
                else:
                    logger.warning(f"Pooled Reg: Data {i}, formula '{formula}', no common indices after alignment. Skipping.")
            else: 
                logger.warning(f"Pooled Reg: Data {i}, formula '{formula}', cov matrix invalid/empty for model type. Skipping. Params were: {result_m_fit.params.index.tolist() if result_m_fit.params is not None else 'None'}")
        else: 
            logger.warning(f"Pooled Reg: Fit failed for data {i}, formula '{formula}'.")

    if not params_list or not cov_matrices_list or valid_model_fits_count == 0 or len(params_list) != len(cov_matrices_list): 
        logger.error(f"Pooled Reg Error: Insufficient valid fits for '{formula}' (Valid fits: {valid_model_fits_count})"); return None
    
    M = valid_model_fits_count
    if M < num_imputations_M : logger.warning(f"Pooled Reg Warning: Only {M}/{num_imputations_M} models valid for '{formula}'.")
    if M == 0 : return None 

    try:
        common_param_names = params_list[0].index
        for p_s_iter in params_list[1:]:
            common_param_names = common_param_names.intersection(p_s_iter.index)
        
        if common_param_names.empty:
            logger.error(f"Pooled Reg Error: No common parameter names across imputations for '{formula}'."); return None
        
        logger.info(f"Pooled Reg: Common parameters for pooling for '{formula}': {common_param_names.tolist()}")

        aligned_params_list_series = [p_s.loc[common_param_names] for p_s in params_list]
        aligned_cov_matrices_list = [cov_m.loc[common_param_names, common_param_names] for cov_m in cov_matrices_list]

        q_bar_series_pooled = pd.concat(aligned_params_list_series, axis=1).mean(axis=1)
        q_bar_values_pooled = q_bar_series_pooled.values 
        
        u_bar_diag_variances = np.mean([np.diag(np.asarray(cov_m_aligned)) for cov_m_aligned in aligned_cov_matrices_list], axis=0)
        
        b_diag_variances = np.zeros_like(q_bar_values_pooled, dtype=float)
        if M > 1:
            param_estimates_array = np.array([p_s_aligned.values for p_s_aligned in aligned_params_list_series])
            b_diag_variances = np.var(param_estimates_array, axis=0, ddof=1) 

        t_diag_total_variances = u_bar_diag_variances + (1 + 1/M) * b_diag_variances
        pooled_std_errors_values = np.sqrt(np.maximum(0, t_diag_total_variances))
        
        df_pooled_final_scalar = -1.0
        if M > 1:
            riv_numerator_for_df = (1 + 1/M) * b_diag_variances
            riv_for_df = np.full_like(b_diag_variances, np.inf, dtype=float)
            mask_u_bar_nonzero = u_bar_diag_variances > 1e-12
            if np.any(mask_u_bar_nonzero): riv_for_df[mask_u_bar_nonzero] = riv_numerator_for_df[mask_u_bar_nonzero] / u_bar_diag_variances[mask_u_bar_nonzero]
            riv_for_df[~mask_u_bar_nonzero & (b_diag_variances > 1e-12)] = np.inf 
            riv_for_df[~mask_u_bar_nonzero & (b_diag_variances <= 1e-12)] = 0
            
            df_m_barnard_rubin_per_param = (M - 1) * (1 + (1 / riv_for_df))**2
            df_m_barnard_rubin_per_param[~np.isfinite(df_m_barnard_rubin_per_param)] = M - 1
            df_m_barnard_rubin_per_param[df_m_barnard_rubin_per_param < 1] = 1
            df_pooled_final_scalar = np.min(df_m_barnard_rubin_per_param) if len(df_m_barnard_rubin_per_param) > 0 else M - 1
            df_pooled_final_scalar = max(1.0, df_pooled_final_scalar)
        else: 
            avg_nobs_single = np.mean(nobs_list) if nobs_list and pd.notna(np.mean(nobs_list)) else (baseline_nobs if baseline_nobs is not None else 0)
            k_params_single = len(q_bar_values_pooled)
            df_pooled_final_scalar = max(1.0, avg_nobs_single - k_params_single if avg_nobs_single > k_params_single else 1.0)

        t_statistics_values = np.zeros_like(q_bar_values_pooled, dtype=float)
        valid_se_mask_for_t = pooled_std_errors_values > 1e-9
        t_statistics_values[valid_se_mask_for_t] = q_bar_values_pooled[valid_se_mask_for_t] / pooled_std_errors_values[valid_se_mask_for_t]
        
        p_values_final = 2 * t_dist.sf(np.abs(t_statistics_values), df=df_pooled_final_scalar)
        avg_nobs_overall = int(np.mean(nobs_list)) if nobs_list and pd.notna(np.mean(nobs_list)) else (baseline_nobs if baseline_nobs is not None else 0)
        
        return PooledRegressionResults(params=q_bar_series_pooled, 
                                     bse=pd.Series(pooled_std_errors_values, index=common_param_names), 
                                     pvalues=pd.Series(p_values_final, index=common_param_names), 
                                     nobs=avg_nobs_overall, 
                                     df_resid=df_pooled_final_scalar, model_formula=formula, method="Custom MI Pool",
                                     fixed_effects_cols_used=actual_fes_used_pooled, 
                                     weights_col_used=actual_weights_col_used_pooled, 
                                     cluster_col_used=actual_cluster_col_used_pooled,
                                     is_linearmodels=model_was_linearmodels_pooled, 
                                     is_r_feols=model_was_r_feols_pooled)
    except Exception as e: 
        logger.error(f"Pooled Reg Error: Combining results for '{formula}': {e}", exc_info=True); return None


# --- get_coef_info_py (for coefficient stability comparison) ---
def get_coef_info_py(model_results_comp, alpha_comp=Config.ALPHA) -> Optional[pd.DataFrame]:
    if model_results_comp is None: return None
    params, bse, pvalues = None, None, None
    is_lm_model = getattr(model_results_comp, 'is_linearmodels', False)
    is_r_feols_model = getattr(model_results_comp, 'is_r_feols', False)
    try:
        if is_r_feols_model: 
            params, bse, pvalues = model_results_comp.params, model_results_comp.bse, model_results_comp.pvalues
        elif is_lm_model: 
            params, bse, pvalues = model_results_comp.params, model_results_comp.std_errors, model_results_comp.pvalues
        elif isinstance(model_results_comp, PooledRegressionResults): 
            params, bse, pvalues = model_results_comp.params, model_results_comp.bse, model_results_comp.pvalues
        elif hasattr(model_results_comp, 'summary2'): 
            summary_obj_comp = model_results_comp.summary2()
            if summary_obj_comp is None or not hasattr(summary_obj_comp, 'tables') or len(summary_obj_comp.tables) < 2: return None
            summary_df_comp = summary_obj_comp.tables[1].copy(); rename_map_comp = {}
            if 'Coef.' in summary_df_comp.columns: rename_map_comp['Coef.'] = 'params'
            elif 'Estimate' in summary_df_comp.columns: rename_map_comp['Estimate'] = 'params'
            if 'Std.Err.' in summary_df_comp.columns: rename_map_comp['Std.Err.'] = 'bse'
            elif 'Std. Err.' in summary_df_comp.columns: rename_map_comp['Std. Err.'] = 'bse'
            if 'P>|t|' in summary_df_comp.columns: rename_map_comp['P>|t|'] = 'pvalues'
            elif 'P>|z|' in summary_df_comp.columns: rename_map_comp['P>|z|'] = 'pvalues'
            elif 'P-value' in summary_df_comp.columns: rename_map_comp['P-value'] = 'pvalues'
            elif 'Pr(>|t|)' in summary_df_comp.columns: rename_map_comp['Pr(>|t|)'] = 'pvalues'
            summary_df_comp = summary_df_comp.rename(columns=rename_map_comp)
            if not all(c in summary_df_comp.columns for c in ['params', 'bse', 'pvalues']):
                 logger.error(f"get_coef_info_py (statsmodels): Missing essential columns after rename. Has: {summary_df_comp.columns.tolist()}")
                 return None
            params, bse, pvalues = summary_df_comp['params'], summary_df_comp['bse'], summary_df_comp['pvalues']
        else: return None 

        if params is None or bse is None or pvalues is None: return None
        
        cleaned_params = pd.Series({clean_coef_name_comp(k): v for k, v in params.items()})
        cleaned_bse = pd.Series({clean_coef_name_comp(k): v for k, v in bse.items()})
        cleaned_pvalues = pd.Series({clean_coef_name_comp(k): v for k, v in pvalues.items()})

        coef_info_df = pd.DataFrame({'est': cleaned_params, 'bse': cleaned_bse, 'pvalues': cleaned_pvalues})
        coef_info_df['sign'] = np.sign(coef_info_df['est'])
        coef_info_df['sig'] = (coef_info_df['pvalues'] < alpha_comp) & (pd.notna(coef_info_df['pvalues']))
        
        substantive_coef_info_df = coef_info_df.copy() 
        if 'Intercept' in substantive_coef_info_df.index:
            substantive_coef_info_df = substantive_coef_info_df.drop('Intercept')
        
        actual_fe_cols_used_list_comp = getattr(model_results_comp, 'fixed_effects_cols_used', [])
        model_key_for_absorb_check_comp = getattr(model_results_comp, 'model_key_name_for_config', None)
        
        if not is_lm_model and not is_r_feols_model and actual_fe_cols_used_list_comp:
            for fe_col_name_filter_comp in actual_fe_cols_used_list_comp:
                fe_col_prefix_comp = f"{clean_coef_name_comp(fe_col_name_filter_comp)}[" 
                substantive_coef_info_df = substantive_coef_info_df[~substantive_coef_info_df.index.str.startswith(fe_col_prefix_comp)]
        
        absorbed_ivs_for_this_model_type_comp = []
        if model_key_for_absorb_check_comp:
            fes_active_in_model = (is_lm_model or is_r_feols_model) or \
                                  (not is_lm_model and not is_r_feols_model and actual_fe_cols_used_list_comp)
            if fes_active_in_model:
                absorbed_ivs_for_this_model_type_comp = Config.MODEL_ABSORBED_IV_BY_FE.get(model_key_for_absorb_check_comp, [])
        
        if absorbed_ivs_for_this_model_type_comp:
            for absorbed_iv_name_comp in absorbed_ivs_for_this_model_type_comp:
                cleaned_absorbed_name = clean_coef_name_comp(absorbed_iv_name_comp)
                if cleaned_absorbed_name in substantive_coef_info_df.index:
                    substantive_coef_info_df = substantive_coef_info_df.drop(cleaned_absorbed_name)
                    
        return substantive_coef_info_df if not substantive_coef_info_df.empty else None
    except Exception as e_get_coef: 
        logger.error(f"get_coef_info_py: Error processing model results: {e_get_coef}", exc_info=True)
        return None


# --- compare_models_py (for overall model comparison metrics) ---
def compare_models_py(baseline_results_comp_model, test_results_comp_model, alpha_comp_models=Config.ALPHA) -> Dict[str, Any]:
    default_error_payload = {'rmse': np.nan, 'avg_rel_se': np.nan, 'vars_sig_changed': ["Model Error"], 'vars_sign_changed': ["Model Error"], 'common_vars_count': 0}
    if baseline_results_comp_model is None: return {**default_error_payload, 'vars_sig_changed': ["Baseline Model Error"], 'vars_sign_changed': ["Baseline Model Error"]}
    if test_results_comp_model is None: return {**default_error_payload, 'vars_sig_changed': ["Test Model Error"], 'vars_sign_changed': ["Test Model Error"]}
    
    baseline_info_comp = get_coef_info_py(baseline_results_comp_model, alpha_comp_models)
    test_info_comp = get_coef_info_py(test_results_comp_model, alpha_comp_models)
    
    if baseline_info_comp is None or baseline_info_comp.empty: return {**default_error_payload, 'vars_sig_changed': ["Baseline Coef Info Error"], 'vars_sign_changed': ["Baseline Coef Info Error"], 'common_vars_count':0 if baseline_info_comp is None else len(baseline_info_comp)}
    if test_info_comp is None or test_info_comp.empty: return {**default_error_payload, 'vars_sig_changed': ["Test Coef Info Error"], 'vars_sign_changed': ["Test Coef Info Error"], 'common_vars_count':0 if test_info_comp is None else len(test_info_comp)}
    
    baseline_info_comp.index = baseline_info_comp.index.astype(str)
    test_info_comp.index = test_info_comp.index.astype(str)
    common_vars_comp = baseline_info_comp.index.intersection(test_info_comp.index)
    
    if len(common_vars_comp) == 0: return {**default_error_payload, 'vars_sig_changed': ["No Common Coefs"], 'vars_sign_changed': ["No Common Coefs"], 'common_vars_count':0}
    
    squared_biases_list, rel_ses_list_comp, vars_sig_changed_list_comp, vars_sign_changed_list_comp = [], [], [], []
    actual_vars_compared_count = 0
    for var_name_comp in common_vars_comp:
        try:
            bl_row_comp = baseline_info_comp.loc[var_name_comp]
            ts_row_comp = test_info_comp.loc[var_name_comp]
            
            if any(pd.isna(val) for val in [bl_row_comp['est'], ts_row_comp['est'], 
                                            bl_row_comp['bse'], ts_row_comp['bse'], 
                                            bl_row_comp['sig'], bl_row_comp['sign'], 
                                            ts_row_comp['sig'], ts_row_comp['sign']]):
                logger.debug(f"compare_models_py: Skipping var '{var_name_comp}' due to NaN in est/bse/sig/sign.")
                continue 

            actual_vars_compared_count += 1
            squared_biases_list.append((ts_row_comp['est'] - bl_row_comp['est'])**2)
            
            if abs(bl_row_comp['bse']) > 1e-9: 
                rel_ses_list_comp.append(ts_row_comp['bse'] / bl_row_comp['bse'])
            elif abs(ts_row_comp['bse']) < 1e-9 : 
                rel_ses_list_comp.append(1.0) 
            else: 
                rel_ses_list_comp.append(np.inf) 
            
            if bl_row_comp['sig'] != ts_row_comp['sig']: 
                vars_sig_changed_list_comp.append(var_name_comp)
            
            if abs(bl_row_comp['est']) > 1e-9 and abs(ts_row_comp['est']) > 1e-9 and \
               bl_row_comp['sign'] != ts_row_comp['sign']:
                vars_sign_changed_list_comp.append(var_name_comp)
        except KeyError:
            logger.debug(f"compare_models_py: Var '{var_name_comp}' not found in one of the coef_info DFs (should not happen with common_vars).")
            continue 
        
    if actual_vars_compared_count == 0: 
        return {**default_error_payload, 'vars_sig_changed': ["No Valid Common Coefs"], 'vars_sign_changed': ["No Valid Common Coefs"], 'common_vars_count':0}
    
    rmse_val = np.sqrt(np.mean(squared_biases_list)) if squared_biases_list else np.nan
    
    valid_rel_ses_vals = [x for x in rel_ses_list_comp if pd.notna(x) and np.isfinite(x)]
    avg_rel_se_val = np.mean(valid_rel_ses_vals) if valid_rel_ses_vals else np.nan
    
    return {'rmse': rmse_val, 'avg_rel_se': avg_rel_se_val, 
            'vars_sig_changed': vars_sig_changed_list_comp if vars_sig_changed_list_comp else ["None"], 
            'vars_sign_changed': vars_sign_changed_list_comp if vars_sign_changed_list_comp else ["None"], 
            'common_vars_count': actual_vars_compared_count}


# --- MODIFIED preprocess_data (for Santamaria et al. 2024) ---
def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data preprocessing for Santamaria et al. (2024)...")
    df = df_raw.copy()

    # Check for required columns before processing
    required_columns = [Config.ID_COLUMN_ORIGINAL, Config.ID_COLUMN_TIME, "Treatment", "CustomersLikert"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Safe numeric conversion with error handling
    for col in [Config.ID_COLUMN_ORIGINAL, Config.ID_COLUMN_TIME, "Treatment", "CustomersLikert"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create unique ID column
    df[Config.ID_COLUMN] = df[Config.ID_COLUMN_ORIGINAL].astype(str) + "_" + df[Config.ID_COLUMN_TIME].astype(str)
    if not df[Config.ID_COLUMN].is_unique:
        logger.warning(f"'{Config.ID_COLUMN}' is not unique. Adding sequence for duplicates.")
        df[Config.ID_COLUMN] = df[Config.ID_COLUMN] + "_dup_" + df.groupby(Config.ID_COLUMN).cumcount().astype(str)
    
    # Safe column processing with existence checks
    if "Age" in df.columns: 
        df["Age"] = pd.to_numeric(df["Age"], errors='coerce')

    if "Gender" in df.columns:
        df["Gender_dummy"] = df["Gender"].apply(
            lambda x: 1 if isinstance(x, str) and x.lower() == 'female' 
            else (0 if isinstance(x, str) and x.lower() == 'male' else np.nan)
        )
        df["Gender_dummy"] = pd.to_numeric(df["Gender_dummy"], errors='coerce')

    if "Ethnicity" in df.columns:
        df["Chinese_dummy"] = df["Ethnicity"].apply(
            lambda x: 1 if isinstance(x, str) and x.lower() == 'chinese' 
            else (0 if pd.notna(x) else np.nan)
        )
        df["Chinese_dummy"] = pd.to_numeric(df["Chinese_dummy"], errors='coerce')
    
    if "HighestEducationAttained" in df.columns:
        edu_map = {
            "phd": 4, "masters": 3, "degree": 2, 
            "diploma or equivalent": 1, 
        }
        df["Education_numeric"] = df["HighestEducationAttained"].astype(str).str.lower().map(edu_map)
        df["Education_numeric"] = pd.to_numeric(df["Education_numeric"], errors='coerce')

    if "Fieldofstudy" in df.columns:
        stem_fields = ['engineering', 'computing', 'science & mathematics']
        df["STEM_dummy"] = df["Fieldofstudy"].astype(str).str.lower().apply(
            lambda x: 1 if x in stem_fields else (0 if pd.notna(x) else np.nan)
        )
        df["STEM_dummy"] = pd.to_numeric(df["STEM_dummy"], errors='coerce')
    
        df["Business_dummy"] = df["Fieldofstudy"].astype(str).str.lower().apply(
            lambda x: 1 if x == 'business' else (0 if pd.notna(x) else np.nan)
        )
        df["Business_dummy"] = pd.to_numeric(df["Business_dummy"], errors='coerce')

    # Process status columns safely
    for col_std in ["Studying", "Working", "EntrepExperience", "Registered"]:
        if col_std in df.columns:
            df[f"{col_std}_dummy"] = pd.to_numeric(df[col_std], errors='coerce') 

    if "WorkExp" in df.columns:
         df["WorkExp_numeric"] = pd.to_numeric(df["WorkExp"], errors='coerce')

    if "IndustryExperience" in df.columns:
        stem_exp_fields = ['engineering', 'technology']
        df["STEMEXP_dummy"] = df["IndustryExperience"].astype(str).str.lower().apply(
            lambda x: 1 if x in stem_exp_fields else (0 if pd.notna(x) else np.nan)
        )
        df["STEMEXP_dummy"] = pd.to_numeric(df["STEMEXP_dummy"], errors='coerce')

    if "TeamSize" in df.columns: 
        df["TeamSize"] = pd.to_numeric(df["TeamSize"], errors='coerce')
    
    if "Sector" in df.columns and "Sector" in Config.POTENTIAL_COVARIATES_TO_DUMMIFY_FOR_IMPUTATION:
        df["Sector_numeric"] = pd.factorize(df["Sector"])[0]
        df["Sector_numeric"] = df["Sector_numeric"].replace(-1, np.nan) 

    if "Round" in df.columns:
        df["Round_dummy"] = pd.to_numeric(df["Round"], errors='coerce')
        # Safely subtract 1, handling NaN
        df["Round_dummy"] = df["Round_dummy"].apply(lambda x: x - 1 if pd.notna(x) else np.nan)

    # Create baseline columns safely
    for outcome_prefix in ["RevenueLikert", "CustomersLikert"]:
        baseline_col_name = f"Initial{outcome_prefix}_Post0"
        if outcome_prefix in df.columns and Config.ID_COLUMN_ORIGINAL in df.columns and Config.ID_COLUMN_TIME in df.columns:
            try:
                baseline_values = df[df[Config.ID_COLUMN_TIME] == 0].set_index(Config.ID_COLUMN_ORIGINAL)[outcome_prefix]
                df[baseline_col_name] = df[Config.ID_COLUMN_ORIGINAL].map(baseline_values)
                df[baseline_col_name] = pd.to_numeric(df[baseline_col_name], errors='coerce')
            except Exception as e:
                logger.warning(f"Could not create baseline column {baseline_col_name}: {e}")
                df[baseline_col_name] = np.nan
        else:
            df[baseline_col_name] = np.nan

    # Update NUMERICAL_COLS_FOR_IMPUTATION with actually created columns
    current_numeric_cols = list(Config.POTENTIAL_COVARIATES_NUMERIC) 
    created_dummies = ["Gender_dummy", "Chinese_dummy", "STEM_dummy", "Business_dummy",
                       "Studying_dummy", "Working_dummy", "EntrepExperience_dummy", "Registered_dummy",
                       "STEMEXP_dummy", "Round_dummy", "Sector_numeric"] 
    for cd in created_dummies:
        if cd in df.columns: 
            current_numeric_cols.append(cd)
    
    current_numeric_cols.extend(Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS) 
    Config.NUMERICAL_COLS_FOR_IMPUTATION = sorted(list(set(col for col in current_numeric_cols if col in df.columns)))
    logger.info(f"Final NUMERICAL_COLS_FOR_IMPUTATION: {Config.NUMERICAL_COLS_FOR_IMPUTATION}")
    
    # Set index safely
    if Config.ID_COLUMN in df.columns:
        df = df.set_index(Config.ID_COLUMN, drop=False) 
    else:
        raise ValueError(f"Preprocessing: {Config.ID_COLUMN} not created/found to set as index.")

    logger.info(f"Finished data preprocessing. DataFrame shape: {df.shape}")
    for col_log_na in ["CustomersLikert", "Post", "Treatment", "SN"]:
        if col_log_na in df.columns:
            logger.info(f"NA count in '{col_log_na}' after preprocessing: {df[col_log_na].isnull().sum()}")
        else:
            logger.warning(f"Key regression variable '{col_log_na}' not found after preprocessing.")
            
    return df


# --- ImputationPipeline (Largely unchanged, relies on Config.NUMERICAL_COLS_FOR_IMPUTATION) ---
class ImputationPipeline:
    def __init__(self, input_df_with_na: pd.DataFrame, original_df_complete_subset: pd.DataFrame, 
                 missingness_level_config: float, mechanism_config: str,
                 current_key_var_imputed: str, iteration_num: int):
        self.missingness_level = missingness_level_config; self.mechanism = mechanism_config
        self.current_key_var_imputed = current_key_var_imputed; self.iteration_num = iteration_num
        
        self.output_dir_imputed_data = Config.get_data_path(
            self.mechanism, self.missingness_level, "imputed_dir", 
            iteration=self.iteration_num, key_var_imputed_for_path=self.current_key_var_imputed
        )

        self.df = input_df_with_na.copy(); self.original_df = original_df_complete_subset.copy() 
        self.id_column_used = Config.ID_COLUMN
        
        if self.df.index.name != self.id_column_used:
            if self.id_column_used in self.df.columns: self.df = self.df.set_index(self.id_column_used, drop=False)
            else: raise ValueError(f"ImputePipeline: ID '{self.id_column_used}' not in input_df_with_na columns to set as index.")
        if self.original_df.index.name != self.id_column_used:
            if self.id_column_used in self.original_df.columns: self.original_df = self.original_df.set_index(self.id_column_used, drop=False)
            else: raise ValueError(f"ImputePipeline: ID '{self.id_column_used}' not in original_df columns to set as index.")

        if not self.df.index.is_unique: logger.warning(f"ImputePipeline: Index of df not unique for {self.current_key_var_imputed} iter {self.iteration_num}.")
        if not self.original_df.index.is_unique: logger.warning(f"ImputePipeline: Index of original_df not unique for {self.current_key_var_imputed} iter {self.iteration_num}.")
        
        common_indices = self.df.index.intersection(self.original_df.index)
        self.df = self.df.loc[common_indices]; self.original_df = self.original_df.loc[common_indices]
        
        self.numeric_cols_present_for_imputation = []
        for col_num_imp in Config.NUMERICAL_COLS_FOR_IMPUTATION: 
            if col_num_imp in self.df.columns:
                if not pd.api.types.is_numeric_dtype(self.df[col_num_imp]): 
                    self.df[col_num_imp] = pd.to_numeric(self.df[col_num_imp], errors='coerce')
                if self.df[col_num_imp].notna().any(): 
                    self.numeric_cols_present_for_imputation.append(col_num_imp)
            
            if col_num_imp in self.original_df.columns and not pd.api.types.is_numeric_dtype(self.original_df[col_num_imp]): 
                self.original_df[col_num_imp] = pd.to_numeric(self.original_df[col_num_imp], errors='coerce')
        
        logger.debug(f"ImputationPipeline: Numeric cols for imputation: {self.numeric_cols_present_for_imputation}")
        if not self.numeric_cols_present_for_imputation:
            logger.warning("ImputationPipeline: No numeric columns identified for imputation methods.")


    def listwise_deletion(self) -> pd.DataFrame: 
        if self.current_key_var_imputed not in self.df.columns:
            logger.warning(f"Listwise Deletion: Key var '{self.current_key_var_imputed}' not in DataFrame. Returning original.")
            return self.df.copy()
        return self.df.dropna(subset=[self.current_key_var_imputed])

    def mean_imputation(self) -> pd.DataFrame: 
        imputed_df = self.df.copy()
        if self.current_key_var_imputed in imputed_df.columns and imputed_df[self.current_key_var_imputed].isna().any():
            if not pd.api.types.is_numeric_dtype(imputed_df[self.current_key_var_imputed]):
                logger.warning(f"MeanImpute: Target '{self.current_key_var_imputed}' not numeric. Attempting coercion.")
                imputed_df[self.current_key_var_imputed] = pd.to_numeric(imputed_df[self.current_key_var_imputed], errors='coerce')

            mean_val = imputed_df[self.current_key_var_imputed].mean()
            if pd.isna(mean_val): logger.warning(f"MeanImpute: Mean for '{self.current_key_var_imputed}' is NaN. Filling NAs with 0.")
            imputed_df[self.current_key_var_imputed] = imputed_df[self.current_key_var_imputed].fillna(mean_val if pd.notna(mean_val) else 0)
        return imputed_df

    def _impute_predictors_mean(self, df_to_fill: pd.DataFrame, cols_list: List[str]) -> pd.DataFrame:
        df_filled = df_to_fill.copy()
        for col_fill in cols_list:
            if col_fill in df_filled.columns and df_filled[col_fill].isna().any():
                if not pd.api.types.is_numeric_dtype(df_filled[col_fill]):
                    logger.debug(f"_impute_predictors_mean: Predictor '{col_fill}' not numeric. Coercing.")
                    df_filled[col_fill] = pd.to_numeric(df_filled[col_fill], errors='coerce')
                mean_val = df_filled[col_fill].mean()
                if pd.isna(mean_val):
                    logger.warning(f"_impute_predictors_mean: Mean for predictor '{col_fill}' is NaN. Filling with 0.")
                df_filled[col_fill] = df_filled[col_fill].fillna(mean_val if pd.notna(mean_val) else 0)
        return df_filled

    def regression_imputation(self) -> pd.DataFrame: 
        imputed_df_reg = self.df.copy()
        target_col = self.current_key_var_imputed
        
        if target_col not in imputed_df_reg.columns or not imputed_df_reg[target_col].isna().any():
            return imputed_df_reg 

        if not pd.api.types.is_numeric_dtype(imputed_df_reg[target_col]):
            imputed_df_reg[target_col] = pd.to_numeric(imputed_df_reg[target_col], errors='coerce')

        predictor_cols = [p for p in self.numeric_cols_present_for_imputation if p != target_col and p in imputed_df_reg.columns and imputed_df_reg[p].notna().sum() > 0.5 * len(imputed_df_reg)] 
        if not predictor_cols: 
            logger.warning(f"RegImpute: No valid predictors for {target_col}. Fallback to mean.")
            return self.mean_imputation()

        temp_df_for_predictors = self._impute_predictors_mean(imputed_df_reg.copy(), predictor_cols)
        
        original_missing_mask = self.df[target_col].isna()
        fallback_mean = self.df[target_col].mean()
        fallback_mean = 0 if pd.isna(fallback_mean) else fallback_mean

        X_train_df = temp_df_for_predictors.loc[~original_missing_mask, predictor_cols]
        y_train_series = self.df.loc[~original_missing_mask, target_col]
        X_pred_df = temp_df_for_predictors.loc[original_missing_mask, predictor_cols]

        # Fix index alignment issues
        common_idx_train = X_train_df.dropna(how='any').index.intersection(y_train_series.dropna().index)
        if common_idx_train.empty:
            logger.warning(f"RegImpute: No common training indices for {target_col}. Using fallback.")
            imputed_df_reg.loc[original_missing_mask, target_col] = fallback_mean
            return imputed_df_reg
            
        X_train_c = X_train_df.loc[common_idx_train]
        y_train_c = y_train_series.loc[common_idx_train]
        X_pred_c = X_pred_df.dropna(how='any') 
        predictable_missing_indices = X_pred_c.index

        if X_train_c.empty or y_train_c.empty or X_train_c.shape[0] < max(2, len(predictor_cols) + 1) or X_pred_c.empty:
            imputed_df_reg.loc[original_missing_mask, target_col] = fallback_mean
            return imputed_df_reg
        
        try:
            model = LinearRegression()
            model.fit(X_train_c, y_train_c)
            preds_np = model.predict(X_pred_c)
            final_imputed_values_np = preds_np
            if Config.ADD_RESIDUAL_NOISE:
                sd = calculate_residual_sd(model, X_train_c, y_train_c)
                if sd > 0 and pd.notna(sd): 
                    final_imputed_values_np = preds_np + np.random.normal(0, sd, size=len(preds_np))
            
            final_imputed_values_np = np.asarray(final_imputed_values_np).flatten()
            # Fix indexing bug - ensure lengths match
            if len(predictable_missing_indices) == len(final_imputed_values_np) and len(predictable_missing_indices) > 0:
                imputed_df_reg.loc[predictable_missing_indices, target_col] = final_imputed_values_np
            
            still_missing_after_pred = imputed_df_reg[target_col].isna() & original_missing_mask
            if still_missing_after_pred.any():
                imputed_df_reg.loc[still_missing_after_pred, target_col] = fallback_mean
        except Exception as e:
            logger.error(f"RegImpute error for {target_col}: {e}", exc_info=False)
            imputed_df_reg.loc[original_missing_mask, target_col] = fallback_mean
        return imputed_df_reg

    def stochastic_iterative_imputation(self) -> pd.DataFrame: 
        imputed_df = self.df.copy()
        
        if self.current_key_var_imputed in imputed_df.columns and not pd.api.types.is_numeric_dtype(imputed_df[self.current_key_var_imputed]):
            imputed_df[self.current_key_var_imputed] = pd.to_numeric(imputed_df[self.current_key_var_imputed], errors='coerce')

        cols_for_iterative_imputer_candidate = [self.current_key_var_imputed] + \
            [p for p in self.numeric_cols_present_for_imputation if p != self.current_key_var_imputed and p in imputed_df.columns]
        
        cols_for_iterative_imputer = []
        for col in cols_for_iterative_imputer_candidate:
            if col in imputed_df.columns:
                if not pd.api.types.is_numeric_dtype(imputed_df[col]):
                     imputed_df[col] = pd.to_numeric(imputed_df[col], errors='coerce') 
                if imputed_df[col].isnull().sum() < len(imputed_df): 
                    cols_for_iterative_imputer.append(col)
        
        cols_for_iterative_imputer = list(set(cols_for_iterative_imputer)) 

        if self.current_key_var_imputed not in cols_for_iterative_imputer or not imputed_df[self.current_key_var_imputed].isnull().any():
             logger.warning(f"IterativeImpute: Target {self.current_key_var_imputed} not in imputer list or no NAs. Returning copy.")
             return imputed_df 

        if len(cols_for_iterative_imputer) <= 1: 
             logger.warning(f"IterativeImpute: Not enough cols for imputer ({len(cols_for_iterative_imputer)}). Fallback to mean.")
             return self.mean_imputation()

        data_subset_for_iterative = imputed_df[cols_for_iterative_imputer].copy()
        n_features = data_subset_for_iterative.shape[1]

        if n_features == 0: return imputed_df 
        try:
            imputer_seed = Config.RANDOM_SEED_IMPUTATION + self.iteration_num 
            imputer = IterativeImputer(estimator=BayesianRidge(), random_state=imputer_seed, 
                                       max_iter=Config.MICE_ITERATIONS, 
                                       n_nearest_features=min(10, n_features -1) if n_features > 1 else None, 
                                       sample_posterior=True, tol=1e-3, verbose=0,
                                       initial_strategy='mean', imputation_order='ascending')
            imputed_values = imputer.fit_transform(data_subset_for_iterative)
            imputed_df[cols_for_iterative_imputer] = imputed_values

            if imputed_df[self.current_key_var_imputed].isnull().any():
                fallback_mean = self.df[self.current_key_var_imputed].mean()
                imputed_df[self.current_key_var_imputed] = imputed_df[self.current_key_var_imputed].fillna(fallback_mean if pd.notna(fallback_mean) else 0)
            return imputed_df
        except Exception as e: 
            logger.error(f"IterativeImpute Error: {e}", exc_info=False)
            return self.mean_imputation()

    def ml_imputation(self) -> pd.DataFrame: 
        imputed_df_ml = self.df.copy()
        target_col = self.current_key_var_imputed

        if target_col not in imputed_df_ml.columns or not imputed_df_ml[target_col].isna().any():
            return imputed_df_ml
        if not pd.api.types.is_numeric_dtype(imputed_df_ml[target_col]):
            imputed_df_ml[target_col] = pd.to_numeric(imputed_df_ml[target_col], errors='coerce')

        predictor_cols = [p for p in self.numeric_cols_present_for_imputation if p != target_col and p in imputed_df_ml.columns and imputed_df_ml[p].notna().sum() > 0.5 * len(imputed_df_ml)]
        if not predictor_cols: return self.mean_imputation()

        temp_df_for_predictors = self._impute_predictors_mean(imputed_df_ml.copy(), predictor_cols)
        
        original_missing_mask = self.df[target_col].isna()
        fallback_mean = self.df[target_col].mean(); fallback_mean = 0 if pd.isna(fallback_mean) else fallback_mean
        
        X_train_df = temp_df_for_predictors.loc[~original_missing_mask, predictor_cols]
        y_train_series = self.df.loc[~original_missing_mask, target_col]
        X_pred_df = temp_df_for_predictors.loc[original_missing_mask, predictor_cols]

        common_idx_train = X_train_df.dropna(how='any').index.intersection(y_train_series.dropna().index)
        X_train_c, y_train_c = X_train_df.loc[common_idx_train], y_train_series.loc[common_idx_train]
        X_pred_c = X_pred_df.dropna(how='any')
        predictable_missing_indices = X_pred_c.index

        if X_train_c.empty or y_train_c.empty or X_train_c.shape[0] < 5 or X_pred_c.empty:
            imputed_df_ml.loc[original_missing_mask, target_col] = fallback_mean
            return imputed_df_ml
        
        try:
            model_seed = Config.RANDOM_SEED_IMPUTATION + self.iteration_num
            model = RandomForestRegressor(n_estimators=Config.MICE_LGBM_N_ESTIMATORS, 
                                          max_depth=Config.MICE_LGBM_MAX_DEPTH, 
                                          min_samples_leaf=max(2, Config.MICE_LGBM_NUM_LEAVES // 2),
                                          random_state=model_seed, n_jobs=1)
            model.fit(X_train_c, y_train_c)
            preds = model.predict(X_pred_c)
            noise = 0.0
            if Config.ADD_RESIDUAL_NOISE:
                sd = calculate_residual_sd(model, X_train_c, y_train_c)
                if sd > 0 and pd.notna(sd): noise = np.random.normal(0, sd, size=len(preds))
            
            imputed_df_ml.loc[predictable_missing_indices, target_col] = preds + noise
            still_missing_after_pred = imputed_df_ml[target_col].isna() & original_missing_mask
            if still_missing_after_pred.any():
                imputed_df_ml.loc[still_missing_after_pred, target_col] = fallback_mean
        except Exception as e:
            logger.error(f"MLImpute (RF) error for {target_col}: {e}", exc_info=False)
            imputed_df_ml.loc[original_missing_mask, target_col] = fallback_mean
        return imputed_df_ml

    def deep_learning_imputation(self) -> pd.DataFrame:
        imputed_df_dl = self.df.copy()
        target_col = self.current_key_var_imputed

        if target_col not in imputed_df_dl.columns or not imputed_df_dl[target_col].isna().any():
            return imputed_df_dl
        if not pd.api.types.is_numeric_dtype(imputed_df_dl[target_col]):
             imputed_df_dl[target_col] = pd.to_numeric(imputed_df_dl[target_col], errors='coerce')


        predictor_cols = [p for p in self.numeric_cols_present_for_imputation if p != target_col and p in imputed_df_dl.columns and imputed_df_dl[p].notna().sum() > 0.5 * len(imputed_df_dl)]
        if not predictor_cols: return self.mean_imputation()

        temp_df_for_predictors = self._impute_predictors_mean(imputed_df_dl.copy(), predictor_cols)

        original_missing_mask = self.df[target_col].isna()
        fallback_mean = self.df[target_col].mean(); fallback_mean = 0 if pd.isna(fallback_mean) else fallback_mean

        X_train_df = temp_df_for_predictors.loc[~original_missing_mask, predictor_cols]
        y_train_series = self.df.loc[~original_missing_mask, target_col]
        X_pred_df = temp_df_for_predictors.loc[original_missing_mask, predictor_cols]
        
        common_idx_train = X_train_df.dropna(how='any').index.intersection(y_train_series.dropna().index)
        X_train_c, y_train_c = X_train_df.loc[common_idx_train], y_train_series.loc[common_idx_train]
        X_pred_c = X_pred_df.dropna(how='any')
        predictable_missing_indices = X_pred_c.index

        if X_train_c.empty or y_train_c.empty or y_train_c.isna().any() or X_train_c.shape[0] < 10 or X_pred_c.empty:
            imputed_df_dl.loc[original_missing_mask, target_col] = fallback_mean
            return imputed_df_dl
        
        try:
            tf.random.set_seed(Config.RANDOM_SEED_IMPUTATION + self.iteration_num)
            scaler_X, scaler_y = StandardScaler(), StandardScaler()
            X_train_s = scaler_X.fit_transform(X_train_c)
            y_train_s = scaler_y.fit_transform(y_train_c.values.reshape(-1,1))
            
            model = create_mlp(X_train_s.shape[1])
            val_split = 0.1 if len(X_train_s)*0.1 >=1 else 0.0; monitor = 'val_loss' if val_split > 0 else 'loss'
            cb = [EarlyStopping(monitor=monitor, patience=Config.DL_PATIENCE, restore_best_weights=True, verbose=0)]
            model.fit(X_train_s, y_train_s, epochs=Config.DL_EPOCHS, batch_size=min(32, len(X_train_s)), validation_split=val_split, callbacks=cb, verbose=0)
            
            if not X_pred_c.empty:
                X_pred_s = scaler_X.transform(X_pred_c)
                y_pred_s = model.predict(X_pred_s, verbose=0)
                preds = scaler_y.inverse_transform(y_pred_s).flatten()
                noise = 0.0
                if Config.ADD_RESIDUAL_NOISE:
                    y_pred_train_s_for_sd = model.predict(X_train_s, verbose=0)
                    y_pred_train_us_for_sd = scaler_y.inverse_transform(y_pred_train_s_for_sd).flatten()
                    resids = y_train_c.values - y_pred_train_us_for_sd
                    non_zero_resids = resids[np.abs(resids) > 1e-9]
                    if len(non_zero_resids) > 1:
                        sd = np.std(non_zero_resids)
                        if not pd.isna(sd) and sd > 0: noise = np.random.normal(0, sd, size=len(preds))
                
                imputed_df_dl.loc[predictable_missing_indices, target_col] = preds + noise
            
            still_missing_after_pred = imputed_df_dl[target_col].isna() & original_missing_mask
            if still_missing_after_pred.any():
                imputed_df_dl.loc[still_missing_after_pred, target_col] = fallback_mean
            del model; tf.keras.backend.clear_session()
        except Exception as e:
            logger.error(f"DLImpute error for {target_col}: {e}", exc_info=False)
            imputed_df_dl.loc[original_missing_mask, target_col] = fallback_mean
            if 'model' in locals() and model is not None: del model
            tf.keras.backend.clear_session()
        return imputed_df_dl

    def custom_multiple_imputation(self) -> List[pd.DataFrame]: 
        all_imputed_datasets_mice: List[pd.DataFrame] = []
        original_df_with_na_mice = self.df.copy() 
        target_col_mice = self.current_key_var_imputed

        if target_col_mice not in original_df_with_na_mice.columns or not original_df_with_na_mice[target_col_mice].isna().any():
            return [original_df_with_na_mice.copy() for _ in range(Config.N_IMPUTATIONS)]
        if not pd.api.types.is_numeric_dtype(original_df_with_na_mice[target_col_mice]):
             original_df_with_na_mice[target_col_mice] = pd.to_numeric(original_df_with_na_mice[target_col_mice], errors='coerce')

        predictor_cols_mice = [p for p in self.numeric_cols_present_for_imputation if p != target_col_mice and p in original_df_with_na_mice.columns and original_df_with_na_mice[p].notna().sum() > 0.5 * len(original_df_with_na_mice)]

        for m_idx_mice in range(Config.N_IMPUTATIONS):
            iter_seed = Config.RANDOM_SEED_IMPUTATION + m_idx_mice + self.iteration_num
            np.random.seed(iter_seed); tf.random.set_seed(iter_seed)
            
            current_imputed_df_m_mice = original_df_with_na_mice.copy()
            df_for_predictors_mice = self._impute_predictors_mean(current_imputed_df_m_mice.copy(), predictor_cols_mice)
            
            for iteration_mice in range(Config.MICE_ITERATIONS):
                missing_mask = original_df_with_na_mice[target_col_mice].isna()
                if not missing_mask.any(): break 

                fallback_mean = original_df_with_na_mice[target_col_mice].mean()
                fallback_mean = 0 if pd.isna(fallback_mean) else fallback_mean

                if not predictor_cols_mice: 
                    current_imputed_df_m_mice.loc[missing_mask, target_col_mice] = fallback_mean
                    df_for_predictors_mice.loc[missing_mask, target_col_mice] = fallback_mean
                    continue

                X_train_df = df_for_predictors_mice.loc[~missing_mask, predictor_cols_mice]
                y_train_series = original_df_with_na_mice.loc[~missing_mask, target_col_mice]
                X_pred_df = df_for_predictors_mice.loc[missing_mask, predictor_cols_mice]

                common_idx_train = X_train_df.dropna(how='any').index.intersection(y_train_series.dropna().index)
                X_train_c, y_train_c = X_train_df.loc[common_idx_train], y_train_series.loc[common_idx_train]
                X_pred_c = X_pred_df.dropna(how='any')
                predictable_missing_indices_mice = X_pred_c.index
                
                if X_train_c.empty or y_train_c.empty or X_pred_c.empty or X_train_c.shape[0] < 5:
                    current_imputed_df_m_mice.loc[predictable_missing_indices_mice, target_col_mice] = fallback_mean
                    df_for_predictors_mice.loc[predictable_missing_indices_mice, target_col_mice] = fallback_mean
                    still_missing_mice = current_imputed_df_m_mice[target_col_mice].isna() & missing_mask
                    if still_missing_mice.any():
                        current_imputed_df_m_mice.loc[still_missing_mice, target_col_mice] = fallback_mean
                        df_for_predictors_mice.loc[still_missing_mice, target_col_mice] = fallback_mean
                    continue
                
                try:
                    model_seed_lgbm = iter_seed + iteration_mice + sum(ord(c) for c in target_col_mice)
                    model = lgb.LGBMRegressor(n_estimators=Config.MICE_LGBM_N_ESTIMATORS, max_depth=Config.MICE_LGBM_MAX_DEPTH, 
                                              learning_rate=Config.MICE_LGBM_LEARNING_RATE, num_leaves=Config.MICE_LGBM_NUM_LEAVES, 
                                              random_state=model_seed_lgbm, n_jobs=1, verbose=Config.MICE_LGBM_VERBOSITY)
                    model.fit(X_train_c, y_train_c)
                    preds = model.predict(X_pred_c)
                    noise = 0.0
                    if Config.ADD_RESIDUAL_NOISE:
                        sd = calculate_residual_sd(model, X_train_c, y_train_c)
                        if sd > 0 and pd.notna(sd): noise = np.random.normal(0, sd, size=len(preds))
                    
                    imputed_vals = preds + noise
                    current_imputed_df_m_mice.loc[predictable_missing_indices_mice, target_col_mice] = imputed_vals
                    df_for_predictors_mice.loc[predictable_missing_indices_mice, target_col_mice] = imputed_vals
                    
                    still_missing_mice_success = current_imputed_df_m_mice[target_col_mice].isna() & missing_mask
                    if still_missing_mice_success.any():
                        current_imputed_df_m_mice.loc[still_missing_mice_success, target_col_mice] = fallback_mean
                        df_for_predictors_mice.loc[still_missing_mice_success, target_col_mice] = fallback_mean
                except Exception as e_mice_inner:
                    logger.error(f"MICE inner loop error for {target_col_mice}: {e_mice_inner}", exc_info=False)
                    current_imputed_df_m_mice.loc[missing_mask, target_col_mice] = fallback_mean
                    df_for_predictors_mice.loc[missing_mask, target_col_mice] = fallback_mean
            
            if current_imputed_df_m_mice[target_col_mice].isnull().any():
                final_fallback_mean = original_df_with_na_mice[target_col_mice].mean()
                current_imputed_df_m_mice[target_col_mice].fillna(final_fallback_mean if pd.notna(final_fallback_mean) else 0, inplace=True)
            
            all_imputed_datasets_mice.append(current_imputed_df_m_mice)
        
        return all_imputed_datasets_mice

    def run_all_imputations_and_save(self) -> Dict[str, Any]:
        imputation_results = {}
        for method_name in tqdm(Config.IMPUTATION_METHODS_TO_COMPARE, 
                              desc=f"Imputing {self.current_key_var_imputed}", unit="method",
                              position=4, leave=False, disable=not Config.USE_PARALLEL): 
            try:
                imputation_method = getattr(self, method_name)
                imputed_output = imputation_method()
                
                if imputed_output is not None:
                    safe_create_directory(self.output_dir_imputed_data) 

                    if isinstance(imputed_output, list): 
                        for m, imputed_df_m in enumerate(imputed_output):
                            if isinstance(imputed_df_m, pd.DataFrame):
                                self.save_dataframe(imputed_df_m, f"{method_name}_m{m}")
                    elif isinstance(imputed_output, pd.DataFrame):
                        self.save_dataframe(imputed_output, method_name)
                
                imputation_results[method_name] = imputed_output 
            except Exception as e:
                logger.error(f"Error in {method_name} for {self.current_key_var_imputed} (iter {self.iteration_num}): {e}", exc_info=True)
                imputation_results[method_name] = None
        return imputation_results

    def save_dataframe(self, df_to_save: pd.DataFrame, method_name_save: str) -> None: 
        if not isinstance(df_to_save, pd.DataFrame): 
            logger.warning(f"Save DF: Input for '{method_name_save}' not DataFrame. Skip."); return
        if df_to_save.empty and method_name_save != "listwise_deletion":
            logger.warning(f"Save DF: Input for '{method_name_save}' is empty. Skip."); return
        
        try:
            output_df_save = df_to_save.copy()
            if output_df_save.index.name == Config.ID_COLUMN:
                if Config.ID_COLUMN in output_df_save.columns: output_df_save = output_df_save.reset_index(drop=True)
                else: output_df_save = output_df_save.reset_index()
            elif Config.ID_COLUMN not in output_df_save.columns:
                logger.error(f"Save DF: Critical - {Config.ID_COLUMN} missing from DF for {method_name_save}.")
                if self.original_df is not None and Config.ID_COLUMN in self.original_df.columns and \
                   output_df_save.index.isin(self.original_df.index).all():
                    id_map = self.original_df.loc[output_df_save.index, Config.ID_COLUMN]
                    output_df_save[Config.ID_COLUMN] = id_map.values
                    logger.info(f"Save DF: Restored {Config.ID_COLUMN} for {method_name_save} from original_df map.")
                else: 
                    logger.warning(f"Save DF: Could not restore {Config.ID_COLUMN} for {method_name_save}.")


            if Config.ID_COLUMN in output_df_save.columns and self.original_df is not None:
                if Config.ID_COLUMN_ORIGINAL not in output_df_save.columns:
                    id_map_sn = self.original_df.set_index(Config.ID_COLUMN)[Config.ID_COLUMN_ORIGINAL]
                    output_df_save[Config.ID_COLUMN_ORIGINAL] = output_df_save[Config.ID_COLUMN].map(id_map_sn)
                if Config.ID_COLUMN_TIME not in output_df_save.columns:
                    id_map_post = self.original_df.set_index(Config.ID_COLUMN)[Config.ID_COLUMN_TIME]
                    output_df_save[Config.ID_COLUMN_TIME] = output_df_save[Config.ID_COLUMN].map(id_map_post)
            
            csv_path_save = Config.get_data_path(self.mechanism, self.missingness_level, "imputed_file", 
                                                 method_name_save, self.iteration_num, self.current_key_var_imputed)
            safe_create_directory(os.path.dirname(csv_path_save))
            output_df_save.to_csv(csv_path_save, index=False)
        except Exception as e_save: 
            logger.error(f"Save DF Error for '{method_name_save}' (Var: {self.current_key_var_imputed}, Iter: {self.iteration_num}): {e_save}", exc_info=True)


# --- Parallel Processing Helper Functions ---
def safe_create_directory(directory_path: str) -> None:
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Error creating directory {directory_path} (worker might race): {e}")
        time.sleep(0.05) 
        try: Path(directory_path).mkdir(parents=True, exist_ok=True)
        except Exception as e2: logger.error(f"Failed to create directory {directory_path} on retry: {e2}")


def recursive_defaultdict_to_dict(d):
    """Convert nested defaultdicts to regular dicts recursively"""
    if isinstance(d, defaultdict):
        d = {k: recursive_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: recursive_defaultdict_to_dict(v) for k, v in d.items()}
    return d

def process_single_iteration_wrapper(args_tuple_wrapper):
    """Process a single iteration with improved error handling"""
    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.get_logger().setLevel('ERROR')

    # 确保Config在子进程中正确初始化
    Config()

    try:
        # Unpack arguments safely
        if len(args_tuple_wrapper) != 8:
            logger.error(f"Invalid args_tuple_wrapper length: {len(args_tuple_wrapper)}, expected 8")
            return {'error': 'Invalid argument count', 'args_length': len(args_tuple_wrapper)}
            
        (key_var_imputed_iter, mechanism_iter, miss_level_iter, i_iter_loop, 
         full_data_path_iter, baseline_models_path_iter, baseline_coef_info_path_iter, numerical_cols_path_iter) = args_tuple_wrapper
        
        # Load data from pickle files
        try:
            with open(full_data_path_iter, 'rb') as f_full:
                full_data_for_sim_iter = pickle.load(f_full)
            with open(baseline_models_path_iter, 'rb') as f_base_m:
                baseline_models_iter = pickle.load(f_base_m)
            with open(baseline_coef_info_path_iter, 'rb') as f_base_c:
                baseline_coef_info_iter = pickle.load(f_base_c)
            with open(numerical_cols_path_iter, 'rb') as f_num:
                Config.NUMERICAL_COLS_FOR_IMPUTATION = pickle.load(f_num)
        except Exception as e_pickle:
            logger.error(f"Error loading data from pickle files in iteration wrapper: {e_pickle}")
            return {'error': f'File loading failed: {str(e_pickle)}', 'key_var_imputed': key_var_imputed_iter, 'mechanism': mechanism_iter, 'miss_level': miss_level_iter, 'i_iter': i_iter_loop}

        # Validate input data
        if not isinstance(full_data_for_sim_iter, pd.DataFrame) or full_data_for_sim_iter.empty:
            logger.error(f"Invalid full_data_for_sim_iter: {type(full_data_for_sim_iter)}")
            return {'error': 'Invalid full_data_for_sim_iter', 'key_var_imputed': key_var_imputed_iter, 'mechanism': mechanism_iter, 'miss_level': miss_level_iter, 'i_iter': i_iter_loop}

        iteration_results_payload = {
            'key_var_imputed': key_var_imputed_iter, 'mechanism': mechanism_iter,
            'miss_level': miss_level_iter, 'i_iter': i_iter_loop,
            'coef_stability_updates_iter': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'both_same_count': 0, 'sign_same_sig_changed_count': 0, 'total_runs': 0}))))),
            'stats_features_updates_iter': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'variances': [], 'skewnesses': []})))),
            'model_comparison_updates_iter': defaultdict(lambda: defaultdict(list)) 
        }
        level_str_iter = f"{int(miss_level_iter*100)}%"
        current_seed_iter = Config.SIMULATION_SEED + i_iter_loop + sum(ord(c) for c in key_var_imputed_iter) + \
                            (1000 if mechanism_iter == "MAR" else (2000 if mechanism_iter == "NMAR" else 0)) + \
                            int(miss_level_iter * 10000)
        
        data_with_missing_sim_iter = simulate_missingness_single_col(
            full_data_for_sim_iter.copy(), col_to_make_missing=key_var_imputed_iter,
            miss_prop=miss_level_iter, seed=current_seed_iter, mechanism=mechanism_iter,
            mar_control_col=Config.MAR_CONTROL_COL, mar_strength=Config.MAR_STRENGTH_FACTOR,
            nmar_strength=Config.NMAR_STRENGTH_FACTOR
        )
        
        sim_filepath_iter = Config.get_data_path(mechanism_iter, miss_level_iter, "simulated", 
                                                 iteration=i_iter_loop, key_var_imputed_for_path=key_var_imputed_iter)
        safe_create_directory(os.path.dirname(sim_filepath_iter)) 
        
        df_to_save_sim = data_with_missing_sim_iter.copy()
        if df_to_save_sim.index.name == Config.ID_COLUMN:
            if Config.ID_COLUMN in df_to_save_sim.columns: df_to_save_sim = df_to_save_sim.reset_index(drop=True)
            else: df_to_save_sim = df_to_save_sim.reset_index()
        df_to_save_sim.to_csv(sim_filepath_iter, index=False)

        imputation_handler_iter = ImputationPipeline(
            data_with_missing_sim_iter, full_data_for_sim_iter.copy(), 
            miss_level_iter, mechanism_iter, key_var_imputed_iter, i_iter_loop
        )
        current_level_imputation_outputs_iter = imputation_handler_iter.run_all_imputations_and_save()

        for method_key_iter, imputed_output_iter in current_level_imputation_outputs_iter.items():
            method_display_name_iter = Config.METHOD_DISPLAY_NAMES.get(method_key_iter, method_key_iter)
            if imputed_output_iter is None: 
                continue 

            if key_var_imputed_iter in Config.KEY_VARS_FOR_STATS_TABLE:
                df_for_stats_iter = imputed_output_iter[0] if isinstance(imputed_output_iter, list) and imputed_output_iter else \
                                   (imputed_output_iter if isinstance(imputed_output_iter, pd.DataFrame) else None)
                if df_for_stats_iter is not None and key_var_imputed_iter in df_for_stats_iter.columns:
                    var_series_imputed_iter = pd.to_numeric(df_for_stats_iter[key_var_imputed_iter], errors='coerce').dropna()
                    if not var_series_imputed_iter.empty:
                        stats_entry = iteration_results_payload['stats_features_updates_iter'][key_var_imputed_iter][mechanism_iter][level_str_iter][method_display_name_iter]
                        stats_entry['variances'].append(var_series_imputed_iter.var())
                        stats_entry['skewnesses'].append(var_series_imputed_iter.skew())
            
            for model_key_reg_iter, formula_reg_iter in Config.MODEL_FORMULAS.items():
                test_results_reg = None
                if method_key_iter == "custom_multiple_imputation":
                    if imputed_output_iter and isinstance(imputed_output_iter, list) and all(isinstance(df, pd.DataFrame) for df in imputed_output_iter):
                        test_results_reg = run_pooled_regression(imputed_output_iter, formula_reg_iter, model_key_reg_iter, 
                                                                 Config.MODEL_FAMILIES.get(model_key_reg_iter), 
                                                                 full_data_for_sim_iter.shape[0])
                else: 
                    if isinstance(imputed_output_iter, pd.DataFrame) and not imputed_output_iter.empty:
                        test_results_reg = safe_run_regression(formula_reg_iter, imputed_output_iter, model_key_reg_iter,
                                                               Config.MODEL_FAMILIES.get(model_key_reg_iter))
                    elif isinstance(imputed_output_iter, pd.DataFrame) and imputed_output_iter.empty and method_key_iter == "listwise_deletion":
                        test_results_reg = None 
                
                if test_results_reg: 
                    reg_txt_dir = Path(Config.REGRESSION_OUTPUT_DIR_TXT) / mechanism_iter / level_str_iter / key_var_imputed_iter / method_key_iter
                    safe_create_directory(reg_txt_dir)
                    mi_suffix = "_mi_pooled" if method_key_iter == "custom_multiple_imputation" else ""
                    reg_txt_path = reg_txt_dir / f"iter{i_iter_loop}_{model_key_reg_iter}{mi_suffix}.txt"
                    summary_text_to_write = "Error: Could not generate summary text."
                    try:
                        summary_obj_for_txt = None
                        if hasattr(test_results_reg, 'summary') and hasattr(test_results_reg.summary, 'tables'): 
                            summary_obj_for_txt = test_results_reg.summary.tables
                        elif hasattr(test_results_reg, 'summary2'): 
                            summary_obj_for_txt = test_results_reg.summary2().tables
                        
                        if summary_obj_for_txt and len(summary_obj_for_txt) >= 2:
                            summary_text_to_write = f"{summary_obj_for_txt[0].to_string()}\n\n{summary_obj_for_txt[1].to_string()}"
                        elif hasattr(test_results_reg, 'summary') and callable(getattr(test_results_reg, 'summary', None)): 
                            summary_text_to_write = test_results_reg.summary().as_text()
                        else: 
                            summary_text_to_write = f"Params:\n{test_results_reg.params.to_string()}"
                    except Exception as e_sum_gen:
                        summary_text_to_write = f"Error generating summary text: {e_sum_gen}"
                    
                    with open(reg_txt_path, "w", encoding='utf-8') as f_txt_out:
                        f_txt_out.write(f"Iter: {i_iter_loop}, Method: {method_display_name_iter}, Key Var Imputed: {key_var_imputed_iter}\n")
                        f_txt_out.write(f"Mechanism: {mechanism_iter}, Level: {level_str_iter}, Model: {model_key_reg_iter}\n\n")
                        f_txt_out.write(summary_text_to_write)

                baseline_model_for_comp = baseline_models_iter.get(model_key_reg_iter)
                comp_metrics = compare_models_py(baseline_model_for_comp, test_results_reg)
                iteration_results_payload['model_comparison_updates_iter'][model_key_reg_iter][method_display_name_iter].append(comp_metrics)

                if key_var_imputed_iter in Config.KEY_VARS_AND_THEIR_MODEL_COEFS and \
                   model_key_reg_iter in Config.KEY_VARS_AND_THEIR_MODEL_COEFS[key_var_imputed_iter]:
                    tracked_coef_names_iter = Config.KEY_VARS_AND_THEIR_MODEL_COEFS[key_var_imputed_iter][model_key_reg_iter]
                    for tracked_coef_name_single_iter in tracked_coef_names_iter:
                        baseline_coef_data_iter_dict = baseline_coef_info_iter.get(model_key_reg_iter, {}).get(tracked_coef_name_single_iter)
                        current_coef_info_df = get_coef_info_py(test_results_reg)
                        
                        stability_entry = iteration_results_payload['coef_stability_updates_iter'][key_var_imputed_iter][tracked_coef_name_single_iter][mechanism_iter][level_str_iter][method_display_name_iter]
                        stability_entry['total_runs'] += 1
                        
                        if baseline_coef_data_iter_dict and current_coef_info_df is not None and tracked_coef_name_single_iter in current_coef_info_df.index:
                            current_coef_row_series = current_coef_info_df.loc[tracked_coef_name_single_iter]
                            sign_same = (baseline_coef_data_iter_dict['sign'] == current_coef_row_series['sign'])
                            sig_same = (baseline_coef_data_iter_dict['sig'] == current_coef_row_series['sig'])
                            if sign_same and sig_same: stability_entry['both_same_count'] += 1
                            elif sign_same and not sig_same: stability_entry['sign_same_sig_changed_count'] += 1
        
        return recursive_defaultdict_to_dict(iteration_results_payload) 
        
    except Exception as e_outer_par:
        logger.error(f"Error in parallel iteration (Var: {args_tuple_wrapper[0] if len(args_tuple_wrapper) > 0 else 'Unknown'}, Mech: {args_tuple_wrapper[1] if len(args_tuple_wrapper) > 1 else 'Unknown'}, Lvl: {args_tuple_wrapper[2] if len(args_tuple_wrapper) > 2 else 'Unknown'}, Iter: {args_tuple_wrapper[3] if len(args_tuple_wrapper) > 3 else 'Unknown'}): {e_outer_par}", exc_info=True)
        error_dict = {'error': str(e_outer_par)}
        if len(args_tuple_wrapper) >= 4:
            error_dict.update({
                'key_var_imputed': args_tuple_wrapper[0], 
                'mechanism': args_tuple_wrapper[1], 
                'miss_level': args_tuple_wrapper[2], 
                'i_iter': args_tuple_wrapper[3]
            })
        return error_dict

def merge_iteration_results(all_iter_results_list, 
                            coef_stability_results_main, 
                            stats_features_results_main, 
                            model_comparison_results_main):
    for iter_res_payload in all_iter_results_list:
        if not isinstance(iter_res_payload, dict) or 'error' in iter_res_payload:
            err_src = iter_res_payload.get('key_var_imputed', 'UnknownVar')
            logger.warning(f"Skipping merge for erroneous/malformed payload from {err_src}: {iter_res_payload.get('error', 'Malformed')}")
            continue
            
        kv_iter = iter_res_payload['key_var_imputed']
        mech_iter = iter_res_payload['mechanism']
        lvl_iter_str = f"{int(iter_res_payload['miss_level']*100)}%"

        for kv, coef_d in iter_res_payload.get('coef_stability_updates_iter', {}).items():
            for tcn, mech_d_inner in coef_d.items(): # Renamed mech_d to mech_d_inner
                for m, lvl_d in mech_d_inner.items():
                    for ls, meth_d in lvl_d.items():
                        for md_name, stab_data in meth_d.items():
                            target = coef_stability_results_main[kv][tcn][m][ls][md_name]
                            target['both_same_count'] += stab_data.get('both_same_count', 0)
                            target['sign_same_sig_changed_count'] += stab_data.get('sign_same_sig_changed_count', 0)
                            target['total_runs'] += stab_data.get('total_runs', 0)
        
        for kv_stat, mech_d_stat in iter_res_payload.get('stats_features_updates_iter', {}).items():
            for m_stat, lvl_d_stat in mech_d_stat.items():
                for ls_stat, meth_d_stat in lvl_d_stat.items():
                    for md_name_stat, stat_vals in meth_d_stat.items():
                        target_stat = stats_features_results_main[kv_stat][m_stat][ls_stat][md_name_stat]
                        target_stat['variances'].extend(stat_vals.get('variances', []))
                        target_stat['skewnesses'].extend(stat_vals.get('skewnesses', []))
        
        for model_k_comp, meth_d_comp in iter_res_payload.get('model_comparison_updates_iter', {}).items():
            for md_name_comp, comp_list in meth_d_comp.items():
                if not isinstance(comp_list, list):
                    logger.warning(f"Invalid comparison data type for {model_k_comp}/{md_name_comp}: {type(comp_list)}")
                    continue
                model_comparison_results_main[mech_iter][lvl_iter_str][kv_iter][model_k_comp][md_name_comp].extend(comp_list)


def cleanup_iteration_artifacts(mechanism: str, key_var: str, miss_level: float) -> None:
    level_dir_name = f"{int(miss_level * 100)}pct_missing"
    path_to_clean_parent = Path(getattr(Config, f"{mechanism.upper()}_BASE_DIR")) / \
                           f"imputed_for_{key_var}" / level_dir_name

    if not path_to_clean_parent.exists():
        logger.info(f"Cleanup: Path not found, no artifacts to delete: {path_to_clean_parent}")
        return

    logger.info(f"Cleaning up iteration artifacts in: {path_to_clean_parent}")
    cleaned_count = 0
    for iter_folder_path in path_to_clean_parent.glob("iter_*"):
        if iter_folder_path.is_dir():
            try:
                shutil.rmtree(iter_folder_path)
                logger.debug(f"Deleted folder and its contents: {iter_folder_path}")
                cleaned_count += 1
            except Exception as e:
                logger.error(f"Error deleting folder {iter_folder_path}: {e}")
    
    logger.info(f"Cleanup complete for {path_to_clean_parent}. Deleted {cleaned_count} 'iter_*' subfolders.")
    try: 
        if not any(path_to_clean_parent.iterdir()): 
            path_to_clean_parent.rmdir()
            logger.debug(f"Removed empty level folder: {path_to_clean_parent}")
            key_var_level_path = path_to_clean_parent.parent
            if not any(key_var_level_path.iterdir()):
                key_var_level_path.rmdir()
                logger.debug(f"Removed empty key_var folder: {key_var_level_path}")

    except OSError as e:
        logger.warning(f"Could not remove empty parent folder {path_to_clean_parent} or above: {e}")


# --- Excel Report Generation Functions ---
def create_excel_regression_table(models: Dict[str, Any], model_titles: List[str]) -> pd.DataFrame:
    """Create Excel regression table with improved error handling"""
    all_coef_names = set()
    processed_results = {}
    model_keys_from_dict = list(models.keys())
    aligned_model_titles = {key_model: (model_titles[i] if i < len(model_titles) else f"Model ({i+1})") for i, key_model in enumerate(model_keys_from_dict)}
    
    model_fe_cols_map, model_absorbed_ivs_map, model_weights_col_map, model_cluster_col_map, model_is_linearmodels_map, model_is_r_feols_map = {}, {}, {}, {}, {}, {}

    for key, res_model_fit_fmt in models.items():
        if res_model_fit_fmt is None: 
            continue
        model_fe_cols_map[key] = getattr(res_model_fit_fmt, 'fixed_effects_cols_used', []) 
        model_weights_col_map[key] = getattr(res_model_fit_fmt, 'weights_col_used', None)
        model_cluster_col_map[key] = getattr(res_model_fit_fmt, 'cluster_col_used', None) 
        model_is_linearmodels_map[key] = getattr(res_model_fit_fmt, 'is_linearmodels', False)
        model_is_r_feols_map[key] = getattr(res_model_fit_fmt, 'is_r_feols', False) 
        
        model_key_for_config_lookup = getattr(res_model_fit_fmt, 'model_key_name_for_config', key)
        absorbed_ivs_for_model_type = Config.MODEL_ABSORBED_IV_BY_FE.get(model_key_for_config_lookup, [])
        fes_active = model_is_linearmodels_map[key] or model_is_r_feols_map[key] or \
                     (not model_is_linearmodels_map[key] and not model_is_r_feols_map[key] and model_fe_cols_map.get(key))
        model_absorbed_ivs_map[key] = absorbed_ivs_for_model_type if fes_active else []

    for key, res_model_fit in models.items():
        if res_model_fit is None: 
            processed_results[key] = {
                'params': pd.Series(dtype=float), 'pvalues': pd.Series(dtype=float), 'std_err': pd.Series(dtype=float), 
                'nobs': 'Error', 'rsquared': 'Error', 'rsquared_adj': 'Error', 'dep_var': 'N/A',
                'fixed_effects_display': 'N/A', 'cluster_display': 'N/A', 'weights_display': 'N/A'
            }
            continue
            
        try:
            summary_tables, dep_var_name_fmt = None, "N/A"
            
            # Get summary tables safely
            if (model_is_linearmodels_map[key] or model_is_r_feols_map[key]):
                if hasattr(res_model_fit, 'summary') and hasattr(res_model_fit.summary, 'tables'):
                    summary_tables = res_model_fit.summary.tables
                    if summary_tables and len(summary_tables) > 0 and isinstance(summary_tables[0], pd.DataFrame):
                        if 'Dep. Variable:' in summary_tables[0].index:
                            dep_var_name_fmt = str(summary_tables[0].loc['Dep. Variable:', 'Value'])
                        elif hasattr(res_model_fit, 'original_formula_str'):
                            dep_var_name_fmt = res_model_fit.original_formula_str.split("~")[0].strip()
                    elif hasattr(res_model_fit, 'original_formula_str'):
                        dep_var_name_fmt = res_model_fit.original_formula_str.split("~")[0].strip()
                else: 
                    dep_var_name_fmt = res_model_fit.original_formula_str.split("~")[0].strip() if hasattr(res_model_fit, 'original_formula_str') else "N/A"
                    
            elif isinstance(res_model_fit, PooledRegressionResults): 
                summary_obj_fmt = res_model_fit.summary2() 
                if summary_obj_fmt and hasattr(summary_obj_fmt, 'tables'): 
                    summary_tables = summary_obj_fmt.tables
                dep_var_name_fmt = res_model_fit.model_formula.split("~")[0].strip() if hasattr(res_model_fit, 'model_formula') and res_model_fit.model_formula else "N/A"
                
            elif hasattr(res_model_fit, 'summary2'): 
                summary_obj_fmt = res_model_fit.summary2()
                if summary_obj_fmt and hasattr(summary_obj_fmt, 'tables'): 
                    summary_tables = summary_obj_fmt.tables
                if hasattr(res_model_fit, 'model') and hasattr(res_model_fit.model, 'endog_names'):
                    dep_var_name_fmt = str(res_model_fit.model.endog_names)
                elif hasattr(res_model_fit, 'original_formula_str'):
                    dep_var_name_fmt = res_model_fit.original_formula_str.split("~")[0].strip()
            else: 
                logger.warning(f"Excel Reg Table: Model '{key}' has unknown summary structure.")
                dep_var_name_fmt = res_model_fit.original_formula_str.split("~")[0].strip() if hasattr(res_model_fit, 'original_formula_str') else "N/A"
                
                # Try direct attribute access
                summary_df_coeffs_direct = None
                if hasattr(res_model_fit, 'params') and hasattr(res_model_fit, 'pvalues'):
                    _params = res_model_fit.params
                    _pvals = res_model_fit.pvalues
                    _stderr = getattr(res_model_fit, 'bse', getattr(res_model_fit, 'std_errors', None))
                    if _params is not None and _pvals is not None and _stderr is not None:
                        summary_df_coeffs_direct = pd.DataFrame({'params': _params, 'pvalues': _pvals, 'std_err': _stderr})
                        
                if summary_df_coeffs_direct is None: 
                    raise ValueError("Model attributes for direct param extraction missing or failed.")

            # Process summary tables
            if summary_tables is None or len(summary_tables) < 2 or not isinstance(summary_tables[1], pd.DataFrame): 
                if 'summary_df_coeffs_direct' in locals() and isinstance(summary_df_coeffs_direct, pd.DataFrame):
                    summary_df_coeffs = summary_df_coeffs_direct.copy()
                else:
                    raise ValueError(f"Invalid summary structure for model {key}.")
            else: 
                summary_df_coeffs = summary_tables[1].copy()

            # Rename columns safely
            rename_map_coeffs = {}
            if 'Estimate' in summary_df_coeffs.columns: 
                rename_map_coeffs['Estimate'] = 'params'
            elif 'Coef.' in summary_df_coeffs.columns: 
                rename_map_coeffs['Coef.'] = 'params'
            if 'Std. Err.' in summary_df_coeffs.columns: 
                rename_map_coeffs['Std. Err.'] = 'std_err'
            elif 'Std.Err.' in summary_df_coeffs.columns: 
                rename_map_coeffs['Std.Err.'] = 'std_err'
            if 'P>|t|' in summary_df_coeffs.columns: 
                rename_map_coeffs['P>|t|'] = 'pvalues'
            elif 'P-value' in summary_df_coeffs.columns: 
                rename_map_coeffs['P-value'] = 'pvalues'
            elif 'Pr(>|t|)' in summary_df_coeffs.columns: 
                rename_map_coeffs['Pr(>|t|)'] = 'pvalues'
            elif 'P>|z|' in summary_df_coeffs.columns: 
                rename_map_coeffs['P>|z|'] = 'pvalues'
            # 添加置信区间的列名映射
            if '[0.025' in summary_df_coeffs.columns: 
                rename_map_coeffs['[0.025'] = 'conf_low'
            elif 'conf.low' in summary_df_coeffs.columns: 
                rename_map_coeffs['conf.low'] = 'conf_low'
            if '0.975]' in summary_df_coeffs.columns: 
                rename_map_coeffs['0.975]'] = 'conf_high'
            elif 'conf.high' in summary_df_coeffs.columns: 
                rename_map_coeffs['conf.high'] = 'conf_high'
            summary_df_coeffs = summary_df_coeffs.rename(columns=rename_map_coeffs)

            # Ensure required columns exist
            for col_check in ['params', 'std_err', 'pvalues', 'conf_low', 'conf_high']:
                if col_check not in summary_df_coeffs.columns:
                    if col_check == 'params' and hasattr(res_model_fit, 'params'): 
                        summary_df_coeffs[col_check] = res_model_fit.params
                    elif col_check == 'std_err' and hasattr(res_model_fit, 'bse'): 
                        summary_df_coeffs[col_check] = res_model_fit.bse
                    elif col_check == 'std_err' and hasattr(res_model_fit, 'std_errors'): 
                        summary_df_coeffs[col_check] = res_model_fit.std_errors
                    elif col_check == 'pvalues' and hasattr(res_model_fit, 'pvalues'): 
                        summary_df_coeffs[col_check] = res_model_fit.pvalues
                    elif col_check == 'conf_low' and hasattr(res_model_fit, 'conf_low'): 
                        summary_df_coeffs[col_check] = res_model_fit.conf_low
                    elif col_check == 'conf_high' and hasattr(res_model_fit, 'conf_high'): 
                        summary_df_coeffs[col_check] = res_model_fit.conf_high
                    else: 
                        summary_df_coeffs[col_check] = np.nan
            
            # Clean coefficient names
            summary_df_coeffs.index = [clean_coef_name_for_html(idx) for idx in summary_df_coeffs.index]
            
            # Filter substantive coefficients
            substantive_coeffs_for_display_list = []
            current_model_fe_cols_list_fmt = model_fe_cols_map.get(key, [])
            fe_prefixes_for_filter_fmt = []
            if not model_is_linearmodels_map[key] and not model_is_r_feols_map[key]:
                fe_prefixes_for_filter_fmt = [f"{clean_coef_name_for_html(fe_col_name_fmt)}[" for fe_col_name_fmt in current_model_fe_cols_list_fmt if fe_col_name_fmt]
            current_absorbed_ivs_list_fmt = [clean_coef_name_for_html(absorbed_iv) for absorbed_iv in model_absorbed_ivs_map.get(key, [])]
            
            for c_name_fmt in summary_df_coeffs.index:
                is_fe_dummy_fmt = any(c_name_fmt.startswith(fe_prefix_fmt) for fe_prefix_fmt in fe_prefixes_for_filter_fmt)
                is_absorbed_iv_fmt = c_name_fmt in current_absorbed_ivs_list_fmt
                if not is_fe_dummy_fmt and not is_absorbed_iv_fmt:
                    substantive_coeffs_for_display_list.append(c_name_fmt)
            all_coef_names.update(substantive_coeffs_for_display_list)
            
            # Get model statistics
            nobs_val_fmt = str(int(res_model_fit.nobs)) if hasattr(res_model_fit, 'nobs') and pd.notna(res_model_fit.nobs) else 'N/A'
            rsq_val_fmt, rsq_adj_val_fmt = 'N/A', 'N/A'
            
            if not isinstance(res_model_fit, PooledRegressionResults): 
                if hasattr(res_model_fit, 'rsquared') and pd.notna(res_model_fit.rsquared): 
                    rsq_val_fmt = f"{res_model_fit.rsquared:.3f}"
                if model_is_r_feols_map[key] and hasattr(res_model_fit, 'rsquared_within') and pd.notna(res_model_fit.rsquared_within):
                    rsq_val_fmt += f" (within: {res_model_fit.rsquared_within:.3f})"
                if hasattr(res_model_fit, 'rsquared_adj') and pd.notna(res_model_fit.rsquared_adj): 
                    rsq_adj_val_fmt = f"{res_model_fit.rsquared_adj:.3f}"
            
            # Create display strings for model features
            fe_disp_str = "No"
            cluster_disp_str = "No"
            weights_disp_str = "No"
            
            fes_used_display_val = model_fe_cols_map.get(key, [])
            if (model_is_linearmodels_map.get(key) or model_is_r_feols_map.get(key)):
                if getattr(res_model_fit, 'entity_effects', False) or (fes_used_display_val and len(fes_used_display_val) > 0):
                    fe_disp_str = f"Yes ({', '.join(fes_used_display_val)})" if fes_used_display_val else "Yes"
            elif fes_used_display_val: 
                fe_disp_str = f"Yes ({', '.join(fes_used_display_val)})"
                
            if hasattr(res_model_fit, 'cov_type'):
                cov_type_obj_val = res_model_fit.cov_type
                cov_type_str_from_obj_val = str(cov_type_obj_val).lower()
                if 'cluster' in cov_type_str_from_obj_val:
                    cluster_var_disp_val = model_cluster_col_map.get(key, 'group')
                    cluster_disp_str = f"Yes ({cluster_var_disp_val})"
                    
            weights_col_used_disp_val = model_weights_col_map.get(key)
            if weights_col_used_disp_val: 
                weights_disp_str = f"Yes ({weights_col_used_disp_val})"

            processed_results[key] = {
                'params': summary_df_coeffs['params'], 'pvalues': summary_df_coeffs['pvalues'], 'std_err': summary_df_coeffs['std_err'], 
                'conf_low': summary_df_coeffs['conf_low'], 'conf_high': summary_df_coeffs['conf_high'],
                'nobs': nobs_val_fmt, 'rsquared': rsq_val_fmt, 'rsquared_adj': rsq_adj_val_fmt, 'dep_var': dep_var_name_fmt,
                'fixed_effects_display': fe_disp_str, 'cluster_display': cluster_disp_str, 'weights_display': weights_disp_str
            }
            
        except Exception as e_fmt_excel: 
            logger.error(f"Excel Reg Table Error creating processed_results for Model '{key}': {e_fmt_excel}", exc_info=True)
            processed_results[key] = {
                'params': pd.Series(dtype=float), 'pvalues': pd.Series(dtype=float), 'std_err': pd.Series(dtype=float), 
                'conf_low': pd.Series(dtype=float), 'conf_high': pd.Series(dtype=float),
                'nobs': 'Error Proc.', 'rsquared': 'Error', 'rsquared_adj': 'Error', 'dep_var': 'Error',
                'fixed_effects_display': 'Error', 'cluster_display': 'Error', 'weights_display': 'Error'
            }

    # Order coefficients for display
    coefs_to_display_ordered = []
    if 'Intercept' in all_coef_names: 
        coefs_to_display_ordered.append('Intercept')
    if Config.MAIN_INTERACTION_TERM_FOR_TRACKING in all_coef_names:
        coefs_to_display_ordered.append(Config.MAIN_INTERACTION_TERM_FOR_TRACKING)
    santamaria_ivs = ["Post", "Treatment"] 
    for iv in santamaria_ivs:
        if iv in all_coef_names and iv not in coefs_to_display_ordered: 
            coefs_to_display_ordered.append(iv)
    interaction_coefs_list = sorted([c for c in all_coef_names if ':' in c and c not in coefs_to_display_ordered], key=str.lower)
    coefs_to_display_ordered.extend(interaction_coefs_list)
    other_main_effects = sorted([c for c in all_coef_names if c not in coefs_to_display_ordered and ':' not in c], key=str.lower)
    coefs_to_display_ordered.extend(other_main_effects)
    # Remove duplicates while preserving order
    coefs_to_display_ordered = [c for i, c in enumerate(coefs_to_display_ordered) if c not in coefs_to_display_ordered[:i]]

    # Create MultiIndex columns
    columns_multi_index = pd.MultiIndex.from_tuples(
        [(aligned_model_titles.get(mk, f"Model({i+1})"), 
          f"DV: {processed_results.get(mk, {}).get('dep_var', 'N/A')}") for i, mk in enumerate(model_keys_from_dict)],
        names=['Model', 'Details']
    )
    df_excel_table = pd.DataFrame(index=coefs_to_display_ordered, columns=columns_multi_index, dtype=str)

    # Fill coefficient table
    for coef_name_row in coefs_to_display_ordered:
        for i_mk, model_key_cell in enumerate(model_keys_from_dict):
            col_tuple_excel = (aligned_model_titles.get(model_key_cell, f"Model({i_mk+1})"), 
                               f"DV: {processed_results.get(model_key_cell, {}).get('dep_var', 'N/A')}")
            res_data_cell = processed_results.get(model_key_cell, {})
            
            # Safe coefficient access
            params_series = res_data_cell.get('params', pd.Series(dtype=float))
            pvalues_series = res_data_cell.get('pvalues', pd.Series(dtype=float))
            stderr_series = res_data_cell.get('std_err', pd.Series(dtype=float))
            conf_low_series = res_data_cell.get('conf_low', pd.Series(dtype=float))
            conf_high_series = res_data_cell.get('conf_high', pd.Series(dtype=float))
            
            param_val_cell = params_series.get(coef_name_row, np.nan) if hasattr(params_series, 'get') else np.nan
            pval_val_cell = pvalues_series.get(coef_name_row, np.nan) if hasattr(pvalues_series, 'get') else np.nan
            stderr_val_cell = stderr_series.get(coef_name_row, np.nan) if hasattr(stderr_series, 'get') else np.nan
            conf_low_val_cell = conf_low_series.get(coef_name_row, np.nan) if hasattr(conf_low_series, 'get') else np.nan
            conf_high_val_cell = conf_high_series.get(coef_name_row, np.nan) if hasattr(conf_high_series, 'get') else np.nan
            
            cell_content_str = ""
            if not (pd.isna(param_val_cell) or pd.isna(stderr_val_cell)):
                stars_str = ""
                if pd.notna(pval_val_cell):
                    if pval_val_cell < 0.001: stars_str = "***"
                    elif pval_val_cell < 0.01: stars_str = "**"
                    elif pval_val_cell < 0.05: stars_str = "*"
                
                # 添加置信区间信息
                ci_str = ""
                if pd.notna(conf_low_val_cell) and pd.notna(conf_high_val_cell):
                    ci_str = f"\n[{conf_low_val_cell:.3f}, {conf_high_val_cell:.3f}]"
                
                cell_content_str = f"{param_val_cell:.3f}{stars_str}\n({stderr_val_cell:.3f}){ci_str}"
            elif pd.notna(param_val_cell):
                cell_content_str = f"{param_val_cell:.3f}"
            df_excel_table.loc[coef_name_row, col_tuple_excel] = cell_content_str
    
    # Create footer statistics
    footer_stats_map = {
        'Observations': 'nobs', 'R-squared': 'rsquared', 'Adj. R-squared': 'rsquared_adj',
        'Fixed Effects': 'fixed_effects_display', 'Clustered SE': 'cluster_display', 'Weights': 'weights_display'
    }
    footer_df_excel = pd.DataFrame(index=list(footer_stats_map.keys()), columns=columns_multi_index, dtype=str)
    for stat_label_footer, stat_key_footer in footer_stats_map.items():
        for i_mk_f, model_key_footer_stat in enumerate(model_keys_from_dict):
            col_tuple_footer_excel = (aligned_model_titles.get(model_key_footer_stat, f"Model({i_mk_f+1})"), 
                                      f"DV: {processed_results.get(model_key_footer_stat, {}).get('dep_var', 'N/A')}")
            footer_df_excel.loc[stat_label_footer, col_tuple_footer_excel] = processed_results.get(model_key_footer_stat, {}).get(stat_key_footer, '')

    final_df_for_excel = pd.concat([df_excel_table, footer_df_excel])
    final_df_for_excel.index.name = "Variable"
    return final_df_for_excel.fillna('')


def create_excel_model_comparison_table(model_comparison_results_agg: dict) -> pd.DataFrame:
    comparison_data_for_excel = []
    for mechanism_comp, levels_data_comp in model_comparison_results_agg.items():
        for level_str_comp, key_vars_data_comp in levels_data_comp.items():
            for key_var_comp, models_data_comp in key_vars_data_comp.items():
                for model_key_comp, methods_data_comp in models_data_comp.items():
                    for method_name_comp, metrics_list_comp in methods_data_comp.items():
                        if metrics_list_comp:
                            avg_rmse = np.nanmean([m.get('rmse', np.nan) for m in metrics_list_comp])
                            avg_rel_se = np.nanmean([m.get('avg_rel_se', np.nan) for m in metrics_list_comp])
                            # Simplified "any changed" logic
                            any_sig_changed = any("None" not in m.get('vars_sig_changed', ["None"]) and not any(err_token in str(m.get('vars_sig_changed', ["None"])[0]) for err_token in ["Error", "No Coefs"]) for m in metrics_list_comp if m.get('vars_sig_changed') and m.get('vars_sig_changed', ["None"]) != ["Model Error"] and m.get('vars_sig_changed', ["None"]) != ["Baseline Model Error"] and m.get('vars_sig_changed', ["None"]) != ["Test Model Error"] and m.get('vars_sig_changed', ["None"]) != ["No Common Coefs"] and m.get('vars_sig_changed', ["None"]) != ["Baseline Coef Info Error"] and m.get('vars_sig_changed', ["None"]) != ["Test Coef Info Error"] and m.get('vars_sig_changed', ["None"]) != ["No Valid Common Coefs"])
                            any_sign_changed = any("None" not in m.get('vars_sign_changed', ["None"]) and not any(err_token in str(m.get('vars_sign_changed', ["None"])[0]) for err_token in ["Error", "No Coefs"]) for m in metrics_list_comp if m.get('vars_sign_changed') and m.get('vars_sign_changed', ["None"]) != ["Model Error"] and m.get('vars_sign_changed', ["None"]) != ["Baseline Model Error"] and m.get('vars_sign_changed', ["None"]) != ["Test Model Error"] and m.get('vars_sign_changed', ["None"]) != ["No Common Coefs"] and m.get('vars_sign_changed', ["None"]) != ["Baseline Coef Info Error"] and m.get('vars_sign_changed', ["None"]) != ["Test Coef Info Error"] and m.get('vars_sign_changed', ["None"]) != ["No Valid Common Coefs"])
                            avg_common_vars = np.nanmean([m.get('common_vars_count', 0) for m in metrics_list_comp])

                            comparison_data_for_excel.append({
                                "Mechanism": mechanism_comp, "Level": level_str_comp,
                                "Imputed Var": key_var_comp, "Model": model_key_comp,
                                "Method": method_name_comp, "Avg RMSE": f"{avg_rmse:.3f}" if pd.notna(avg_rmse) else "N/A",
                                "Avg Rel SE": f"{avg_rel_se:.3f}" if pd.notna(avg_rel_se) else "N/A",
                                "Sig Changed?": "Yes" if any_sig_changed else "No",
                                "Sign Changed?": "Yes" if any_sign_changed else "No",
                                "Avg Common Vars": f"{avg_common_vars:.1f}" if pd.notna(avg_common_vars) else "N/A"
                            })
    if comparison_data_for_excel:
        return pd.DataFrame(comparison_data_for_excel)
    return pd.DataFrame()

def create_excel_stats_features_table(stats_features_results_agg: dict) -> pd.DataFrame:
    stats_features_data_for_excel = []
    for key_var_stat, mechs_data_stat in stats_features_results_agg.items():
        for mech_stat, levels_data_stat in mechs_data_stat.items():
            for level_str_stat, methods_data_stat in levels_data_stat.items():
                for method_name_stat, values_stat in methods_data_stat.items():
                    if values_stat.get('variances') or values_stat.get('skewnesses'):
                        avg_var = np.nanmean(values_stat['variances']) if values_stat.get('variances') else np.nan
                        avg_skew = np.nanmean(values_stat['skewnesses']) if values_stat.get('skewnesses') else np.nan
                        stats_features_data_for_excel.append({
                            "Imputed Var": key_var_stat, "Mechanism": mech_stat, "Level": level_str_stat,
                            "Method": method_name_stat, 
                            "Avg Variance": f"{avg_var:.3f}" if pd.notna(avg_var) else "N/A",
                            "Avg Skewness": f"{avg_skew:.3f}" if pd.notna(avg_skew) else "N/A"
                        })
    if stats_features_data_for_excel:
        return pd.DataFrame(stats_features_data_for_excel)
    return pd.DataFrame()

def _compute_missingness_patterns_table(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    cols_present = [c for c in cols if c in df.columns]
    if not cols_present:
        return pd.DataFrame(columns=['Section', 'Name', 'Value'])
    var_rates = df[cols_present].isna().mean().rename('Value').reset_index().rename(columns={'index': 'Name'})
    var_rates.insert(0, 'Section', 'VarRate')
    miss_matrix = df[cols_present].isna().astype(int)
    try:
        pattern_series = miss_matrix.apply(lambda row: ''.join(row.astype(str).tolist()), axis=1)
        pattern_counts = pattern_series.value_counts().rename('Value').reset_index().rename(columns={'index': 'Name'})
        pattern_counts.insert(0, 'Section', 'Pattern')
        return pd.concat([var_rates, pattern_counts], ignore_index=True)
    except Exception:
        return var_rates

def _fit_models_on_imputed_list(imputed_list: List[pd.DataFrame], formula: str, model_key: str) -> List[Any]:
    results: List[Any] = []
    for df_imp in imputed_list:
        try:
            res = safe_run_regression(formula=formula, data=df_imp, model_key=model_key, family=Config.MODEL_FAMILIES.get(model_key))
            if res is not None:
                results.append(res)
        except Exception:
            continue
    return results

def _compute_mi_param_diagnostics(model_results_list: List[Any]) -> Tuple[pd.DataFrame, Dict[str, List[float]], Dict[str, List[float]]]:
    if not model_results_list:
        return pd.DataFrame(columns=['Coef', 'Qbar', 'Ubar', 'B', 'T', 'r', 'FMI', 'RE', 'df_BR', 'SE', 'MCSE', 'MCSE_over_SE', 'm_suggest_99RE', 'Flag_OK', 'm']), {}, {}
    m = len(model_results_list)
    all_names = set()
    for res in model_results_list:
        params = getattr(res, 'params', pd.Series(dtype='float64'))
        if isinstance(params, pd.Series):
            all_names.update(list(params.index))
    all_names = sorted(list(all_names))

    traces: Dict[str, List[float]] = {name: [] for name in all_names}
    within_vars: Dict[str, List[float]] = {name: [] for name in all_names}

    for res in model_results_list:
        params = getattr(res, 'params', pd.Series(dtype='float64'))
        cov = getattr(res, 'cov', None)
        for name in all_names:
            q = params.get(name, np.nan) if isinstance(params, pd.Series) else np.nan
            traces[name].append(q)
            if isinstance(cov, pd.DataFrame) and name in cov.index and name in cov.columns:
                within_vars[name].append(float(cov.loc[name, name]))
            else:
                within_vars[name].append(np.nan)

    rows = []
    for name in all_names:
        q_vals = np.array(traces[name], dtype=float)
        u_vals = np.array(within_vars[name], dtype=float)
        valid_mask = ~np.isnan(q_vals)
        q_use = q_vals[valid_mask]
        u_use = u_vals[valid_mask]
        m_eff = len(q_use)
        if m_eff == 0:
            continue
        Qbar = float(np.nanmean(q_use))
        Ubar = float(np.nanmean(u_use)) if m_eff > 0 else np.nan
        B = float(np.nanvar(q_use, ddof=1)) if m_eff > 1 else 0.0
        T = float((Ubar if np.isfinite(Ubar) else 0.0) + (1.0 + 1.0/max(1, m_eff)) * B)
        r = np.nan
        if np.isfinite(Ubar) and Ubar > 0:
            r = ((1.0 + 1.0/max(1, m_eff)) * B) / Ubar
        FMI = r/(1.0 + r) if (isinstance(r, (float, np.floating)) and r >= 0 and np.isfinite(r)) else np.nan
        RE = 1.0 / (1.0 + (FMI/max(1, m_eff))) if (isinstance(FMI, (float, np.floating)) and np.isfinite(FMI)) else np.nan
        df_BR = np.nan
        if m_eff > 1 and np.isfinite(Ubar) and Ubar > 0:
            riv = ((1.0 + 1.0/m_eff) * B) / Ubar
            if np.isfinite(riv) and riv >= 0:
                df_BR = (m_eff - 1.0) * (1.0 + 1.0/riv)**2 if riv > 0 else float('inf')
        SE = float(np.sqrt(T)) if np.isfinite(T) and T >= 0 else np.nan
        var_Q = float(np.nanvar(q_use, ddof=1)) if m_eff > 1 else 0.0
        MCSE = float(np.sqrt(var_Q / max(1, m_eff)))
        MCSE_over_SE = (MCSE / SE) if (SE and np.isfinite(SE) and SE > 0) else np.nan
        target_inv_minus_1 = (1.0/0.99) - 1.0
        m_suggest = np.nan
        if isinstance(FMI, (float, np.floating)) and np.isfinite(FMI) and FMI >= 0 and target_inv_minus_1 > 0:
            m_suggest = int(np.ceil(FMI / target_inv_minus_1))
        flags = []
        if (RE is not np.nan) and (not np.isnan(RE)) and RE < 0.99:
            flags.append("RE<0.99")
        if (MCSE_over_SE is not np.nan) and (not np.isnan(MCSE_over_SE)) and MCSE_over_SE >= 0.10:
            flags.append("MCSE/SE≥0.10")
        flag_ok = "OK" if not flags else "; ".join(flags)
        rows.append({'Coef': name, 'Qbar': Qbar, 'Ubar': Ubar, 'B': B, 'T': T, 'r': r, 'FMI': FMI, 'RE': RE,
                     'df_BR': df_BR, 'SE': SE, 'MCSE': MCSE, 'MCSE_over_SE': MCSE_over_SE,
                     'm_suggest_99RE': m_suggest, 'Flag_OK': flag_ok, 'm': m_eff})

    diag_df = pd.DataFrame(rows)
    return diag_df, traces, within_vars

def _overimputation_metrics_for_var(df_complete: pd.DataFrame, var: str, mechanism: str, level: float,
                                     model_formula: str, model_key: str) -> Optional[Dict[str, Any]]:
    if var not in df_complete.columns:
        return None
    series = pd.to_numeric(df_complete[var], errors='coerce')
    obs_idx = series.dropna().index
    if len(obs_idx) < 10:
        return None
    rng = np.random.default_rng(Config.SIMULATION_SEED + 2025 + hash(var) % 1000)
    test_n = max(1, int(0.1 * len(obs_idx)))
    test_idx = rng.choice(obs_idx.to_numpy(), size=test_n, replace=False)
    df_masked = df_complete.copy()
    df_masked.loc[test_idx, var] = np.nan
    imp = ImputationPipeline(df_masked, df_complete.copy(), missingness_level_config=level,
                             mechanism_config=mechanism, current_key_var_imputed=var, iteration_num=0)
    mi_list = imp.custom_multiple_imputation()
    if not mi_list:
        return None
    preds_matrix = []
    for df_m in mi_list:
        preds_matrix.append(pd.to_numeric(df_m.loc[test_idx, var], errors='coerce').to_numpy())
    preds_matrix = np.vstack(preds_matrix) if len(preds_matrix) > 0 else np.empty((0, len(test_idx)))
    if preds_matrix.size == 0:
        return None
    pred_mean = np.nanmean(preds_matrix, axis=0)
    pred_sd = np.nanstd(preds_matrix, axis=0, ddof=1)
    y_true = pd.to_numeric(df_complete.loc[test_idx, var], errors='coerce').to_numpy()
    rmse = float(np.sqrt(np.nanmean((pred_mean - y_true)**2)))
    mae = float(np.nanmean(np.abs(pred_mean - y_true)))
    z = 1.96
    covered = np.logical_and(y_true >= pred_mean - z*pred_sd, y_true <= pred_mean + z*pred_sd)
    coverage = float(np.nanmean(covered.astype(float))) * 100.0
    std_pred_error = (y_true - pred_mean) / np.where(pred_sd > 1e-12, pred_sd, np.nan)
    std_pred_error_mean = float(np.nanmean(std_pred_error))
    def _smd(a: np.ndarray, b: np.ndarray) -> float:
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        if a.size == 0 or b.size == 0: return float('nan')
        sd_pooled = np.sqrt(((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0))
        if sd_pooled <= 0 or np.isnan(sd_pooled): return float('nan')
        return (np.mean(a) - np.mean(b)) / sd_pooled
    smd_val = float(_smd(y_true, pred_mean))
    vr = float(np.var(pred_mean, ddof=1) / np.var(y_true, ddof=1)) if np.var(y_true, ddof=1) > 0 else float('nan')
    return {
        'DatasetID': 'Santamaria2024', 'Var': var, 'Method': Config.METHOD_DISPLAY_NAMES.get('custom_multiple_imputation', 'MI'),
        'Mechanism': mechanism, 'Level': f"{int(level*100)}%", 'RMSE': rmse, 'MAE': mae,
        'Coverage95%': coverage, 'StdPredError_Mean': std_pred_error_mean, 'SMD': smd_val, 'VarRatio': vr, 'n_test': int(test_n)
    }

def _distribution_compare_for_var(df_with_missing: pd.DataFrame, df_imputed_avg: pd.DataFrame, var: str,
                                  mechanism: str, level: float) -> Optional[Dict[str, Any]]:
    if var not in df_with_missing.columns or var not in df_imputed_avg.columns:
        return None
    miss_mask = df_with_missing[var].isna()
    if not miss_mask.any():
        return None
    imp_vals = pd.to_numeric(df_imputed_avg.loc[miss_mask, var], errors='coerce').to_numpy()
    obs_vals = pd.to_numeric(df_with_missing.loc[~miss_mask, var], errors='coerce').to_numpy()
    if imp_vals.size == 0 or obs_vals.size == 0:
        return None
    mean_obs, var_obs = float(np.nanmean(obs_vals)), float(np.nanvar(obs_vals, ddof=1))
    mean_imp, var_imp = float(np.nanmean(imp_vals)), float(np.nanvar(imp_vals, ddof=1))
    def _smd(a: np.ndarray, b: np.ndarray) -> float:
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        if a.size == 0 or b.size == 0: return float('nan')
        sd_pooled = np.sqrt(((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0))
        if sd_pooled <= 0 or np.isnan(sd_pooled): return float('nan')
        return (np.mean(a) - np.mean(b)) / sd_pooled
    smd_val = float(_smd(obs_vals, imp_vals))
    vr = float(var_imp / var_obs) if var_obs > 0 else float('nan')
    ks_p = float(ks_2samp(obs_vals[~np.isnan(obs_vals)], imp_vals[~np.isnan(imp_vals)]).pvalue)
    return {'Var': var, 'Method': Config.METHOD_DISPLAY_NAMES.get('custom_multiple_imputation', 'MI'),
            'Mechanism': mechanism, 'Level': f"{int(level*100)}%",
            'Mean_obs': mean_obs, 'Mean_imp': mean_imp, 'SMD': smd_val,
            'Var_obs': var_obs, 'Var_imp': var_imp, 'VR': vr, 'KS_p': ks_p}

def _nmar_residual_association(df_with_missing: pd.DataFrame, var: str, model_formula: str) -> Optional[Dict[str, Any]]:
    try:
        dep, rhs = model_formula.split('~')
        terms = [t.strip() for t in re.split(r'\+|\*', rhs) if t.strip()]
        X_cols = sorted(list(set([re.sub(r'C\(([^,)]+).*\)', r'\1', t) for t in terms])))
        df_work = df_with_missing.copy()
        df_work[var] = pd.to_numeric(df_work[var], errors='coerce')
        M = df_work[var].isna().astype(int)
        obs_mask = df_work[var].notna()
        if obs_mask.sum() < 10: return None
        X = df_work.loc[obs_mask, X_cols].copy()
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]): X[c] = pd.to_numeric(X[c], errors='coerce')
        X = sm.add_constant(X, has_constant='add')
        y = df_work.loc[obs_mask, var]
        ols_res = sm.OLS(y, X, missing='drop').fit()
        ehat = pd.Series(index=df_work.index, dtype=float)
        ehat.loc[obs_mask] = ols_res.resid
        X_all = df_work[X_cols].copy()
        for c in X_all.columns:
            if not pd.api.types.is_numeric_dtype(X_all[c]): X_all[c] = pd.to_numeric(X_all[c], errors='coerce')
            X_all[c] = X_all[c].fillna(0)
        X_all = sm.add_constant(X_all, has_constant='add')
        X_all['ehat'] = ehat.fillna(0)
        logit_mod = sm.Logit(M, X_all)
        logit_res = logit_mod.fit(disp=0, maxiter=100)
        beta_resid = float(logit_res.params.get('ehat', np.nan))
        se_resid = float(logit_res.bse.get('ehat', np.nan))
        pval_resid = float(logit_res.pvalues.get('ehat', np.nan))
        try:
            pred_prob = logit_res.predict()
            auc_val = float(roc_auc_score(M, pred_prob))
        except Exception:
            auc_val = float('nan')
        return {'Model': 'logit', 'beta_resid': beta_resid, 'se': se_resid, 'pval': pval_resid, 'AUC': auc_val}
    except Exception:
        return None

def _nmar_delta_sensitivity_rows(df_with_missing: pd.DataFrame, mi_list: List[pd.DataFrame], var: str,
                                 deltas: List[float], formula: str, model_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not mi_list:
        return rows
    miss_mask = df_with_missing[var].isna() if var in df_with_missing.columns else None
    if miss_mask is None or not miss_mask.any():
        return rows
    base_results = run_pooled_regression(mi_list, formula=formula, model_key=model_key,
                                         family=Config.MODEL_FAMILIES.get(model_key))
    base_params = getattr(base_results, 'params', pd.Series(dtype=float)) if base_results else pd.Series(dtype=float)
    base_pvals = getattr(base_results, 'pvalues', pd.Series(dtype=float)) if base_results else pd.Series(dtype=float)
    imputed_values_stack = []
    for df_m in mi_list:
        imputed_values_stack.append(pd.to_numeric(df_m.loc[miss_mask, var], errors='coerce').to_numpy())
    imputed_values_stack = np.vstack(imputed_values_stack) if len(imputed_values_stack) > 0 else np.empty((0, miss_mask.sum()))
    sd_base = float(np.nanstd(imputed_values_stack, axis=None, ddof=1)) if imputed_values_stack.size > 1 else 0.0
    for delta in deltas:
        adjusted_list = []
        for df_m in mi_list:
            df_adj = df_m.copy()
            if sd_base > 0:
                df_adj.loc[miss_mask, var] = df_adj.loc[miss_mask, var] + delta * sd_base
            adjusted_list.append(df_adj)
        pooled_res = run_pooled_regression(adjusted_list, formula=formula, model_key=model_key,
                                           family=Config.MODEL_FAMILIES.get(model_key))
        if not pooled_res:
            continue
        params = getattr(pooled_res, 'params', pd.Series(dtype=float))
        bse = getattr(pooled_res, 'bse', pd.Series(dtype=float))
        pvals = getattr(pooled_res, 'pvalues', pd.Series(dtype=float))
        for coef_name in params.index:
            se_val = float(bse.get(coef_name, np.nan))
            p_val = float(pvals.get(coef_name, np.nan))
            sign_val = int(np.sign(params.get(coef_name, np.nan)))
            base_sign = int(np.sign(base_params.get(coef_name, np.nan))) if not base_params.empty else 0
            base_sig = float(base_pvals.get(coef_name, np.nan)) if not base_pvals.empty else np.nan
            same_sign = (sign_val == base_sign) if base_sign != 0 and sign_val != 0 else False
            sig_changed = ((p_val < Config.ALPHA) != ((base_sig < Config.ALPHA) if np.isfinite(base_sig) else False)) if np.isfinite(p_val) else False
            rows.append({'Var': var, 'Delta_SD': delta, 'Coef': coef_name, 'SE': se_val, 'p': p_val,
                        'Sign': sign_val, 'B_SameSign': same_sign, 'SS_Changed': sig_changed})
    if rows:
        df_rows = pd.DataFrame(rows)
        tip_map: Dict[Tuple[str, str], float] = {}
        for (v, c), g in df_rows.groupby(['Var', 'Coef']):
            g_sorted = g.sort_values(by='Delta_SD')
            tip = None
            for _, r in g_sorted.iterrows():
                if (not r['B_SameSign']) or r['SS_Changed']:
                    tip = r['Delta_SD']; break
            if tip is not None:
                tip_map[(v, c)] = tip
        for r in rows:
            key = (r['Var'], r['Coef'])
            r['Tip_Delta'] = tip_map.get(key, '')
    return rows

def write_stability_tables_to_excel(writer: pd.ExcelWriter, 
                                    stability_data: dict, 
                                    mechanisms: List[str],
                                    levels: List[float],
                                    methods_display_names: Dict[str, str],
                                    key_vars_iterative_missingness: List[str]
                                    ):
    methods_sorted = sorted(list(methods_display_names.values()))
    level_strs = [f"{int(l*100)}%" for l in levels]

    # Wilson CI setup using Config.ALPHA (matches reference layout with CI)
    alpha = getattr(Config, 'ALPHA', 0.05)
    z = norm.ppf(1 - alpha / 2) if np.isfinite(alpha) else norm.ppf(0.975)

    def wilson_ci_percent(successes: int, trials: int) -> Tuple[float, float]:
        if trials <= 0:
            return (np.nan, np.nan)
        phat = successes / trials
        z_sq = z * z
        denom = 1.0 + z_sq / trials
        center = (phat + z_sq / (2.0 * trials)) / denom
        inner = (phat * (1.0 - phat) / trials) + (z_sq / (4.0 * trials * trials))
        inner = inner if inner > 0.0 else 0.0
        half = (z / denom) * math.sqrt(inner)
        lower = max(0.0, center - half)
        upper = min(1.0, center + half)
        return (lower * 100.0, upper * 100.0)

    for mech in mechanisms:
        all_rows_mech = []
        for method_disp_name in methods_sorted:
            row_data = {'Method': method_disp_name}
            for level_str_iter in level_strs:
                total_both_same_count_for_level_method = 0
                total_sign_same_sig_changed_count_for_level_method = 0
                total_runs_for_level_method = 0

                for key_var_imputed_loop in key_vars_iterative_missingness:
                    if key_var_imputed_loop in stability_data:
                        for tracked_coef_name_loop in stability_data[key_var_imputed_loop]:
                            current_level_data = stability_data[key_var_imputed_loop]\
                                                             [tracked_coef_name_loop]\
                                                             .get(mech, {})\
                                                             .get(level_str_iter, {})\
                                                             .get(method_disp_name)
                            if current_level_data:
                                total_both_same_count_for_level_method += current_level_data.get('both_same_count', 0)
                                total_sign_same_sig_changed_count_for_level_method += current_level_data.get('sign_same_sig_changed_count', 0)
                                total_runs_for_level_method += current_level_data.get('total_runs', 0)
                
                prop_b = (total_both_same_count_for_level_method / total_runs_for_level_method * 100) if total_runs_for_level_method > 0 else 0.0
                prop_ss = (total_sign_same_sig_changed_count_for_level_method / total_runs_for_level_method * 100) if total_runs_for_level_method > 0 else 0.0
                
                # Calculate Wilson confidence intervals
                b_ci_l, b_ci_u = wilson_ci_percent(total_both_same_count_for_level_method, total_runs_for_level_method) if total_runs_for_level_method > 0 else (np.nan, np.nan)
                ss_ci_l, ss_ci_u = wilson_ci_percent(total_sign_same_sig_changed_count_for_level_method, total_runs_for_level_method) if total_runs_for_level_method > 0 else (np.nan, np.nan)
                
                row_data[f'{level_str_iter}_B'] = prop_b
                row_data[f'{level_str_iter}_B_CI_L'] = b_ci_l
                row_data[f'{level_str_iter}_B_CI_U'] = b_ci_u
                row_data[f'{level_str_iter}_SS'] = prop_ss
                row_data[f'{level_str_iter}_SS_CI_L'] = ss_ci_l
                row_data[f'{level_str_iter}_SS_CI_U'] = ss_ci_u
            
            all_rows_mech.append(row_data)
        
        if all_rows_mech:
            summary_df_mech = pd.DataFrame(all_rows_mech).set_index('Method')
            multi_cols_tuples = []
            for l_str in level_strs:
                multi_cols_tuples.append((l_str, 'B'))
                multi_cols_tuples.append((l_str, 'B_CI_L'))
                multi_cols_tuples.append((l_str, 'B_CI_U'))
                multi_cols_tuples.append((l_str, 'SS'))
                multi_cols_tuples.append((l_str, 'SS_CI_L'))
                multi_cols_tuples.append((l_str, 'SS_CI_U'))
            summary_df_mech.columns = pd.MultiIndex.from_tuples(multi_cols_tuples)
            summary_df_mech.to_excel(writer, sheet_name=f'Mean_Stability_{mech}', float_format="%.1f")
        else:
            logger.warning(f"No stability data to write for mechanism: {mech}")


def write_coef_stability_summary(writer: pd.ExcelWriter, stability_data: dict, methods_display_names: Dict[str, str],
                                 mechanisms: List[str], levels: List[float], key_vars_iterative_missingness: List[str]) -> None:
    try:
        rows_css = []
        level_strs = [f"{int(l*100)}%" for l in levels]
        for mechanism in mechanisms:
            for level_str in level_strs:
                for method_disp in sorted(methods_display_names.values()):
                    total_runs = 0
                    both_same = 0
                    sign_same_sig_changed = 0
                    for key_var in key_vars_iterative_missingness:
                        if key_var in stability_data:
                            for coef_name in stability_data[key_var]:
                                counts = stability_data[key_var][coef_name].get(mechanism, {}).get(level_str, {}).get(method_disp)
                                if counts:
                                    total_runs += counts.get('total_runs', 0)
                                    both_same += counts.get('both_same_count', 0)
                                    sign_same_sig_changed += counts.get('sign_same_sig_changed_count', 0)
                    if total_runs > 0:
                        rows_css.append({
                            'Key Var': ','.join(key_vars_iterative_missingness),
                            'Tracked Coef': '',
                            'Mechanism': mechanism,
                            'Level': level_str,
                            'Method': method_disp,
                            'Stability (%)': (both_same/total_runs)*100.0,
                            'Sign Consistency (%)': (sign_same_sig_changed/total_runs)*100.0
                        })
        coef_stab_summary_df = pd.DataFrame(rows_css)
        if not coef_stab_summary_df.empty:
            coef_stab_summary_df.to_excel(writer, sheet_name='Coef_Stability_Summary', index=False)
    except Exception as e_css:
        logger.warning(f"Failed writing Coef_Stability_Summary: {e_css}")


def write_benchmark_tables_to_excel(writer: pd.ExcelWriter, stability_data: dict,
                                    var_groups: Dict[str, List[str]],
                                    levels: List[float],
                                    methods_display_names: Dict[str, str]) -> None:
    try:
        level_strs = [f"{int(l*100)}%" for l in levels]
        methods_sorted = sorted(list(methods_display_names.values()))
        for group_name, var_list in var_groups.items():
            if not var_list:
                continue
            all_rows = []
            for method in methods_sorted:
                row = {'Method': method}
                for level_str in level_strs:
                    both_same_total = 0
                    runs_total = 0
                    for key_var in var_list:
                        if key_var not in stability_data:
                            continue
                        for coef_name in stability_data[key_var]:
                            for mech in ["MCAR", "MAR", "NMAR"]:  # 处理所有三种机制
                                counts = stability_data[key_var][coef_name].get(mech, {}).get(level_str, {}).get(method)
                                if counts:
                                    both_same_total += counts.get('both_same_count', 0)
                                    runs_total += counts.get('total_runs', 0)
                    row[level_str] = (both_same_total/runs_total)*100.0 if runs_total > 0 else np.nan
                all_rows.append(row)
            df_bench = pd.DataFrame(all_rows).set_index('Method')
            df_bench.columns.name = 'Missingness Level (% Sign & Sig Same)'
            df_bench.to_excel(writer, sheet_name=f'Benchmark_{group_name}', float_format="%.1f")
    except Exception as e_bench:
        logger.warning(f"Failed writing benchmark tables: {e_bench}")


# --- Main Analysis Pipeline (run_full_analysis) ---
def run_full_analysis():
    logger.info(f"\nStarting full analysis for Santamaria et al. (2024) setup...")
    Config() 
    
    coef_stability_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'both_same_count': 0, 'sign_same_sig_changed_count': 0, 'total_runs': 0})))))
    stats_features_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'variances': [], 'skewnesses': []}))))
    model_comparison_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
    
    # --- Temp Pickle Directory for Parallel Processing ---
    temp_pickle_dir = Path("temp_pickles_santamaria")  # Santamaria for Santamaria2024
    if Config.USE_PARALLEL:
        temp_pickle_dir.mkdir(exist_ok=True)
    full_data_path_pickle = temp_pickle_dir / "full_data_santamaria.pkl"
    baseline_models_path_pickle = temp_pickle_dir / "baseline_models_santamaria.pkl"
    baseline_coef_info_path_pickle = temp_pickle_dir / "baseline_coef_info_santamaria.pkl"

    logger.info("Loading and preprocessing original data...")
    original_df = pd.read_csv(Config.ORIGINAL_DATA_FILE)
    full_data = preprocess_data(original_df) 

    logger.info("Running baseline models on full (preprocessed) data...")
    baseline_models = {} 
    baseline_coef_info = {} 

    for model_key, formula in Config.MODEL_FORMULAS.items():
        logger.info(f"Running baseline for: {model_key}")
        baseline_result = safe_run_regression(formula, full_data.copy(), model_key) 
        if baseline_result:
            baseline_models[model_key] = baseline_result
            coef_info_df = get_coef_info_py(baseline_result)
            if coef_info_df is not None:
                baseline_coef_info[model_key] = {
                    coef_name: row_data.to_dict() for coef_name, row_data in coef_info_df.iterrows()
                }
            else: baseline_coef_info[model_key] = {}
        else: logger.error(f"Baseline model failed for {model_key}")

    # Save data to pickle files for parallel processing
    if Config.USE_PARALLEL:
        logger.info("Saving data to pickle files for parallel processing...")
        with open(full_data_path_pickle, 'wb') as f:
            pickle.dump(full_data, f)
        with open(baseline_models_path_pickle, 'wb') as f:
            pickle.dump(baseline_models, f)
        with open(baseline_coef_info_path_pickle, 'wb') as f:
            pickle.dump(baseline_coef_info, f)
        # 保存NUMERICAL_COLS_FOR_IMPUTATION
        numerical_cols_path_pickle = temp_pickle_dir / "numerical_cols_santamaria.pkl"
        with open(numerical_cols_path_pickle, 'wb') as f:
            pickle.dump(Config.NUMERICAL_COLS_FOR_IMPUTATION, f)

    # 测试所有三种缺失机制
    mechanisms_to_test = ["MCAR", "MAR", "NMAR"]
    pbar_mech_total = len(mechanisms_to_test)
    pbar_var_total = len(Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS)
    pbar_level_total = len(Config.MISSINGNESS_LEVELS)

    for mech_idx, mechanism in enumerate(mechanisms_to_test):
        logger.info(f"Processing Mechanism: {mechanism} ({mech_idx+1}/{pbar_mech_total})")
        base_dir_mech = getattr(Config, f"{mechanism.upper()}_BASE_DIR")
        safe_create_directory(base_dir_mech)

        for var_idx, key_var in enumerate(Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS):
            logger.info(f"  Processing Variable for Missingness: {key_var} ({var_idx+1}/{pbar_var_total}) under {mechanism}")
            key_var_path_dir = Path(base_dir_mech) / f"imputed_for_{key_var}"
            safe_create_directory(key_var_path_dir)

            for lvl_idx, miss_level in enumerate(Config.MISSINGNESS_LEVELS):
                logger.info(f"    Processing Missingness Level: {miss_level*100:.0f}% ({lvl_idx+1}/{pbar_level_total}) for {key_var} under {mechanism}")
                level_path_dir = key_var_path_dir / f"{int(miss_level*100)}pct_missing"
                safe_create_directory(level_path_dir) 

                iteration_tasks_args = []
                for i_iter_loop in range(Config.NUM_ITERATIONS_PER_SCENARIO):
                    # Prepare tasks for parallel processing using file paths
                    args_tuple = (key_var, mechanism, miss_level, i_iter_loop,
                                  str(full_data_path_pickle), str(baseline_models_path_pickle), str(baseline_coef_info_path_pickle), str(numerical_cols_path_pickle))
                    iteration_tasks_args.append(args_tuple)

                all_iteration_results_payloads = []
                if Config.USE_PARALLEL and len(iteration_tasks_args) > 0:
                    with ProcessPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                        results_iterator = executor.map(process_single_iteration_wrapper, iteration_tasks_args, chunksize=Config.CHUNK_SIZE)
                        all_iteration_results_payloads = list(tqdm(results_iterator, total=len(iteration_tasks_args), 
                                                                  desc=f"Iter ({key_var} {miss_level*100:.0f}%)", unit="run", 
                                                                  position=0, leave=True)) 
                else: 
                    for args_s in tqdm(iteration_tasks_args, desc=f"Iter ({key_var} {miss_level*100:.0f}%)", unit="run", position=0, leave=True):
                        all_iteration_results_payloads.append(process_single_iteration_wrapper(args_s))
                
                merge_iteration_results(all_iteration_results_payloads,
                                        coef_stability_results,
                                        stats_features_results,
                                        model_comparison_results)
                
                if Config.CLEANUP_IMPUTED_FILES:
                    cleanup_iteration_artifacts(mechanism, key_var, miss_level)

    logger.info("\n--- Analysis Loop Completed ---")
    
    logger.info("Generating Excel report...")
    with pd.ExcelWriter(Config.OUTPUT_EXCEL_FILE, engine='openpyxl') as writer:
        # Baseline Descriptives and Correlations
        try:
            desc_cols_actual = [c for c in (getattr(Config, 'COLS_DESCRIPTIVE', []) or []) if c in full_data.columns]
            if desc_cols_actual:
                full_data[desc_cols_actual].describe().T.to_excel(writer, sheet_name='Baseline_Descriptives')
            corr_cols_actual = [c for c in (getattr(Config, 'COLS_CORRELATION', []) or []) if c in full_data.columns]
            if len(corr_cols_actual) >= 2:
                corstars_py(full_data, corr_cols_actual, remove_triangle='lower').to_excel(writer, sheet_name='Baseline_Correlations')
        except Exception as e_bd:
            logger.warning(f"Error writing baseline descriptives/correlations: {e_bd}")

        # Baseline Model Table
        if baseline_models:
            logger.info("Writing Baseline Regression table to Excel...")
            baseline_model_key_to_report = list(Config.MODEL_FORMULAS.keys())[0]
            if baseline_model_key_to_report in baseline_models:
                baseline_reg_df = create_excel_regression_table(
                    {baseline_model_key_to_report: baseline_models[baseline_model_key_to_report]},
                    [f"Baseline: {baseline_model_key_to_report}"]
                )
                baseline_reg_df.to_excel(writer, sheet_name="Baseline_Regression")
            else:
                logger.warning("Baseline model selected for reporting not found.")
        else:
            logger.warning("No baseline models were successfully run to report.")

        # Model Comparison Metrics Table
        logger.info("Writing Model Comparison Metrics table to Excel...")
        model_comp_df = create_excel_model_comparison_table(model_comparison_results)
        if not model_comp_df.empty:
            model_comp_df.to_excel(writer, sheet_name="Model_Comparison_Metrics", index=False)
        else:
            logger.warning("No model comparison data to write.")

        # Coefficient Stability Tables
        logger.info("Writing Coefficient Stability tables to Excel...")
        write_stability_tables_to_excel(writer, 
                                        coef_stability_results, 
                                        ["MCAR", "MAR", "NMAR"],  # 处理所有三种机制
                                        Config.MISSINGNESS_LEVELS,
                                        Config.METHOD_DISPLAY_NAMES,
                                        Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS)

        # Stats Features Table
        logger.info("Writing Stats Features table to Excel...")
        stats_features_df = create_excel_stats_features_table(stats_features_results)
        if not stats_features_df.empty:
            stats_features_df.to_excel(writer, sheet_name="Stats_Features", index=False)
        else:
            logger.warning("No stats features data to write.")

        # Coef Stability Summary (aggregate)
        write_coef_stability_summary(writer,
                                     coef_stability_results,
                                     Config.METHOD_DISPLAY_NAMES,
                                     ["MCAR", "MAR", "NMAR"],  # 处理所有三种机制
                                     Config.MISSINGNESS_LEVELS,
                                     Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS)

        # Benchmarks
        variable_groups_benchmark = {
            f"{','.join(Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS)}_stats": list(Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS)
        }
        write_benchmark_tables_to_excel(writer, coef_stability_results, variable_groups_benchmark, Config.MISSINGNESS_LEVELS, Config.METHOD_DISPLAY_NAMES)

        # Missingness Patterns
        try:
            miss_cols = sorted(list(set(getattr(Config, 'NUMERICAL_COLS_FOR_IMPUTATION', []) or [])))
            miss_patterns_df = _compute_missingness_patterns_table(full_data, miss_cols)
            if not miss_patterns_df.empty:
                miss_patterns_df.to_excel(writer, sheet_name='Missingness_Patterns', index=False)
        except Exception as e_mpat:
            logger.warning(f"Failed writing Missingness_Patterns: {e_mpat}")

        # MI Diagnostics and Trace (single illustrative scenario)
        try:
            mech0 = mechanisms_to_test[0]  # 使用第一个机制
            lvl0 = Config.MISSINGNESS_LEVELS[0]
            key_var0 = Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS[0] if Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS else None
            if key_var0:
                df_with_missing_sample = simulate_missingness_single_col(
                    full_data.copy(), col_to_make_missing=key_var0, miss_prop=lvl0,
                    seed=Config.SIMULATION_SEED + 999, mechanism=mech0,
                    mar_control_col=getattr(Config, 'MAR_CONTROL_COL', None), mar_strength=getattr(Config, 'MAR_STRENGTH_FACTOR', 1.0),
                    nmar_strength=getattr(Config, 'NMAR_STRENGTH_FACTOR', 1.0)
                )
                imp_handler = ImputationPipeline(
                    input_df_with_na=df_with_missing_sample, original_df_complete_subset=full_data.copy(),
                    missingness_level_config=lvl0, mechanism_config=mech0,
                    current_key_var_imputed=key_var0, iteration_num=0
                )
                mi_list = imp_handler.custom_multiple_imputation()
                model_key0 = list(Config.MODEL_FORMULAS.keys())[0]
                model_results_list = _fit_models_on_imputed_list(mi_list, Config.MODEL_FORMULAS[model_key0], model_key0)
                diag_df, traces, within_vars = _compute_mi_param_diagnostics(model_results_list)
                if not diag_df.empty:
                    diag_df_out = diag_df.copy()
                    diag_df_out.insert(0, 'Method', Config.METHOD_DISPLAY_NAMES.get('custom_multiple_imputation', 'MI'))
                    diag_df_out.insert(0, 'KeyVar', key_var0)
                    diag_df_out.insert(0, 'Level', f"{int(lvl0*100)}%")
                    diag_df_out.insert(0, 'Mechanism', mech0)
                    diag_df_out.to_excel(writer, sheet_name='MI_Diagnostics_ByParam', index=False)
                trace_rows = []
                for coef_name, q_vals in traces.items():
                    for idx_m, qv in enumerate(q_vals, start=1):
                        trace_rows.append({
                            'Mechanism': mech0, 'Level': f"{int(lvl0*100)}%",
                            'KeyVar': key_var0, 'Coef': coef_name, 'm': idx_m, 'Q_m': qv,
                            'Method': Config.METHOD_DISPLAY_NAMES.get('custom_multiple_imputation', 'MI')
                        })
                if trace_rows:
                    pd.DataFrame(trace_rows).to_excel(writer, sheet_name='MI_Trace_MI', index=False)
        except Exception as e_mid:
            logger.warning(f"MI diagnostics/trace failed: {e_mid}")

        # Overimputation and Distribution comparisons
        try:
            mech0 = "MCAR"
            lvl0 = Config.MISSINGNESS_LEVELS[0]
            key_var0 = Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS[0] if Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS else None
            if key_var0:
                df_with_missing_example = simulate_missingness_single_col(full_data.copy(), key_var0, lvl0,
                                                                          seed=Config.SIMULATION_SEED+321, mechanism=mech0,
                                                                          mar_control_col=getattr(Config, 'MAR_CONTROL_COL', None),
                                                                          mar_strength=getattr(Config, 'MAR_STRENGTH_FACTOR', 1.0),
                                                                          nmar_strength=getattr(Config, 'NMAR_STRENGTH_FACTOR', 1.0))
                imp_for_dist = ImputationPipeline(df_with_missing_example, full_data.copy(), lvl0, mech0, key_var0, 0)
                mi_list_for_dist = imp_for_dist.custom_multiple_imputation()
                if mi_list_for_dist:
                    try:
                        df_imputed_avg = sum(mi_list_for_dist) / len(mi_list_for_dist)
                    except Exception:
                        df_imputed_avg = pd.concat(mi_list_for_dist).groupby(level=0).mean(numeric_only=True)
                else:
                    df_imputed_avg = df_with_missing_example.copy()
                over_rows = []
                dist_rows = []
                model_key0 = list(Config.MODEL_FORMULAS.keys())[0]
                met = _overimputation_metrics_for_var(full_data, key_var0, mech0, lvl0,
                                                      Config.MODEL_FORMULAS[model_key0], model_key0)
                if met: over_rows.append(met)
                dist = _distribution_compare_for_var(df_with_missing_example, df_imputed_avg, key_var0, mech0, lvl0)
                if dist: dist_rows.append(dist)
                over_df = pd.DataFrame(over_rows) if over_rows else pd.DataFrame(columns=['DatasetID','Var','Method','Mechanism','Level','RMSE','MAE','Coverage95%','StdPredError_Mean','SMD','VarRatio','n_test'])
                dist_df = pd.DataFrame(dist_rows) if dist_rows else pd.DataFrame(columns=['Var','Method','Mechanism','Level','Mean_obs','Mean_imp','SMD','Var_obs','Var_imp','VR','KS_p'])
                over_df.to_excel(writer, sheet_name='MI_Overimputation_Checks', index=False)
                dist_df.to_excel(writer, sheet_name='MI_Distribution_Compare', index=False)
        except Exception as e_over:
            logger.warning(f"Overimputation/Distribution comparison failed: {e_over}")

        # NMAR diagnostics
        try:
            rows_nmar_resid = []
            model_key0 = list(Config.MODEL_FORMULAS.keys())[0]
            key_var0 = Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS[0] if Config.KEY_VARIABLES_FOR_ITERATIVE_MISSINGNESS else None
            if key_var0:
                res = _nmar_residual_association(full_data.copy(), key_var0, Config.MODEL_FORMULAS[model_key0])
                if res:
                    row = {'Var': key_var0, 'Mechanism': 'RealData', 'Level': '', 'Model': res.get('Model',''), 'beta_resid': res.get('beta_resid',np.nan), 'se': res.get('se',np.nan), 'pval': res.get('pval',np.nan), 'AUC': res.get('AUC',np.nan), 'Note': ''}
                    rows_nmar_resid.append(row)
                df_nmar_resid = pd.DataFrame(rows_nmar_resid) if rows_nmar_resid else pd.DataFrame(columns=['Var','Mechanism','Level','Model','beta_resid','se','pval','AUC','Note'])
                df_nmar_resid.to_excel(writer, sheet_name='NMAR_Residual_Association', index=False)

                rows_delta = []
                deltas = [-1.0,-0.5,-0.25,0.0,0.25,0.5,1.0]
                mech0 = "MCAR"
                lvl0 = Config.MISSINGNESS_LEVELS[0]
                df_with_missing_example = simulate_missingness_single_col(full_data.copy(), key_var0, lvl0,
                                                                          seed=Config.SIMULATION_SEED+654, mechanism=mech0,
                                                                          mar_control_col=getattr(Config, 'MAR_CONTROL_COL', None),
                                                                          mar_strength=getattr(Config, 'MAR_STRENGTH_FACTOR', 1.0),
                                                                          nmar_strength=getattr(Config, 'NMAR_STRENGTH_FACTOR', 1.0))
                imp_pipeline = ImputationPipeline(df_with_missing_example, full_data.copy(), lvl0, mech0, key_var0, 0)
                mi_list = imp_pipeline.custom_multiple_imputation()
                rows_delta.extend(_nmar_delta_sensitivity_rows(df_with_missing_example, mi_list, key_var0, deltas,
                                                               Config.MODEL_FORMULAS[model_key0], model_key0))
                df_delta = pd.DataFrame(rows_delta) if rows_delta else pd.DataFrame(columns=['Var','Delta_SD','Coef','SE','p','Sign','B_SameSign','SS_Changed','Tip_Delta'])
                if not df_delta.empty:
                    df_delta.insert(1, 'Method', Config.METHOD_DISPLAY_NAMES.get('custom_multiple_imputation','MI'))
                    df_delta.insert(2, 'Mechanism', mech0)
                    df_delta.insert(3, 'Level', f"{int(lvl0*100)}%")
                df_delta.to_excel(writer, sheet_name='NMAR_Delta_Sensitivity', index=False)
        except Exception as e_nmar:
            logger.warning(f"NMAR diagnostics failed: {e_nmar}")
            
    # Clean up temporary pickle files
    if Config.USE_PARALLEL and temp_pickle_dir.exists():
        try:
            import shutil
            shutil.rmtree(temp_pickle_dir)
            logger.info("Cleaned up temporary pickle files")
        except Exception as e_cleanup:
            logger.warning(f"Failed to clean up temporary pickle files: {e_cleanup}")
    
    logger.info(f"Full analysis completed. Excel report generated: {Config.OUTPUT_EXCEL_FILE}")
    return coef_stability_results, stats_features_results, model_comparison_results


# --- Execute ---
if __name__ == "__main__":
    # Initialize Config properly
    config_instance = Config()
    
    if "R_HOME" not in os.environ:
         logger.warning("R_HOME environment variable is not set. rpy2 will attempt to find R automatically. If it fails, R regressions will not work.")
         common_r_paths = ["/usr/lib/R", "/usr/local/lib/R", "/opt/R/current/lib/R"]
         for r_path_try in common_r_paths:
             if os.path.exists(r_path_try):
                 os.environ["R_HOME"] = r_path_try
                 logger.info(f"Found R at {r_path_try} and set R_HOME.")
                 break
         if "R_HOME" not in os.environ:
              logger.warning("Could not automatically find R. Please set R_HOME environment variable.")

    if not R_OK:
        logger.error("CRITICAL: rpy2 setup failed or R packages (fixest, broom) not found. R regressions will not be possible.")
        logger.error("Please ensure R and the required packages (fixest, broom) are installed and rpy2 can connect to R.")

    if not os.path.exists(Config.ORIGINAL_DATA_FILE):
        logger.error(f"CRITICAL ERROR: Original data file '{Config.ORIGINAL_DATA_FILE}' not found.")
        logger.warning(f"Creating dummy '{Config.ORIGINAL_DATA_FILE}' for testing structure based on Santamaria et al. (2024).")
        
        num_sn = 100 
        rng_dummy = np.random.default_rng(Config.SIMULATION_SEED)
        
        dummy_rows = []
        for sn_id in range(1, num_sn + 1):
            treatment_status = rng_dummy.choice([0, 1]) 
            age = rng_dummy.integers(25, 55)
            gender_text = rng_dummy.choice(["Male", "Female"])
            ethnicity_text = rng_dummy.choice(["Chinese", "Malay", "Indian", "Other"])
            edu_text = rng_dummy.choice(["Diploma or equivalent", "Degree", "Masters", "PhD"])
            field_text = rng_dummy.choice(["Engineering", "Computing", "Science & Mathematics", "Business", "Arts", "Other"])
            studying_val = rng_dummy.choice([0,1])
            working_val = rng_dummy.choice([0,1]) if not studying_val else 0
            entrep_exp_val = rng_dummy.choice([0,1])
            work_exp_val = rng_dummy.integers(0,5) 
            industry_text = rng_dummy.choice(["Technology", "Retail", "Services", "Manufacturing", "Other"])
            teamsize_val = rng_dummy.integers(1,5)
            registered_val = rng_dummy.choice([0,1])
            sector_text = rng_dummy.choice(["Commerce & E-commerce", "Online Platform and Apps", "Services", "Others"])
            round_val = rng_dummy.choice([1,2]) 

            rev_likert_base = rng_dummy.integers(1, 3) 
            cust_likert_base = rng_dummy.integers(1, 3)

            common_attrs = {
                "SN": sn_id, "Treatment": treatment_status, "Round": round_val, "Age": age, "Gender": gender_text,
                "Ethnicity": ethnicity_text, "HighestEducationAttained": edu_text, "Fieldofstudy": field_text,
                "Studying": studying_val, "Working": working_val, "EntrepExperience": entrep_exp_val,
                "WorkExp": work_exp_val, "IndustryExperience": industry_text, "TeamSize": teamsize_val,
                "Registered": registered_val, "Sector": sector_text,
                "Residence": "Singaporean", "TimeEffort": rng_dummy.integers(0,5),
                "CustomerInteraction": rng_dummy.integers(0,5), "ExpertInteraction": rng_dummy.integers(0,5),
                "Networking": rng_dummy.integers(0,5), "exit": 0, "AttendanceCertificate": 1
            }

            dummy_rows.append({**common_attrs, "Post": 0, "RevenueLikert": rev_likert_base, "CustomersLikert": cust_likert_base})

            cust_likert_end = cust_likert_base + rng_dummy.choice([0,1]) 
            if treatment_status == 1: 
                cust_likert_end += rng_dummy.choice([0,1,1,2]) 
            cust_likert_end = np.clip(cust_likert_end, 1, 5)

            rev_likert_end = rev_likert_base + rng_dummy.choice([0,1])
            if treatment_status == 1: rev_likert_end += rng_dummy.choice([0,1,1,2])
            rev_likert_end = np.clip(rev_likert_end, 1, 5)
            
            dummy_rows.append({**common_attrs, "Post": 1, "RevenueLikert": rev_likert_end, "CustomersLikert": cust_likert_end})

        dummy_df = pd.DataFrame(dummy_rows)
        
        try:
            dummy_df.to_csv(Config.ORIGINAL_DATA_FILE, index=False)
            logger.info(f"Created dummy data: '{Config.ORIGINAL_DATA_FILE}' ({len(dummy_df)} rows).")
            run_full_analysis()
        except Exception as e_create_dummy:
            logger.error(f"Could not create dummy CSV or run analysis: {e_create_dummy}", exc_info=True)
            print(f"Failed to create dummy CSV or run analysis. Please provide '{Config.ORIGINAL_DATA_FILE}'.")
    else:
        run_full_analysis()
