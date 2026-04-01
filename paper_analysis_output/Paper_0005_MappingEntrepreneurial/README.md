# Paper 0005 — Mapping Entrepreneurial Inclusion Across US Neighborhoods

## Metadata

| Field | Value |
|---|---|
| Paper ID | 0005 |
| Title | Mapping Entrepreneurial Inclusion Across US Neighborhoods: The Case of Low-Code E-commerce Entrepreneurship |
| Authors | Bryan Stroube, Gary Dushnitsky |
| Journal | Strategic Management Journal (~2025) |
| Source model | Table 3, Column 1 (Model 3_1) |

---

## Data

`DATA.csv` is a cross-sectional dataset at the ZIP Code Tabulation Area (ZCTA) level,
derived from `source_artifacts/mapping_entrepreneurial_inclusion.dta`. It contains
32,647 observations across 22 columns. All key variables arrive pre-logged in the
source data (variable names carry a `log_` prefix). No merges, lags, or winsorization
are applied; the preprocessing step loads the `.dta`, casts the two fixed-effect columns
(`state_name_fe`, `MSA_fe`) to integer, and writes `DATA.csv`.

---

## Regression Specification

**Estimator:** `pyfixest.feols()` with CRV1 standard errors clustered on `MSA_fe`

**Model:**
```
log_shopify_count_1 ~ log_pop_black_aa + log_pop_total + log_total_bachelor_deg
                    + log_pop_total_poverty + log_total_social_cap
                    | state_name_fe + MSA_fe
```

**Fixed effects:** Two-way absorption — `state_name_fe` (state) and `MSA_fe`
(Metropolitan Statistical Area). `pyfixest.feols()` is used because naive dummy
expansion in `statsmodels` produced a 39% coefficient distortion due to collinearity
between nested state/MSA FE dummies.

**Clustered SEs:** CRV1 on `MSA_fe` — no weights.

---

## Baseline Validation

Source: Paper Table 3, Column 1. All coefficients match to four decimal places.

| Variable | Published | Replicated | Match |
|---|---|---|---|
| `log_pop_black_aa` | 0.0307 | 0.0307 | exact |
| `log_pop_total` | -0.0905 | -0.0905 | exact |
| `log_total_bachelor_deg` | 0.1431 | 0.1431 | exact |
| `log_pop_total_poverty` | -0.0120 | -0.0120 | exact |
| `log_total_social_cap` | 0.3684 | 0.3684 | exact |
| N | 32,647 | 32,647 | exact |
| R-squared | 0.6883 | 0.688 | within rounding |

Baseline status: **MATCHED** (2026-03-28).

---

## Simulation Configuration

### Key variables (4, locked 2026-03-28)

| Variable | Rationale |
|---|---|
| `log_pop_black_aa` | Focal independent variable; primary theoretical construct |
| `log_total_bachelor_deg` | Human capital measure; substantively central to entrepreneurship literature |
| `log_pop_total_poverty` | Socioeconomic control; continuous, high variation across ZCTAs |
| `log_total_social_cap` | Social infrastructure measure; largest coefficient in model |

### MAR control variable

`log_pop_total` — total ZCTA population. Continuous, complete, plausibly correlated with
all key variables (denser areas have more of everything), and excluded from the key
variable set.

### Full simulation grid

| Parameter | Value |
|---|---|
| Mechanisms | MCAR, MAR, NMAR |
| Missingness proportions | 1%, 5%, 10%, 20%, 30%, 40%, 50% |
| Methods | LD, Mean, Reg, Iter, RF, DL, MILGBM |
| Iterations per cell | 30 |
| MAR/NMAR strength | 1.5 |
| MI completed datasets (M) | 5 |
| Predictor pool (imputation) | `log_pop_black_aa`, `log_pop_total`, `log_total_bachelor_deg`, `log_pop_total_poverty`, `log_total_social_cap` |
| DL method | TensorFlow 2.20.0, MLP 32->Dropout(0.1)->16->1, ReLU, Adam lr=0.005, EarlyStopping |
| MILGBM method | LightGBM 4.6.0, MICE M=5, Rubin's Rules, 30 estimators, max_depth=4 |

---

## Simulation Status

| Phase | Status |
|---|---|
| Baseline replication | Matched (2026-03-28) |
| Smoke test | Passed -- 84 runs, 0 errors (2026-03-29 10:10-10:13) |
| Full run | Complete -- 17,640 runs, 0 errors (2026-03-29 21:17 to 2026-04-01 07:38) |
| QC signoff | All 11 checks passed (2026-04-01) |

---

## Running the Simulation

```bash
# From repo root
python "paper_analysis_output/Paper_0005_MappingEntrepreneurial/scripts/simulation_0005.py" --mode baseline
python "paper_analysis_output/Paper_0005_MappingEntrepreneurial/scripts/simulation_0005.py" --mode smoke
python "paper_analysis_output/Paper_0005_MappingEntrepreneurial/scripts/simulation_0005.py" --mode full
```

The script reads `source_artifacts/mapping_entrepreneurial_inclusion.dta` on first run
(if `DATA.csv` is absent) to build `DATA.csv`, then proceeds with simulation. Structured
logs are written to `logs/` (progress logs are gitignored). Per-iteration regression
outputs are written to `regression_outputs/`.

---

## Output

**Primary deliverable:** `full_run/Stroube2025Report_0005.xlsx` — 17-sheet workbook:
- Baseline coefficient table vs. published values
- Stability heatmaps per mechanism (MCAR, MAR, NMAR): "Both Same" (B) and "Sign Same,
  Sig Changed" (SS) fractions across proportion x method grids
- Coefficient summaries and model comparison tables
- MILGBM diagnostics (FMI, Relative Efficiency, Rubin-pooled SEs)
- QC flags and iteration counts

**Figures:** `full_run/figures/`
- `fig1_stability_heatmap_paper0005.png` — B-proportion heatmap (method x missingness, MCAR)
- `fig2_method_comparison_paper0005.png` — mean B-proportion by imputation method
- `fig3_stability_trajectory_paper0005.png` — B-proportion trajectory by method

**Supporting exports:** `full_run/paper_info_0005.xlsx`, `full_run/regression_results_0005.xlsx`,
`Paper_Info_Record.pdf`
