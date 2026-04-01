# Paper 0005 — Mapping Entrepreneurial Inclusion Across US Neighborhoods

## Metadata

| Field | Value |
|---|---|
| Paper ID | 0005 |
| Title | Mapping Entrepreneurial Inclusion Across US Neighborhoods: The Case of Low-Code E-commerce Entrepreneurship |
| Authors | Bryan Stroube, Gary Dushnitsky |
| Journal | Strategic Management Journal |
| Year | ~2025 |
| Source model | Table 3, Column 1 (Model 3_1) |

---

## Data

`DATA.csv` is a cross-sectional dataset at the ZIP Code Tabulation Area (ZCTA) level,
derived from the original Stata file `mapping_entrepreneurial_inclusion.dta`. It
contains 32,647 observations across 22 columns. All key variables arrive pre-logged
in the source data (variable names carry a `log_` prefix). No merges, lags, or
winsorization are applied; the preprocessing step loads the `.dta`, casts the two
fixed-effect columns (`state_name_fe`, `MSA_fe`) to integer, and writes `DATA.csv`.

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
between nested state/MSA FE dummies. `linearmodels.AbsorbingLS` was also rejected
(array length errors with the clusters argument).

**Clustered SEs:** CRV1 on `MSA_fe` — **no weights**.

---

## Baseline Validation

Source: Paper Table 3, Column 1. All coefficients match to four decimal places.

| Variable | Published | Replicated | Match |
|---|---|---|---|
| `log_pop_black_aa` | 0.0307 | 0.0307 | exact |
| `log_pop_total` | −0.0905 | −0.0905 | exact |
| `log_total_bachelor_deg` | 0.1431 | 0.1431 | exact |
| `log_pop_total_poverty` | −0.0120 | −0.0120 | exact |
| `log_total_social_cap` | 0.3684 | 0.3684 | exact |
| N | 32,647 | 32,647 | exact |
| R² | 0.6883 | 0.688 | within rounding |

Baseline status: **MATCHED** (2026-03-28).

---

## Simulation Configuration

| Parameter | Value |
|---|---|
| Key variables (4, locked 2026-03-28) | `log_pop_black_aa`, `log_total_bachelor_deg`, `log_pop_total_poverty`, `log_total_social_cap` |
| MAR control variable | `log_pop_total` |
| Predictor pool (imputation) | `log_pop_black_aa`, `log_pop_total`, `log_total_bachelor_deg`, `log_pop_total_poverty`, `log_total_social_cap` |
| Mechanisms | MCAR, MAR, NMAR |
| Missingness proportions | 1%, 5%, 10%, 20%, 30%, 40%, 50% |
| Methods | LD, Mean, Reg, Iter, RF, DL, MILGBM |
| Iterations per cell | 30 |
| MAR/NMAR strength | 1.5 |
| MI completed datasets | M = 5 |

**Note on DL method:** TensorFlow installation failed on this machine due to the
Windows Long Path limitation. The DL method (Method 6) silently falls back to mean
imputation via `try/except`. Results labelled "DL" in the report therefore reflect
mean imputation for this run.

---

## Simulation Status

| Phase | Status |
|---|---|
| Baseline replication | Matched (2026-03-28) |
| Smoke test | Passed — 84 runs, 0 errors (2026-03-29 10:10–10:13) |
| Full run | Started 2026-03-29 ~10:14 |
| QC signoff | Pending |

---

## Running the Simulation

```bash
# From repo root (after folder rename)
python "paper_analysis_output/Paper_0005_MappingEntrepreneurial/scripts/simulation_0005.py" --mode baseline
python "paper_analysis_output/Paper_0005_MappingEntrepreneurial/scripts/simulation_0005.py" --mode smoke
python "paper_analysis_output/Paper_0005_MappingEntrepreneurial/scripts/simulation_0005.py" --mode full
```

The script reads `source_artifacts/mapping_entrepreneurial_inclusion.dta` on first run
(if DATA.csv is absent) to build `DATA.csv`, then proceeds with simulation. Structured
logs are written to `logs/` (gitignored). Per-iteration regression outputs are written
to `regression_outputs/`.

---

## Output

**`SMOKE_Stroube2025Report_0005.xlsx`** — 17-sheet workbook containing:
- Baseline coefficient table vs. published values
- Stability heatmaps per mechanism (MCAR, MAR, NMAR): "Both Same" (B) and "Sign Same,
  Sig Changed" (SS) fractions across proportion × method grids
- Coefficient summaries and model comparison tables
- MILGBM diagnostics (FMI, Relative Efficiency, Rubin-pooled SEs)
- QC flags and iteration counts
