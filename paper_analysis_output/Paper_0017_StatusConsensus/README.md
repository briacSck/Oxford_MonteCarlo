# Paper 0017 — Status and Consensus: Heterogeneity in Audience Evaluations of Female- versus Male-Lead Films

## Metadata

| Field | Value |
|---|---|
| Paper ID | 0017 |
| Title | Status and Consensus: Heterogeneity in Audience Evaluations of Female- versus Male-Lead Films |
| Authors | Bryan Stroube |
| Journal | Strategic Management Journal, 45:994-1024 (2024) |
| Source model | Table 2, Model 2 (`m.pooled.sd`) |

---

## Data

`DATA.csv` is a cross-sectional film-level dataset, derived from
`source_artifacts/movie_data.csv` (4,012 rows x 49 columns). The dependent variable
`sd_pooled` (pooled rating standard deviation) and its sibling `mean_pooled` are
pre-computed in the source CSV. The key derived variable `log_bom_opening_theaters` is
computed as `np.log(bom_opening_theaters)` at load time (no zero values; min = 1).

**Preprocessing applied at load time:**
- Column rename: dots to underscores (`genres.count` -> `genres_count`,
  `genre.Sci.Fi` -> `genre_Sci_Fi`, etc.) for Python compatibility.
- `FLead` and `major` cast from boolean to integer (0/1).
- Year (`bom_year`) and month (`bom_open_month`) fixed effects expanded via
  `pd.get_dummies(drop_first=True)`, matching R's `as.factor()` treatment coding.
- 22 genre dummy columns included as pre-built binary integers from the source CSV.

---

## Regression Specification

**Estimator:** `statsmodels.OLS` with standard (non-clustered) standard errors

**Model:**
```
sd_pooled ~ FLead + mean_pooled + kim_violence_gore + kim_sex_nudity
          + kim_language + major + log_bom_opening_theaters + genres_count
          + C(bom_year) + C(bom_open_month) + [22 genre dummies]
```

**Fixed effects:** Year and month dummies expanded explicitly (treatment coding,
`drop_first=True`). Genre dummies are non-mutually exclusive, so `statsmodels` dummy
expansion is valid.

**Clustered SEs:** None — standard OLS SEs throughout (matching R's `lm()` defaults).

---

## Baseline Validation

Source: Paper Table 2, Column 2 (DV = `sd_pooled`).

| Variable | Published | Replicated | Match |
|---|---|---|---|
| `FLead` | 0.047** | 0.0468 | rounds exactly |
| SE (`FLead`) | -- | 0.0080 | -- |
| p-value (`FLead`) | < 0.01 | 6.23e-09 | consistent |
| N | -- | 4,012 | -- |
| R-squared | -- | 0.5620 | -- |

Baseline status: **MATCHED** (2026-03-28). The published coefficient 0.047 is the
3-decimal rounding of the replicated value 0.0468.

---

## Simulation Configuration

### Key variables (4, locked 2026-03-28)

| Variable | Rationale |
|---|---|
| `kim_violence_gore` | Content rating dimension; continuous, theoretically linked to audience heterogeneity |
| `kim_sex_nudity` | Content rating dimension; continuous, high variance across film types |
| `kim_language` | Content rating dimension; continuous, captures audience segmentation |
| `log_bom_opening_theaters` | Distribution scale; continuous log-transform, strong predictor of rating variance |

### MAR control variable

`genres_count` — number of genres assigned to the film. Continuous ordinal (1-7+),
complete in the dataset, plausibly correlated with all content-rating key variables
(broader films span more content dimensions), and excluded from the key variable set.

### Full simulation grid

| Parameter | Value |
|---|---|
| Mechanisms | MCAR, MAR, NMAR |
| Missingness proportions | 1%, 5%, 10%, 20%, 30%, 40%, 50% |
| Methods | LD, Mean, Reg, Iter, RF, DL, MILGBM |
| Iterations per cell | 30 |
| MAR/NMAR strength | 1.5 |
| MI completed datasets (M) | 5 |
| Predictor pool (imputation) | `FLead`, `mean_pooled`, `kim_violence_gore`, `kim_sex_nudity`, `kim_language`, `major`, `log_bom_opening_theaters`, `genres_count` |
| DL method | TensorFlow 2.20.0, MLP 32->Dropout(0.1)->16->1, ReLU, Adam lr=0.005, EarlyStopping |
| MILGBM method | LightGBM 4.6.0, MICE M=5, Rubin's Rules, 30 estimators, max_depth=4 |

---

## Simulation Status

| Phase | Status |
|---|---|
| Baseline replication | Matched (2026-03-28) |
| Smoke test | Passed -- 84 runs, 0 errors (2026-03-29 09:52-09:53) |
| Full run | Complete -- 17,640 runs, 0 errors (2026-03-31 11:00 to 2026-04-01 00:56) |
| QC signoff | All 11 checks passed (2026-04-01) |

---

## Running the Simulation

```bash
# From repo root
python "paper_analysis_output/Paper_0017_StatusConsensus/scripts/simulation_0017.py" --mode baseline
python "paper_analysis_output/Paper_0017_StatusConsensus/scripts/simulation_0017.py" --mode smoke
python "paper_analysis_output/Paper_0017_StatusConsensus/scripts/simulation_0017.py" --mode full
```

The script reads `source_artifacts/movie_data.csv` on first run (if `DATA.csv` is absent)
to build `DATA.csv`, then proceeds with simulation. Structured logs are written to `logs/`
(progress logs are gitignored). Per-iteration regression outputs are written to
`regression_outputs/`.

---

## Output

**Primary deliverable:** `full_run/Stroube2024Report_0017.xlsx` — 17-sheet workbook:
- Baseline coefficient table vs. published values
- Stability heatmaps per mechanism (MCAR, MAR, NMAR): "Both Same" (B) and "Sign Same,
  Sig Changed" (SS) fractions across proportion x method grids
- Coefficient summaries and model comparison tables
- MILGBM diagnostics (FMI, Relative Efficiency, Rubin-pooled SEs)
- QC flags and iteration counts

**Figures:** `full_run/figures/`
- `fig1_stability_heatmap_paper0017.png` — B-proportion heatmap (method x missingness, MCAR)
- `fig2_method_comparison_paper0017.png` — mean B-proportion by imputation method
- `fig3_stability_trajectory_paper0017.png` — B-proportion trajectory by method

**Supporting exports:** `full_run/paper_info_0017.xlsx`, `full_run/regression_results_0017.xlsx`,
`Paper_Info_Record.pdf`
