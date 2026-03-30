# Paper 0017 — Status and Consensus: Heterogeneity in Audience Evaluations of Female- versus Male-Lead Films

## Metadata

| Field | Value |
|---|---|
| Paper ID | 0017 |
| Title | Status and Consensus: Heterogeneity in Audience Evaluations of Female- versus Male-Lead Films |
| Authors | Bryan Stroube |
| Journal | Strategic Management Journal |
| Year | 2024 |
| Citation | SMJ 45:994–1024 |
| Source model | Table 2, Model 2 (`m.pooled.sd`) |

---

## Data

`DATA.csv` is a cross-sectional film-level dataset, derived from the original
`movie_data.csv` (4,012 rows × 49 columns). The dependent variable `sd_pooled`
(pooled rating standard deviation) and its sibling `mean_pooled` are pre-computed
in the source CSV. The key derived variable `log_bom_opening_theaters` is computed
as `np.log(bom_opening_theaters)` at load time (no zero values; min = 1).

**Preprocessing applied at load time:**
- Column rename: dots to underscores (`genres.count` → `genres_count`,
  `genre.Sci.Fi` → `genre_Sci_Fi`, etc.) for Python compatibility.
- `FLead` and `major` cast from boolean to integer (0/1).
- Year (`bom_year`) and month (`bom_open_month`) fixed effects expanded via
  `pd.get_dummies(drop_first=True)`, matching R's `as.factor()` treatment coding.
- 22 genre dummy columns included as pre-built binary integers from the source CSV.

`experiment_pairs_data.csv` (also in the source folder) is used only for figures
and the Table 4 subsample analyses; it is not required for the Table 2 preferred
specification.

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
`drop_first=True`). Unlike Paper 0005, there is no FE nesting issue here; genre
dummies are non-mutually exclusive, so `statsmodels` dummy expansion is valid.

**Clustered SEs:** None — standard OLS SEs throughout (matching R's `lm()` defaults).

---

## Baseline Validation

Source: Paper Table 2, Column 2 (DV = `sd_pooled`).

| Variable | Published | Replicated | Match |
|---|---|---|---|
| `FLead` | 0.047** | 0.0468 | rounds exactly |
| SE (`FLead`) | — | 0.0080 | — |
| p-value (`FLead`) | < 0.01 | 6.23e-09 | consistent |
| N | — | 4,012 | — |
| R² | — | 0.5620 | — |

Baseline status: **MATCHED** (2026-03-28). The published coefficient 0.047 is the
3-decimal rounding of the replicated value 0.0468.

---

## Simulation Configuration

| Parameter | Value |
|---|---|
| Key variables (4, locked 2026-03-28) | `kim_violence_gore`, `kim_sex_nudity`, `kim_language`, `log_bom_opening_theaters` |
| MAR control variable | `genres_count` |
| Predictor pool (imputation) | `FLead`, `mean_pooled`, `kim_violence_gore`, `kim_sex_nudity`, `kim_language`, `major`, `log_bom_opening_theaters`, `genres_count` |
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
| Smoke test | Passed — 84 runs, 0 errors (2026-03-29 09:52–09:53) |
| Full run | Complete — 17,640 runs, 0 errors (2026-03-29 16:51–20:49) |
| QC signoff | All 11 checks passed |

QC highlights: B-proportion = 100% at 1% missingness across all methods/variables;
LD at 50% missingness shows mean N = 2,006 (vs. baseline N = 4,012).

---

## Running the Simulation

```bash
# From repo root
python "Week 2/Paper_0017_StatusConsensus/simulation_0017.py" --mode baseline
python "Week 2/Paper_0017_StatusConsensus/simulation_0017.py" --mode smoke
python "Week 2/Paper_0017_StatusConsensus/simulation_0017.py" --mode full
```

The script reads `../paper_statusandconsensus/movie_data.csv` on first run to
build `DATA.csv`, then proceeds with simulation. Structured logs are written to
`simulation_0017.log` (gitignored). Per-iteration regression outputs are written
to `regression_txt_outputs/` (gitignored).

---

## Output

**`SMOKE_Stroube2024Report_0017.xlsx`** — 17-sheet workbook containing:
- Baseline coefficient table vs. published values
- Stability heatmaps per mechanism (MCAR, MAR, NMAR): "Both Same" (B) and "Sign Same,
  Sig Changed" (SS) fractions across proportion × method grids
- Coefficient summaries and model comparison tables
- MILGBM diagnostics (FMI, Relative Efficiency, Rubin-pooled SEs)
- QC flags and iteration counts
