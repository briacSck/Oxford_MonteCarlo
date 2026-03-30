# Oxford Monte Carlo — Missing Data Imputation Robustness Study

This repository implements a Monte Carlo simulation framework for evaluating the
robustness of regression findings in published economics papers under artificial
missingness. For each paper, the framework introduces missing data across three
mechanisms (MCAR, MAR, NMAR) at seven severity levels (1–50%), applies seven
imputation methods to each incomplete dataset, re-estimates the paper's preferred
regression specification, and tracks coefficient stability across 30 iterations per
cell. The primary metric is the fraction of iterations in which the focal coefficient
preserves both its sign and statistical significance relative to the complete-data
baseline.

---

## Repository Structure

```
Oxford_MonteCarlo/
├── RA_MISSING_DATA.pdf                        # Operations manual (full protocol)
├── requirements.txt                           # Python dependencies
│
└── Week 2/
    ├── Paper_0005_MappingEntrepreneurial/     # Paper 0005 — simulation workspace
    │   ├── DATA.csv                           # Pre-processed analysis dataset
    │   ├── Paper_Info_Record.txt              # Metadata, baseline validation, QC log
    │   ├── confignotes.txt                    # Model spec, estimator decisions, key vars
    │   ├── simulation_0005.py                 # Main simulation script
    │   └── SMOKE_Stroube2025Report_0005.xlsx  # Final output workbook (17 sheets)
    │
    ├── Paper_0017_StatusConsensus/            # Paper 0017 — simulation workspace
    │   ├── DATA.csv                           # Pre-processed analysis dataset
    │   ├── Paper_Info_Record.txt              # Metadata, baseline validation, QC log
    │   ├── confignotes.txt                    # Model spec, estimator decisions, key vars
    │   ├── simulation_0017.py                 # Main simulation script
    │   └── SMOKE_Stroube2024Report_0017.xlsx  # Final output workbook (17 sheets)
    │
    ├── paper_mappingentrepreneurialinclusion/ # Source materials for Paper 0005
    │   ├── mapping_entrepreneurial_inclusion.dta   # Original Stata dataset (read by simulation_0005.py)
    │   ├── mapping_entrepreneurial_inclusion.do    # Original Stata replication code
    │   └── 0005 - Mapping entrepreneurial inclusion [...].pdf  # Published paper
    │
    ├── paper_statusandconsensus/              # Source materials for Paper 0017
    │   ├── movie_data.csv                     # Original dataset (read by simulation_0017.py)
    │   ├── experiment_pairs_data.csv          # Subsample data (figures/Table 4 only)
    │   ├── movies.R                           # Original R replication code
    │   └── 0017 - Status and consensus [...].pdf   # Published paper
    │
    └── paper_demandpull/                      # Unassigned paper (demand-pull, SMJ 3560)
        ├── All_in_One refer for 3560.py       # Legacy simulation script (not part of this study)
        ├── STATA CODE 3560.do
        ├── STATA CODE 3560.dta
        └── smj.3560.pdf
```

> **Note on gitignored files:** Per-iteration regression text outputs
> (`regression_txt_outputs/`) and simulation run logs (`simulation_*.log`)
> are excluded from version control due to volume. The final report workbooks
> (`SMOKE_*Report_*.xlsx`) are tracked.

---

## Simulation Protocol

Full methodology is documented in [`RA_MISSING_DATA.pdf`](RA_MISSING_DATA.pdf),
an 18-page operations manual covering:

- Missingness mechanism specifications (MCAR, MAR, NMAR)
- Imputation method implementations (7 methods)
- Simulation grid defaults and deviation policy
- Key-variable and MAR-control selection rules
- Baseline replication gate and QC checklist
- Output contract and archive structure

---

## Papers Analyzed

| Paper ID | Title | Authors | Journal | Year | Status |
|---|---|---|---|---|---|
| 0005 | Mapping Entrepreneurial Inclusion Across US Neighborhoods: The Case of Low-Code E-commerce Entrepreneurship | Stroube & Dushnitsky | Strategic Management Journal | ~2025 | Run in progress / QC pending |
| 0017 | Status and Consensus: Heterogeneity in Audience Evaluations of Female- versus Male-Lead Films | Stroube | Strategic Management Journal | 2024 | Complete |

---

## Key Output Files

Each paper produces a single Excel workbook (`SMOKE_*Report_*.xlsx`) containing
17 sheets:

| Sheet group | Contents |
|---|---|
| **Baseline** | Replicated coefficient table vs. published values; discrepancy flags |
| **Stability heatmaps** (×3) | Per-mechanism (MCAR / MAR / NMAR) grids: proportion × method, showing "Both Same" (B), "Sign Same, Sig Changed" (SS), and complementary fractions |
| **Coefficient summaries** | Mean estimated coefficient per cell; deviation from baseline |
| **Model comparison** | Method-level aggregates across all key variables and proportions |
| **MI diagnostics** | MILGBM-specific: FMI, Relative Efficiency, average Rubin-pooled SE |
| **QC flags** | Iteration counts, blank-cell detection, sanity check results |

The primary metric is **B (Both Same)**: the fraction of 30 iterations in which
the focal coefficient preserves both its sign and statistical significance at α = 0.05
relative to the complete-data baseline.

---

## Requirements

See [`requirements.txt`](requirements.txt) for the full dependency list.

Key packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyfixest` (Paper 0005),
`statsmodels` (Paper 0017), `lightgbm`, `openpyxl`, `pyreadstat`.

TensorFlow is optional (required only for Method 6 — Deep Learning imputation).
If unavailable, the DL method falls back to mean imputation silently.
Python 3.11+ is recommended.

---

## Reproduction

Each paper has its own self-contained simulation script with three run modes:

```bash
# Validate baseline replication only (fast, ~seconds)
python "Week 2/Paper_0005_MappingEntrepreneurial/simulation_0005.py" --mode baseline
python "Week 2/Paper_0017_StatusConsensus/simulation_0017.py" --mode baseline

# Smoke test: 2 iterations × reduced missingness levels (fast, ~2 minutes)
python "Week 2/Paper_0005_MappingEntrepreneurial/simulation_0005.py" --mode smoke
python "Week 2/Paper_0017_StatusConsensus/simulation_0017.py" --mode smoke

# Full simulation: 30 iterations × all levels (hours)
python "Week 2/Paper_0005_MappingEntrepreneurial/simulation_0005.py" --mode full
python "Week 2/Paper_0017_StatusConsensus/simulation_0017.py" --mode full
```

The scripts are self-contained: they read source data from the adjacent
`paper_*/` folders, write processed data to `DATA.csv`, emit structured logs
to `simulation_*.log`, and write the final report workbook on completion.
See each paper's `README.md` for paper-specific details.
