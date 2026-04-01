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
├── paper_info_0005.xlsx                       # Flat export: Paper 0005 information record
├── paper_info_0017.xlsx                       # Flat export: Paper 0017 information record
├── regression_results_0005.xlsx               # Flat export: Paper 0005 regression results
├── regression_results_0017.xlsx               # Flat export: Paper 0017 regression results
├── Paper_Info_Record_0005.pdf                 # Flat export: Paper 0005 PDF info record
├── Paper_Info_Record_0017.pdf                 # Flat export: Paper 0017 PDF info record
│
├── source_artifacts/                          # Original paper source files (read-only archive)
│   ├── mapping_entrepreneurial_inclusion.dta  # Paper 0005 Stata dataset
│   ├── mapping_entrepreneurial_inclusion.do   # Paper 0005 Stata replication code
│   ├── movie_data.csv                         # Paper 0017 original dataset
│   ├── movies.R                               # Paper 0017 R replication code
│   └── ...                                    # Published PDFs, demand-pull paper files
│
└── paper_analysis_output/                     # (folder is named "Week 2" — rename manually)
    ├── Paper_0005_MappingEntrepreneurial/
    │   ├── DATA.csv                           # Pre-processed analysis dataset
    │   ├── Paper_Info_Record.txt              # Metadata, baseline validation, QC log
    │   ├── Paper_Info_Record.pdf              # PDF version (generated)
    │   ├── confignotes.txt                    # Model spec, estimator decisions, key vars
    │   ├── scripts/
    │   │   └── simulation_0005.py             # Main simulation script
    │   ├── logs/                              # Run logs and progress files
    │   ├── smoke/                             # Smoke-test workbooks
    │   ├── full_run/                          # Final report workbooks + exports
    │   │   ├── Stroube2025Report_0005.xlsx    # Final workbook (17 sheets, QC passed)
    │   │   ├── paper_info_0005.xlsx           # Structured information record
    │   │   └── regression_results_0005.xlsx   # Baseline + simulation summary
    │   └── regression_outputs/               # Per-iteration regression text outputs
    │
    └── Paper_0017_StatusConsensus/
        ├── DATA.csv
        ├── Paper_Info_Record.txt
        ├── Paper_Info_Record.pdf
        ├── confignotes.txt
        ├── scripts/
        │   └── simulation_0017.py
        ├── logs/
        ├── smoke/
        ├── full_run/
        │   ├── Stroube2024Report_0017.xlsx
        │   ├── paper_info_0017.xlsx
        │   └── regression_results_0017.xlsx
        └── regression_outputs/
```

> **Folder rename:** The analysis folder is currently named `Week 2/`. To complete the
> submission structure, rename it: `ren "Week 2" paper_analysis_output` (close all
> Explorer windows pointing to it first, then run from `cmd.exe` in the repo root).

> **Note on gitignored files:** Simulation run logs (`simulation_*.log`, `*progress*.log`)
> are excluded from version control. All workbooks and regression outputs are tracked.

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
python "paper_analysis_output/Paper_0005_MappingEntrepreneurial/scripts/simulation_0005.py" --mode baseline
python "paper_analysis_output/Paper_0017_StatusConsensus/scripts/simulation_0017.py" --mode baseline

# Smoke test: 2 iterations x reduced missingness levels (fast, ~2 minutes)
python "paper_analysis_output/Paper_0005_MappingEntrepreneurial/scripts/simulation_0005.py" --mode smoke
python "paper_analysis_output/Paper_0017_StatusConsensus/scripts/simulation_0017.py" --mode smoke

# Full simulation: 30 iterations x all levels (hours)
python "paper_analysis_output/Paper_0005_MappingEntrepreneurial/scripts/simulation_0005.py" --mode full
python "paper_analysis_output/Paper_0017_StatusConsensus/scripts/simulation_0017.py" --mode full
```

> Until the folder is renamed, substitute `Week 2` for `paper_analysis_output` in the
> commands above.

Scripts read source data from `source_artifacts/` (if DATA.csv is absent), write logs
to `logs/`, and write the final report workbook to `full_run/` on completion.
