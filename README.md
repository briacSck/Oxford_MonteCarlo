# Oxford Monte Carlo — Missing Data Imputation Robustness Study

Monte Carlo simulation framework evaluating the robustness of regression findings in
published economics papers under artificial missingness. For each paper, the framework
introduces missing data across three mechanisms (MCAR, MAR, NMAR) at seven severity
levels (1–50%), applies seven imputation methods to each incomplete dataset,
re-estimates the paper's preferred regression specification, and tracks coefficient
stability across 30 iterations per cell.

**Primary metric:** fraction of iterations in which the focal coefficient preserves
both its sign and statistical significance relative to the complete-data baseline
("Both Same", B).

---

## Repository Structure

```
Oxford_MonteCarlo/
├── RA_MISSING_DATA.pdf                        # Operations manual (full protocol, 18 pp)
├── requirements.txt                           # Python dependencies
├── generate_figures.py                        # Generates 3 publication-quality figures per paper
├── generate_deliverables.py                   # Generates PDFs, paper_info.xlsx, regression_results.xlsx
│
├── paper_info_0005.xlsx                       # Root copy: Paper 0005 information record
├── paper_info_0017.xlsx                       # Root copy: Paper 0017 information record
├── regression_results_0005.xlsx               # Root copy: Paper 0005 regression results
├── regression_results_0017.xlsx               # Root copy: Paper 0017 regression results
├── Paper_Info_Record_0005.pdf                 # Root copy: Paper 0005 PDF info record
├── Paper_Info_Record_0017.pdf                 # Root copy: Paper 0017 PDF info record
│
├── source_artifacts/                          # Original paper source files (read-only archive)
│   ├── mapping_entrepreneurial_inclusion.dta  # Paper 0005 Stata dataset
│   ├── mapping_entrepreneurial_inclusion.do   # Paper 0005 Stata replication code
│   ├── movie_data.csv                         # Paper 0017 original dataset
│   ├── movies.R                               # Paper 0017 R replication code
│   └── ...                                    # Published PDFs, demand-pull paper files
│
└── paper_analysis_output/
    ├── Paper_0005_MappingEntrepreneurial/
    │   ├── DATA.csv                           # Pre-processed analysis dataset (32,647 obs)
    │   ├── Paper_Info_Record.txt              # Metadata, baseline validation, QC log
    │   ├── Paper_Info_Record.pdf              # PDF version
    │   ├── confignotes.txt                    # Model spec, estimator decisions, key vars
    │   ├── scripts/
    │   │   └── simulation_0005.py             # Main simulation script
    │   ├── logs/                              # Run logs (gitignored progress logs)
    │   ├── smoke/                             # Smoke-test workbooks
    │   ├── full_run/                          # Final report workbooks + exports
    │   │   ├── Stroube2025Report_0005.xlsx    # Final workbook (17 sheets, QC passed)
    │   │   ├── paper_info_0005.xlsx           # Structured information record
    │   │   ├── regression_results_0005.xlsx   # Baseline + simulation summary
    │   │   └── figures/                       # Publication-quality visualizations
    │   │       ├── fig1_stability_heatmap_paper0005.png
    │   │       ├── fig2_method_comparison_paper0005.png
    │   │       └── fig3_stability_trajectory_paper0005.png
    │   └── regression_outputs/               # Per-iteration regression text outputs
    │       ├── MCAR/{1pct..50pct}/{var}/{method}/iter{N}_model_{var}.txt
    │       ├── MAR/
    │       └── NMAR/
    │
    └── Paper_0017_StatusConsensus/
        ├── DATA.csv                           # Pre-processed analysis dataset (4,012 obs)
        ├── Paper_Info_Record.txt
        ├── Paper_Info_Record.pdf
        ├── confignotes.txt
        ├── scripts/
        │   └── simulation_0017.py
        ├── logs/
        ├── smoke/
        ├── full_run/
        │   ├── Stroube2024Report_0017.xlsx    # Final workbook (17 sheets, QC passed)
        │   ├── paper_info_0017.xlsx
        │   ├── regression_results_0017.xlsx
        │   └── figures/
        │       ├── fig1_stability_heatmap_paper0017.png
        │       ├── fig2_method_comparison_paper0017.png
        │       └── fig3_stability_trajectory_paper0017.png
        └── regression_outputs/
            ├── MCAR/
            ├── MAR/
            └── NMAR/
```

---

## Papers Analyzed

| Paper ID | Title | Authors | Journal | Year | Focal IV | Baseline Coef | Status |
|---|---|---|---|---|---|---|---|
| 0005 | Mapping Entrepreneurial Inclusion Across US Neighborhoods: The Case of Low-Code E-commerce Entrepreneurship | Stroube & Dushnitsky | SMJ | ~2025 | `log_pop_black_aa` | 0.0307*** | **QC Passed** (2026-04-01) |
| 0017 | Status and Consensus: Heterogeneity in Audience Evaluations of Female- versus Male-Lead Films | Stroube | SMJ | 2024 | `FLead` | 0.0468*** | **QC Passed** (2026-04-01) |

---

## Simulation Protocol

Full methodology is documented in [`RA_MISSING_DATA.pdf`](RA_MISSING_DATA.pdf), covering:
missingness mechanism specifications (MCAR, MAR, NMAR), imputation method implementations
(7 methods), simulation grid defaults, key-variable and MAR-control selection rules,
baseline replication gate, QC checklist, and output contract.

**Simulation grid (per paper):**

| Parameter | Value |
|---|---|
| Mechanisms | MCAR, MAR, NMAR |
| Missingness proportions | 1%, 5%, 10%, 20%, 30%, 40%, 50% |
| Imputation methods | LD, Mean, Reg, Iter, RF, DL, MILGBM |
| Iterations per cell | 30 |
| MAR/NMAR strength | 1.5 |
| MI completed datasets (M) | 5 |
| Total runs per paper | 17,640 |

---

## Reproduction

```bash
# Validate baseline replication only (seconds)
python "paper_analysis_output/Paper_0005_MappingEntrepreneurial/scripts/simulation_0005.py" --mode baseline
python "paper_analysis_output/Paper_0017_StatusConsensus/scripts/simulation_0017.py" --mode baseline

# Smoke test: 2 iterations x reduced missingness levels (~2 minutes)
python "paper_analysis_output/Paper_0005_MappingEntrepreneurial/scripts/simulation_0005.py" --mode smoke
python "paper_analysis_output/Paper_0017_StatusConsensus/scripts/simulation_0017.py" --mode smoke

# Full simulation: 30 iterations x all levels (several hours)
python "paper_analysis_output/Paper_0005_MappingEntrepreneurial/scripts/simulation_0005.py" --mode full
python "paper_analysis_output/Paper_0017_StatusConsensus/scripts/simulation_0017.py" --mode full
```

Scripts read source data from `source_artifacts/` (if `DATA.csv` is absent), write
structured logs to `logs/`, and write the final report workbook to `full_run/`.

---

## Key Output Files

Each paper produces a 17-sheet Excel workbook in `full_run/`:

| Sheet group | Contents |
|---|---|
| **Baseline** | Replicated coefficient table vs. published values; discrepancy flags |
| **Stability heatmaps** (x3) | Per-mechanism (MCAR / MAR / NMAR): B and SS fractions across proportion x method |
| **Coefficient summaries** | Mean estimated coefficient per cell; deviation from baseline |
| **Model comparison** | Method-level aggregates across all key variables and proportions |
| **MI diagnostics** | MILGBM-specific: FMI, Relative Efficiency, average Rubin-pooled SE |
| **QC flags** | Iteration counts, blank-cell detection, sanity check results |

Three publication-quality figures per paper are saved to `full_run/figures/`:

| Figure | Description |
|---|---|
| `fig1_stability_heatmap` | B-proportion heatmap (method x missingness level, MCAR mechanism) |
| `fig2_method_comparison` | Bar chart: mean B-proportion by method across all key variables |
| `fig3_stability_trajectory` | Line chart: B-proportion vs. missingness level per method (first key variable) |

---

## Deliverable Generation

```bash
# Re-generate PDFs, paper_info.xlsx, regression_results.xlsx for both papers
python generate_deliverables.py

# Re-generate all figures (reads from full_run workbooks)
python generate_figures.py
```

---

## Requirements

See [`requirements.txt`](requirements.txt). Key packages: `numpy`, `pandas`, `scipy`,
`scikit-learn`, `pyfixest` (Paper 0005), `statsmodels` (Paper 0017), `lightgbm`,
`tensorflow`, `openpyxl`, `pyreadstat`. Python 3.11+ recommended.
