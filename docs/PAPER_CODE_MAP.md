# Manuscript-to-code coverage, October 2026 candidate

The manuscript title is *Direct and residual NWP integration for photovoltaic
power forecasting: Metric, subset and temporal dependence*. Section labels below
are conceptual; figure/table numbering can differ between manuscript layouts.

| Evidence | Included | Directly rerunnable here | Not implied |
|---|---|---|---|
| Legacy PVOD/corrected GEFCom main comparison | Six classes in six clearly named model files; pre-rename source hashes retained; 468 scalar records; input builders/hashes | `scripts/summarize_primary.py`; synthetic model/loader checks | Original 468 argv records, trained weights and prediction replay |
| Independent PVDAQ--HRRR | 108 configs; train scalers; unchanged execution-lock record; frozen loader; relocated fit/analysis implementations; 9 scalar tables | `scripts/summarize_independent.py`; synthetic contracts; archive-assisted runners with additional authorized files | Download/QC/cache reconstruction, publicly supplied caches/weights, CUDA accuracy rerun |
| Calendar-climatology PVOD control | 300 scalar records = 120 new fits + 180 reused references, all paired summaries | `scripts/summarize_legacy_posthoc.py` reaggregates seed/site/task means | Raw-data preparation, 120-fit rerun, bootstrap interval recalculation |
| PVOD suffix temporal sensitivity | 360 reused checkpoints represented by 1800 scalar rows over 5 subsets | Same scalar checker; 600 site and 60 task rows | Untouched independent validation or checkpoint replay |
| PVOD active-generation thresholds | 360 reused checkpoints x 4 thresholds = 1440 scalar rows | Same scalar checker; 480 site and 48 task rows | Fresh model fits or regeneration of masks/predictions |
| Frozen NWP/gate diagnostics | Existing diagnostic aggregate JSON | Inspection and package hash validation | Per-window recomputation or retrained constant-gate controls |
| Constraint ablation and paper figures | Existing core model ablation interface, but not a full reproduction package for these displays | Source inspection only | Complete 180-fit ablation launcher, full plot/LaTeX regeneration |

`results/independent_validation.json` and `results/legacy_posthoc_tables.json`
preserve numeric CSV text under named columns, with SHA256 identifiers for the
source tables. Runtime readers explicitly convert named metrics. JSON is a
transport format, not a relabeling of MSE/MAE or a new experiment.

All scalar tables are complete for the listed study scopes, including negative
results. A row count does not necessarily equal a fit count. All-subset and
threshold rows must not be added together as new independent runs.

This package supports source inspection and a substantial numeric audit, not a
claim of universal one-command reproduction. Additional artifact requests should
specify exact study, run keys and required outputs. Provider data terms apply
independently of the eventual code license.
