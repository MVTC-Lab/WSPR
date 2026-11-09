# PVDAQ--HRRR independent validation

This guide describes the executed 108-fit study, not a new experiment or a claim
that residual correction wins. The four equal-site task comparisons favor
Direct-NWP on both MSE and MAE. Both NWP pathways improve over History-only.

## Frozen scientific design

- PVDAQ systems: 2107, 9068, 14200. Hourly measured AC power is averaged over
  complete hours; the main target is then `max(hourly_measured_AC_kw, 0)`.
  This is not reconstructed gross generation. Raw negative values are retained
  in the research archive; raw-AC sensitivity evaluates the same predictions,
  not a new raw-target fit.
- All paths receive 96 hours of historical power. Seven standardized-space
  zero-padding channels preserve frozen model interfaces; they are not observed
  historical weather. Direct/Residual additionally use forecast TMP, RH, UGRD,
  VGRD, PRES, TCDC, DSWRF, in that order.
- UTC split boundaries: train 2020-01-01 to 2021-10-01; validation 2021-10-01
  to 2022-01-01; test 2022-01-01 to 2023-01-01. See the loader for interval-end
  inclusivity and complete-history/target-window requirements. No percentage
  split of the newly supplied data is permitted for a frozen-study replay.
- Origins are nominal HRRR initialization cycles 00/06/12/18 UTC; leads are
  1 through 4 or 24 hours. The 17,635 retained site/horizon/split windows preserve
  forecast vintages. Nominal `init=origin` does not prove real-time availability;
  forecast-release/telemetry latency has not been reconstructed. Instantaneous
  meteorology and hour-ending power averages have different temporal support.
- Per-site power and NWP scalers use training rows only. Repeated horizon views
  do not reweight the same train NWP initialization/lead pair.
- Three sites x two horizons x two backbones x three paths x seeds 2024/2025/2026
  gives 108 fits. Twelve initial engineering-gate fits are included in this total.
- Adam, MSE, batch size 32, up to 20 epochs, patience 5, learning rate 0.001 and the
  frozen type1 schedule; strict validation improvement retains the earlier epoch
  on ties. No test-based model selection or prediction clipping. Exact settings
  are in `configs/independent_validation.json`, not inferred from generic defaults.

The plan was internally specified before these fits. It was not externally
preregistered and is not an independent research-team replication. The historical
manifest's `DESIGN_RECORDED_EXECUTION_NOT_AUTHORIZED` strings are preserved from
the preparation stage; the later execution lock records that training was
authorized. Neither historical field is current permission to start a new fit.

## What works without private data

```bash
python scripts/summarize_independent.py
python scripts/independent_validation/test_contracts.py --output outputs/independent_contracts.json
```

The first reaggregates all 108 scalar rows into 36 site/pathway and 12 task/pathway
rows and checks the 24 paired task effects and 1512 lead rows. It also checks named
MSE/MAE/RMSE and physical-unit conversion identities. It does not reconstruct
predictions or recalculate the 144 stored time intervals from scratch.

The second uses synthetic data and 12 CPU forward cases. It checks target
clipping, common history, future-target exclusion, split/vintage rejection,
train-only scaling and the opt-in training guard. It neither fits nor evaluates
research checkpoints.

## Archive-assisted entry points (additional artifacts required)

`scripts/independent_validation/` includes the exact frozen loader and relocated
training/analysis scripts. They are not a new end-to-end HRRR downloader or cache
builder. To use the executable archive route, obtain authorized access to the
**complete original input package**: 18 cache NPZs, scalers, run-manifest CSV,
execution lock and all metadata files bound by its `package_sha256` map. Analysis
also needs all 108 completed run artifacts. These files are **not distributed**
by this repository; access/redistribution remains subject to rights and author
coordination. Do not generate invented hash-compatible placeholders.

Inspect the CLI safely:

```bash
python scripts/independent_validation/train_one.py --help
python scripts/independent_validation/analyze.py --help
```

For an authorized archive copy only, the analysis command is:

```bash
python scripts/independent_validation/analyze.py --package /path/to/authorized/package --results /path/to/copied/run_results
```

It checks original input/artifact hashes and writes an analysis directory only
if absent; an existing valid analysis is checked without replacement. Use a
separate working copy for generated outputs. The relocated runner additionally
requires explicit `--allow-training`, run ID, package/results paths and GPU
visibility, and retains the Linux/Python 3.8/PyTorch 2.4.0 CUDA constraints. No
training command is executed as part of release preparation.

The historical `execution_lock.json` is preserved byte-for-byte. Current script
hashes live separately in `RELEASE_MANIFEST.json`; the revised checker validates
the release adaptations instead of requiring unpublished transport scripts.
All original package hashes remain mandatory. New fits also record the release
manifest hash and an implementation-scope label; they are not automatically
bitwise-identical copies of the historical runs.

## Statistical and unit boundaries

Primary metrics are per-site train-standardized MSE and MAE. kW MAE/RMSE use the
recorded train scale; no unverified installed-capacity normalization is claimed.
Seed errors are averaged within each site, then sites receive equal weight;
seed sample SD and observed site ranges are retained. Do not turn seeds into
independent geographical samples. Paired candidate-minus-reference differences
are negative when the candidate is better. Relative improvements use
`(reference - candidate) / reference`; both site-ratio means and task-mean ratios
are stored and must not be conflated.

The original analysis uses paired circular 7/14-day calendar blocks, 10,000 draws,
seed 20260927, over the 365-day 2022 origin calendar including empty days. These
are descriptive conditional within-site percentile intervals, not a three-site
population interval or multiplicity-controlled significance test. Deterministic
persistence and train-only month/hour climatology use the same eligible windows.
