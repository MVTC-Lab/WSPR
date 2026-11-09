# GitHub package preparation 9 November 2026

The repository retains the six previously checked release model implementations
and the forecasting runtime unchanged. Model names map to the historical run
identifiers through `scripts/summarize_primary.py`; model renaming changed imports
and names, not tensor operations. Historical deterministic CPU fingerprints and
runtime tests are documented in `TEST_REPORT_20261002.md`.

This preparation adds a Chinese landing README and an English README, inserts
the current manuscript's displayed MSE/MAE task means, and synchronizes the vector
architecture image with the confirmed manuscript figure source. The selected
original scalar registry can differ from later frozen-array rescoring at small
floating-point precision; the displayed four-decimal task means agree.

No dataset, checkpoint, prediction array, training log, manuscript PDF, account
credential or private Git history is packaged. The `data_provider` directory is
Python loading code, and `results` contains aggregate scalar evidence. Model
training has not been performed to prepare this upload. Existing licenses and
third-party notices are retained unchanged. Publication and repository visibility
are controlled by the repository owner; assembling these files does not upload them.
