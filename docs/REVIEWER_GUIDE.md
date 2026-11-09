# Reviewer inspection guide

This package supports source inspection, input-version validation and scalar
result reaggregation. It does not contain the full internal research archive.

## Start without datasets or training

Run from the extracted repository root:

```bash
python scripts/verify_release.py
python scripts/summarize_primary.py
python scripts/summarize_independent.py
python scripts/summarize_legacy_posthoc.py
```

These use only the Python standard library (Git is needed only for a Git checkout).
The first checks the reviewed file inventory, SHA256 values, syntax and model
identity. The second recomputes MSE, MAE and R2 from 468 frozen scalar records:
three seeds within each physical unit, then an equal mean across units. It does
not recompute per-run metrics from predictions. The GEFCom registry contains only
the corrected 108 runs, never the legacy counterpart. No pooled 13-unit test is
introduced. MSE and MAE are standardized-scale errors.

The additional commands verify all 108 independent scalar records (including
physical-unit errors) and three legacy post-hoc scalar analyses. None starts a
model, downloads data or recomputes metrics from prediction arrays. All result
directions are retained, including Direct-NWP's lower independent task means.
Consult `PAPER_CODE_MAP.md` for the exact limits by analysis.

`configs/primary_run_registry.json` retains source-file checksums, data hashes and
checkpoint setting identifiers without personal server paths. All 312 NWP records
also retain checkpoint/prediction/truth checksums. History-only artifact hashes
are not added where not verified in that diagnostic manifest. Raw argv archives,
weights and prediction arrays are not implied by these identifiers.

## Test implementation and inputs

Install the documented environment, then run `scripts/smoke_test.py`. This uses
random synthetic inputs in evaluation mode on CPU, without backward passes,
optimizers, checkpoints or a training process. It checks finite output shapes,
future-target invariance, chronological window boundaries and train-only scaling.
It does not demonstrate accuracy or reproduce a frozen checkpoint.
Run `scripts/independent_validation/test_contracts.py` for the independent loader
contracts and 12 additional CPU model-task cases. The test uses synthetic data.

Obtain data under the providers' terms, follow `DATA.md`, and run `verify_data.py`
before any result-reproduction claim. Neither preparation entry point downloads
data or changes the original source archive. Both require a new output directory.

## If exact replay is requested

Contact the corresponding author, Feng Yan (yf@xju.edu.cn), for a versioned code
snapshot and for clarification of artifact access. Ask for the exact run key,
archived argument/configuration record, checkpoint and expected prediction hash.
Availability of public code does not automatically provide these additional
artifacts or grant dataset redistribution rights. Requests for excluded artifacts
must be handled through an appropriate author and journal-approved channel.

The public repository has passed a bounded rights, license and access review. See
`THIRD_PARTY_NOTICES.md`; there is no blanket MIT license for the entire package.
Author scientific approval and journal submission are separate from passing the
technical checks.
