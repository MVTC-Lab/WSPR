# Release-only adaptations

The original workspace and experiments were not changed. The initial candidate
preserved seven research-server model/helper files and their exact hashes. During
publication preparation, misleading `TimeMixer_7*` filenames were replaced with
`WSPR`, `DirectNWP`, and explicit backbone suffixes; the byte-identical duplicate
helper was removed. The required imports, diagnostic text and naming comments were
updated without changing tensor operations. Both pre-rename provenance hashes and
current release hashes are recorded in `RELEASE_MANIFEST.json`.

The initial candidate adapted release copies of three runtime files:

1. `exp/exp_basic.py`: restrict eager imports/registry to the six primary model
   classes; reject other models. The original device-selection and build flow remain.
2. `run.py` (renamed from the research entry point): remove imports/dispatch for nonforecasting tasks, restrict
   use to `long_term_forecast` plus `custom`, and require explicit
   `--allow_training`. Original parser hyperparameter defaults remain; use the
   printed paper-setting command, not the bare runner defaults.
3. `exp/exp_long_term_forecasting.py`: output plots/arrays to relative `outputs/`
   paths instead of a research-server absolute path; require exact checkpoint
   lookup instead of silently searching similarly named experiments. Training loss,
   optimizer, target masking and metric calculations were not changed.

The dependency list records selected observed versions rather than copying the
stale upstream requirement list. Standard-library helpers verify the package,
validate input hashes, reaggregate scalar records and print commands. Two portable
preparation wrappers call unchanged research preprocessing scripts and require new
output directories. Synthetic smoke tests use PyTorch in evaluation mode. None
of these helpers trains a model or pushes to GitHub.

Original copies of the three adapted files are retained outside the upload directory
for the owner's local audit. Model hashes can be compared directly against the
frozen diagnostic audit. A clean-environment runtime/numerical equivalence check
was still needed at that stage. Existing-environment runtime and raw-to-frozen data checks passed
as documented in `TEST_REPORT_20260919.md`; these do not replace a clean installation.

## October 2 publication preparation

- Preserved the previous candidate and its dirty Git state in a local backup;
  no original experiment, model architecture, data or manuscript was edited.
- Updated the title, scope and coverage map for PVDAQ--HRRR and the post-hoc
  analyses. Exported all named scalar tables without raw arrays or private paths.
- Copied the frozen independent loader unchanged. Relocated `common.py` paths
  and input verification to the release directory; original input lock/hash
  requirements remain. Private scheduler/SSH transport scripts are excluded.
- Added explicit path arguments and a training opt-in guard to the independent
  CLI. Training math, optimizer settings, validation selection and metric formulas
  were not changed. New-fit config metadata identifies the release manifest and
  relocated implementation, instead of claiming original-deployment identity.
- Added scalar reaggregation and independent synthetic tests. Fixed Windows
  UTF-8 decoding in smoke-test subprocesses; no model logic was changed.
- Pinned torch 2.4.0 in the installation requirements to prevent a transitive
  dependency from silently selecting 2.4.1. CPU and historical CUDA builds remain
  distinct. See `TEST_REPORT_20261002.md` for observed results.
- WSPR-specific contributions now use a scoped MIT license. Third-party exceptions
  and notices remain explicit; repository visibility is a separate action.
- Renamed the public modules to `WSPR`, `WSPR_iTransformer`, `DirectNWP` and
  `DirectNWP_iTransformer`, and renamed the entry point to `run.py`. Historical
  result identifiers remain untouched and are resolved through an explicit alias
  map. A separate 24-case deterministic CPU comparison produced identical model
  state and output fingerprints before and after the rename.
- Delayed training-stack imports in `run.py` until after argument parsing, so
  `python run.py --help` works before the research dependencies are installed.
  Actual training/evaluation requirements and computations are unchanged.
