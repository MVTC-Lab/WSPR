# Technical checks, 19 September 2026

Executed in an isolated release directory in the existing research environment:
Python 3.8.20, PyTorch 2.4.0, NumPy 1.23.5, pandas 1.5.3. CPU was used for synthetic
model forward checks. Original experiments, checkpoints and data were read only.

| Check | Observed result |
|---|---|
| Six model classes across four task shapes | 24/24 finite outputs of expected shape |
| Future-target perturbation (+10,000) | Maximum output change 0 in all 24 cases |
| Synthetic chronological loader | Window counts 589 / 85 / 185 for train / validation / test |
| Training-only StandardScaler statistics | Match first 700 of 1,000 synthetic rows |
| Validation/test first forecast targets | Start at synthetic row indices 700 / 800 |
| Runner help / unacknowledged training refusal | Both passed |
| PVOD raw-to-prefix preparation | 10/10 byte-identical frozen CSVs |
| Corrected GEFCom raw-to-input preparation | 3/3 byte-identical frozen CSVs |
| GEFCom negative increments before clipping | VAR169: 597; VAR175: 0; VAR178: 974; VAR228: 180 |
| Scalar run registry | 468 unique records, 156 unit/pathway groups, 24 task/pathway groups |

PVOD raw archive SHA256:
`0f726b42235bf99a2c118a3cb7544722150489b01af3a2853a7c9778b538883e`.
GEFCom raw archive SHA256:
`d68d957270edd93b26a37d0f9b5e901f942abdf34c75eacbe14e417beb16e154`.
Per-input frozen hashes are in `configs/primary_run_registry.json`.

No training, backward pass, model-source edit or checkpoint replacement was
performed. This is not a clean-room environment test, a CUDA reproducibility
claim, a full replay of paper figures or independent validation of performance.
The scripts let reviewers rerun the applicable checks. Full local test reports
are retained by the authors; private infrastructure paths are not published.
