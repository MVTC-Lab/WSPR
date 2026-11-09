# Clean CPU installation and publication-candidate tests

Date: 2 October 2026. A new isolated virtual environment was created from an
existing Python 3.8.20 interpreter, without inheriting its site packages.
Dependencies were freshly installed from PyPI and the official PyTorch CPU index.
This tests a clean dependency installation, not a new operating system or GPU.
Observed versions: Python 3.8.20, torch 2.4.0+cpu, NumPy 1.23.5, pandas 1.5.3.
The full observed package list is in `configs/environment_cpu_windows_py38_20261002.json`.

| Check | Result |
|---|---|
| Installed dependency consistency (`pip check`) | PASS |
| Legacy synthetic CPU model/task forwards | 24/24 finite, expected shapes |
| Legacy future-target perturbation | Output difference 0 in all 24 cases |
| Synthetic chronological loader | 589 / 85 / 185 train/validation/test windows |
| Legacy train-only scaling and CLI/training guard | PASS |
| Independent synthetic data/metric/guard tests | 9/9 PASS |
| Independent model/task forwards | 12/12 finite, future-target invariant, state unchanged |
| Original primary scalar registry | 468 records; seed-first/equal-unit aggregation PASS |
| Independent scalar tables | 108 records; 36 site/pathway, 12 task/pathway, 24 paired task and 1512 lead rows checked |
| Three legacy post-hoc scalar analyses | Seed-first/equal-site means match all stored task tables |

Machine-readable synthetic results are in `configs/runtime_checks_20261002.json`.
The separate release checker validates current inventory/hash/syntax identity.

Two environment/portability issues were observed and resolved in the release
candidate: the initially unpinned transitive torch dependency selected 2.4.1,
so torch 2.4.0 was installed explicitly and added to requirements; the first
legacy smoke run reached a Windows GBK/UTF-8 subprocess decoding error, so the
test wrapper now explicitly uses UTF-8 for the child stream. The final legacy
smoke run passed. These changes do not alter model forward code or training math.
Third-party AMP deprecation warnings may still appear with the frozen dependencies.

No optimizer step, backward pass, research data download, model fitting or CUDA
accuracy replay was performed. September raw-to-frozen hash checks remain
historical evidence; this October run did not repeat preprocessing against raw
provider archives. The independent archive-assisted training/analysis CLI was
checked for import/help/guard behavior and source adaptation, not a new 108-fit
run or full private-artifact replay. A passing test is not scientific author
approval, a license clearance or a guarantee of identical GPU results.

After the publication modules were renamed, all 24 model/task cases were also
instantiated from the pre-rename candidate and the renamed candidate in separate
processes with identical CPU seeds and inputs. Parameter counts, complete state
tensor SHA256 fingerprints and output tensor SHA256 fingerprints matched in all
24 cases. The regular 24-case smoke test and the independent 12-case/9-test suite
then passed under the new names. This verifies the tested CPU rename equivalence;
it is not a new checkpoint, training or CUDA result reproduction.

The public `run.py` now delays numerical/training imports until after CLI
argument parsing. Consequently `python run.py --help` was also checked with the
machine's unrelated Python 3.13.2 interpreter lacking scikit-learn and returned
help successfully. This convenience check does not make Python 3.13 a supported
training environment; fitting and evaluation still require the documented
Python 3.8 dependency environment.
