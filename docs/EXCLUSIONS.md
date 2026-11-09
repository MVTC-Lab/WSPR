# Excluded from this release candidate

- Raw and prepared PVOD/GEFCom/PVDAQ/HRRR datasets, cache tensors and prediction/truth arrays.
- Checkpoints, binary tensors, training logs, plots and generated experiment folders.
  The author-provided model overview in `assets/wspr_architecture.svg` is included as documentation.
- SSH/SCP helpers, server identities/addresses, keys, passwords, access tokens,
  connection profiles, environment secrets and IDE settings.
- Original `.git` history and unrelated research code/experiments.
- Manuscript drafts, PDFs, Word/PowerPoint files, peer-review reports, author-private
  declarations, conversations, attachment caches and task handoff packages.
- Superseded GEFCom numerical evidence and legacy detached direct comparators.
- Models not used in the six primary study groups, except genuine import dependencies.

An allowlist in `.gitignore` exposes only selected text-code/documentation formats, the named architecture SVG,
and the explicitly named scalar-result JSON files. `RELEASE_MANIFEST.json` provides the stricter exact
file inventory checked by `scripts/verify_release.py`. Never force-add excluded
files. Material new files require explicit review and a manifest update.

Raw data should be obtained from the authorized providers. Large weights, if later
approved for sharing, need a separately designed release/archive policy; they
should not simply be forced into Git history.
