# Reproduction status and release gate

This October candidate supports source inspection, scalar reaggregation and
archive-assisted execution. It is not a complete evidence or data archive.

| Component | Status |
|---|---|
| Six primary model classes | Publication filenames plus original source hashes; only imports, diagnostic text and naming comments changed during the rename |
| Forecast trainer/data loader dependencies | Included; release-only changes documented |
| Environment versions | Historical CUDA versions plus a separately tested Windows CPU snapshot; not a universal cross-platform lock |
| Main task matrix | 468 sanitized scalar/provenance records; command printer is not the original argv archive |
| Raw data and frozen checkpoints | Not distributed; access/rights require confirmation |
| PVOD raw-to-frozen conversion | Rebuilt; 10/10 prepared prefix hashes match frozen inputs |
| Corrected GEFCom preprocessing | Rebuilt; 3/3 corrected model-input hashes match frozen inputs |
| Frozen diagnostic aggregate | Included unchanged; no replay inputs included |
| Independent validation | 108 scientific configurations, scalers and 9 complete scalar tables; loader and archive-assisted training/analysis entry points |
| PVOD calendar/suffix/active-generation | 14 scalar tables and seed-first/equal-site reaggregation; raw-array pipelines not packaged |
| Ablation and plotting reproduction | Not included in this initial candidate |
| Training or end-to-end reproduction in release directory | Not run |
| Synthetic CPU/runtime/data-contract checks | Historical report plus October clean-install report; neither is CUDA accuracy replay |

Before advertising full reproducibility, recover and sanitize the exact run
configurations, resolve checkpoint/data access, and package the remaining
download/QC/cache-building, analysis and plotting entry points. The scientific
scope is detailed in `PAPER_CODE_MAP.md`.
Do not replace absent results with newly generated or favorable subsets.

## Sharing checklist

1. Confirm any remaining institutional/redistribution approvals. The root MIT
   license covers WSPR-specific contributions only; the retained M4 loader remains
   subject to CC-BY-NC-4.0 unless it is removed in a later reviewed release.
2. Review `git status` and every candidate path. `.gitignore` is not a secret scanner
   and can be bypassed with `git add -f`. Never upload the original workspace wholesale.
3. Run `python scripts/verify_release.py`; inspect its failures, not just exit status.
4. In a clean pinned environment, test CLI imports and numerical/data contracts.
   No fresh training is necessary simply to publish source; do not imply retraining
   or exact reproduction has already happened.
5. Confirm the target remains the intended public repository before any future push.
6. Verify the public URL and versioned tag from an unauthenticated session; public
   access does not grant redistribution rights for excluded datasets or artifacts.
7. Keep the paper's code-availability statement synchronized with the reviewed tag.
   A later DOI archive is useful, but no DOI or permanent availability is promised
   by GitHub alone.
