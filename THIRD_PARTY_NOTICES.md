# Licensing and attribution

This public companion-code release contains files under different license scopes.
The root `LICENSE` grants MIT terms only for WSPR-specific contributions and does
not replace the licenses or attribution notices listed below.

- The source project is based on THUML's Time-Series-Library:
  https://github.com/thuml/Time-Series-Library . Its existing MIT license and
  copyright notice are retained in `licenses/Time-Series-Library-MIT.txt`.
  No exact upstream commit is asserted; the working tree contains research changes.
- `utils/timefeatures.py` retains the Amazon/GluonTS Apache-2.0 header. The license
  text is included in `licenses/Apache-2.0.txt`.
- `data_provider/m4.py` retains its Element AI / N-BEATS CC-BY-NC-4.0 notice.
  Its license text is included in `licenses/CC-BY-NC-4.0.txt`. M4 is not a paper
  dataset; the file is retained because the historical loader eagerly imports it.
  Its noncommercial restriction must not be replaced by a repository-wide MIT claim.
- PatchTST and iTransformer sources retain their original paper links and comments.
  Dependency packages retain their own licenses; installing them does not relicense
  them under the repository's terms.

License texts were obtained from the official Apache and Creative Commons license
URLs on 2026-09-18. Per-file attribution in copied sources is preserved. The authors
should review any additional upstream notices before publication.

Dataset access rights are independent of code licenses. No raw dataset or checkpoint
is distributed in this candidate package. Repository visibility and any later
archival release remain separate author-controlled actions.
