# Data access and input contract

Obtain PVOD, GEFCom2014 Solar, PVDAQ and HRRR from their authorized providers, following each
dataset's access and redistribution terms. No dataset is included or automatically
downloaded. The source manuscript cites the original dataset records; this draft
does not invent a new download URL or assert that redistribution is permitted.

The expected prepared CSV has `date` first and `OT` last. Do not add station metadata,
observed future weather, or reliability labels to the feature matrix.

## PVOD

Ordered numeric columns: `nwp_globalirrad`, `nwp_directirrad`, `nwp_temperature`,
`nwp_humidity`, `nwp_windspeed`, `nwp_winddir_sin`, `nwp_winddir_cos`, `nwp_pressure`,
`OT`. Thus `enc_in=dec_in=c_out=9`; future-known dimensionality is 8 for NWP pathways
and 0 for History-only. Use `freq=t`, 96 historical steps and horizons 16 or 96.

The frozen main records use the **PVOD_pre80** prepared representation: retain the
first floor(0.8*N) rows of the original prepared station series, then the loader
splits that retained series 70/10/20 chronologically (with integer rounding).
Using the entire original station series changes the experiment.
Preparation shifts raw timestamps by +8 h, averages duplicate timestamps, encodes
wind direction as sine/cosine, and retains the longest continuous 15-minute segment.
There is no hourly-to-15-minute interpolation. `lmd_*` fields are not known-future
inputs. The older `prepare_pvod_v4.py` adds a measured-weather-derived auxiliary
label and is intentionally not distributed as the current preparation pipeline.
The wrapper below imports the unchanged original NWP-only preparation routine,
then applies the frozen prefix rule. It refuses an existing output directory.

```bash
python scripts/prepare_pvod.py --raw-dir /path/to/authorized/raw_archives --out-dir outputs/pvod_prepared
python scripts/verify_data.py --dataset PVOD --data-dir outputs/pvod_prepared/PVOD_pre80
```

All ten station outputs matched the frozen CSV SHA256 values on 19 September 2026
using NumPy 1.23.5 and pandas 1.5.3. Byte identity can depend on numeric/CSV-library
versions. A mismatch must be investigated, not silently treated as the same input.

For the printed example, place the approved file at
`data/PVOD_pre80/PVOD_station00_nwp.csv`. The study covers station00 through station09.

## Corrected GEFCom2014 Solar

Ordered numeric columns: `VAR78`, `VAR79`, `VAR134`, `VAR157`, `VAR164`, `VAR165`,
`VAR166`, `VAR167`, `VAR169`, `VAR175`, `VAR178`, `VAR228`, `OT`.
Use 13 total channels, 12 known-future channels, `freq=h`, (history,horizon) pairs
(96,24) and (168,96), and zones 01 through 03.

VAR169/175/178/228 must use the corrected midnight-cycle, within-cycle
deaccumulation. Do not train on the superseded cumulative representation and label
it corrected. The archived correction script is in `scripts/wspr_q1/` and accepts
`--raw_zip`, `--old_dir`, `--out_dir`, and `--audit_dir`. It requires a prior legacy
export only for invariance checking; `prepare_gefcom2014_solar.py` is supplied as
that prerequisite reference, NOT as the final training representation. Both scripts
keep their historical relative defaults; supply all path flags explicitly to avoid
writing outside this repository. Read the script options before running either.

The portable wrapper creates both prerequisite and corrected exports under a new
directory, with all historical path defaults overridden:

```bash
python scripts/prepare_gefcom.py --raw-zip /path/to/authorized/GEFCom2014.zip --out-dir outputs/gefcom_prepared
python scripts/verify_data.py --dataset GEFCom --data-dir outputs/gefcom_prepared/GEFCom2014Solar_deaccum_v1
```

All three corrected outputs matched frozen CSV hashes in the existing research
environment on 19 September 2026. The legacy output is an intermediate for
invariance checks only. Do not use it as the corrected model input.
Place approved corrected files under `data/GEFCom2014Solar_deaccum_v1/`, or pass
their actual root to the model runner. Example: `GEFCom2014Solar_zone01_nwp.csv`.

## PVDAQ--HRRR

This study does not use the generic percentage-split CSV loader above. Use the
origin-indexed loader and explicitly frozen UTC dates described in
`INDEPENDENT_VALIDATION.md`. The public candidate includes configs, scalers and
scalar outputs, not raw/prepared data, NPZ caches or the full download/QC pipeline.
Do not fabricate historical observed weather by relabeling forecast channels.

Source records cited by the manuscript: PVOD `https://doi.org/10.11922/sciencedb.01094`,
PVDAQ `https://doi.org/10.25984/1846021`, HRRR documentation
`https://rapidrefresh.noaa.gov/hrrr/` and archive `noaa-hrrr-bdp-pds`.
These references do not grant redistribution permission or guarantee present
access. Follow original-provider terms and resolve exact versions before reuse.

## Leakage, scaling and safety

`Dataset_Custom` fits StandardScaler only on the chronological training segment.
Forecast windows may borrow historical context across split boundaries, but labels
belong to the appropriate split. The NWP models sanitize future target values to
zero. Data contract validation and an exact match to frozen data hashes remain
necessary before claiming numerical reproduction.
