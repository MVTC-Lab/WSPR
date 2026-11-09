from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPERS_ROOT = PROJECT_ROOT.parent
DEFAULT_RAW_ZIP = PAPERS_ROOT / "datasets/PV/raws_datasets/GEFCom2014/GEFCom2014.zip"
DEFAULT_OUT_DIR = PAPERS_ROOT / "datasets/PV/processed_research/all_stations/GEFCom2014Solar"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "experiments/wspr_q1/stage7_gefcom2014/reports"

NWP_COLUMNS = [
    "VAR78",
    "VAR79",
    "VAR134",
    "VAR157",
    "VAR164",
    "VAR165",
    "VAR166",
    "VAR167",
    "VAR169",
    "VAR175",
    "VAR178",
    "VAR228",
]

NWP_DESCRIPTIONS = {
    "VAR78": "ECMWF total column liquid water (tclw)",
    "VAR79": "ECMWF total column ice water (tciw)",
    "VAR134": "ECMWF surface pressure (SP)",
    "VAR157": "ECMWF relative humidity at 1000 mbar",
    "VAR164": "ECMWF total cloud cover (TCC)",
    "VAR165": "ECMWF 10 metre U wind component",
    "VAR166": "ECMWF 10 metre V wind component",
    "VAR167": "ECMWF 2 metre temperature",
    "VAR169": "ECMWF surface solar radiation downwards (SSRD)",
    "VAR175": "ECMWF surface thermal radiation downwards (STRD)",
    "VAR178": "ECMWF top net solar radiation (TSR)",
    "VAR228": "ECMWF total precipitation (TP)",
}


def read_solar_zip(raw_zip: Path) -> zipfile.ZipFile:
    if not raw_zip.exists():
        raise FileNotFoundError(f"Missing raw GEFCom2014 zip: {raw_zip}")
    with zipfile.ZipFile(raw_zip) as outer:
        candidates = [name for name in outer.namelist() if name.endswith("GEFCom2014-S_V2.zip")]
        if not candidates:
            raise FileNotFoundError("GEFCom2014-S_V2.zip was not found inside the outer archive.")
        data = outer.read(candidates[0])
    return zipfile.ZipFile(io.BytesIO(data))


def load_task(solar_zip: zipfile.ZipFile, task: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    train_name = f"Solar/Task {task}/train{task}.csv"
    predictor_name = f"Solar/Task {task}/predictors{task}.csv"
    instructions = solar_zip.read("Solar/Instructions.txt").decode("utf-8", errors="replace")
    train = pd.read_csv(solar_zip.open(train_name))
    predictors = pd.read_csv(solar_zip.open(predictor_name))
    return train, predictors, instructions


def normalize_task(train: pd.DataFrame, predictors: pd.DataFrame) -> pd.DataFrame:
    required_train = {"ZONEID", "TIMESTAMP", "POWER"}
    required_predictors = {"ZONEID", "TIMESTAMP", *NWP_COLUMNS}
    missing_train = required_train.difference(train.columns)
    missing_predictors = required_predictors.difference(predictors.columns)
    if missing_train:
        raise ValueError(f"train file missing required columns: {sorted(missing_train)}")
    if missing_predictors:
        raise ValueError(f"predictors file missing required columns: {sorted(missing_predictors)}")

    train = train[["ZONEID", "TIMESTAMP", "POWER"]].copy()
    predictors = predictors[["ZONEID", "TIMESTAMP", *NWP_COLUMNS]].copy()
    merged = train.merge(predictors, on=["ZONEID", "TIMESTAMP"], how="inner", validate="one_to_one")
    merged["date"] = pd.to_datetime(merged["TIMESTAMP"], format="%Y%m%d %H:%M", errors="raise")
    merged = merged.drop(columns=["TIMESTAMP"]).rename(columns={"POWER": "OT"})
    merged = merged[["ZONEID", "date", *NWP_COLUMNS, "OT"]]

    for col in [*NWP_COLUMNS, "OT"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["OT"] = merged["OT"].clip(lower=0)
    return merged.sort_values(["ZONEID", "date"]).reset_index(drop=True)


def audit_zone(df: pd.DataFrame, zone: int) -> dict[str, object]:
    zdf = df[df["ZONEID"] == zone].copy()
    times = zdf["date"].sort_values()
    diffs = times.diff().dropna()
    expected_step = pd.Timedelta(hours=1)
    expected_rows = int(((times.max() - times.min()) / expected_step) + 1) if len(times) else 0
    return {
        "zone": f"zone{zone:02d}",
        "rows": len(zdf),
        "expected_hourly_rows": expected_rows,
        "start": times.min().strftime("%Y-%m-%d %H:%M:%S"),
        "end": times.max().strftime("%Y-%m-%d %H:%M:%S"),
        "missing_cells_nwp_file": int(zdf[["date", *NWP_COLUMNS, "OT"]].isna().sum().sum()),
        "duplicate_time": int(times.duplicated().sum()),
        "non_1h_diffs": int((diffs != expected_step).sum()),
        "ot_min": float(zdf["OT"].min()),
        "ot_max": float(zdf["OT"].max()),
        "ot_zero_pct": float(zdf["OT"].eq(0).mean() * 100),
    }


def write_outputs(df: pd.DataFrame, task: int, raw_zip: Path, out_dir: Path, report_dir: Path, instructions: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    audit_rows = []
    zones = sorted(int(z) for z in df["ZONEID"].unique())
    for zone in zones:
        zdf = df[df["ZONEID"] == zone].drop(columns=["ZONEID"]).copy()
        zdf["date"] = zdf["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        zone_name = f"zone{zone:02d}"

        nwp_file = f"GEFCom2014Solar_{zone_name}_nwp.csv"
        history_file = f"GEFCom2014Solar_{zone_name}_history.csv"
        nwp_df = zdf[["date", *NWP_COLUMNS, "OT"]]
        history_df = zdf[["date", "OT"]]
        nwp_df.to_csv(out_dir / nwp_file, index=False, encoding="utf-8-sig")
        history_df.to_csv(out_dir / history_file, index=False, encoding="utf-8-sig")

        audit = audit_zone(df, zone)
        audit_rows.append(audit)
        manifest_rows.append(
            {
                "dataset": "GEFCom2014Solar",
                "station": zone_name,
                "task": task,
                "nwp_file": nwp_file,
                "history_file": history_file,
                "rows": audit["rows"],
                "start": audit["start"],
                "end": audit["end"],
                "freq": "1h",
                "target": "OT",
                "historical_target_allowed": "yes",
                "future_target_allowed": "no",
                "future_covariates_allowed": ",".join(NWP_COLUMNS),
                "source_zip": str(raw_zip),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    audit = pd.DataFrame(audit_rows)
    known_future = pd.DataFrame(
        [{"column": col, "description": NWP_DESCRIPTIONS[col], "future_known": True} for col in NWP_COLUMNS]
    )
    known_future = pd.concat(
        [
            known_future,
            pd.DataFrame(
                [
                    {
                        "column": "OT",
                        "description": "Solar power target copied from train*.csv; only historical OT is allowed as an input.",
                        "future_known": False,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    manifest_path = out_dir / "manifest_gefcom2014solar.csv"
    audit_path = report_dir / "gefcom2014solar_quality_audit.csv"
    known_future_path = out_dir / "known_future_columns_gefcom2014solar.csv"
    instruction_path = report_dir / "gefcom2014solar_instructions_excerpt.txt"
    report_path = report_dir / "gefcom2014solar_prepare_report.md"

    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    known_future.to_csv(known_future_path, index=False, encoding="utf-8-sig")
    instruction_path.write_text(instructions, encoding="utf-8")

    lines = [
        "# GEFCom2014 Solar Prepared Dataset",
        "",
        f"Source archive: `{raw_zip}`",
        f"Solar task used: Task {task}",
        "",
        "## Leakage Policy",
        "",
        "- `OT` is the solar power target renamed from `POWER` in `train*.csv`.",
        "- Historical `OT` can be used by autoregressive/history models.",
        "- Future `OT` must never be used as a known future covariate.",
        "- The allowed known-future NWP covariates are `VAR78` to `VAR228` listed below.",
        "",
        "## Known Future NWP Columns",
        "",
        "| column | description |",
        "|---|---|",
    ]
    for col in NWP_COLUMNS:
        lines.append(f"| {col} | {NWP_DESCRIPTIONS[col]} |")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "| station | rows | start | end | NWP file | history file |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for row in manifest.itertuples(index=False):
        lines.append(f"| {row.station} | {row.rows} | {row.start} | {row.end} | {row.nwp_file} | {row.history_file} |")

    lines.extend(
        [
            "",
            "## Quality Audit",
            "",
            "| station | rows | expected rows | missing cells | duplicate time | non-1h diffs | OT min | OT max | OT zero % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in audit.itertuples(index=False):
        lines.append(
            f"| {row.zone} | {row.rows} | {row.expected_hourly_rows} | {row.missing_cells_nwp_file} | "
            f"{row.duplicate_time} | {row.non_1h_diffs} | {row.ot_min:.6f} | {row.ot_max:.6f} | {row.ot_zero_pct:.3f} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"out_dir={out_dir}")
    print(f"manifest={manifest_path}")
    print(f"known_future_columns={known_future_path}")
    print(f"audit={audit_path}")
    print(f"report={report_path}")
    print(manifest.to_string(index=False))
    print(audit.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GEFCom2014 Solar for Time-Series-Library experiments.")
    parser.add_argument("--raw_zip", type=Path, default=DEFAULT_RAW_ZIP)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--task", type=int, default=15, help="GEFCom2014 Solar task to use; Task 15 is the longest train set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with read_solar_zip(args.raw_zip) as solar_zip:
        train, predictors, instructions = load_task(solar_zip, args.task)
        df = normalize_task(train, predictors)
    write_outputs(df, args.task, args.raw_zip, args.out_dir, args.report_dir, instructions)


if __name__ == "__main__":
    main()
