from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


FREQ = "15min"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-aware PV forecasting CSVs for Time-Series-Library."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "raws_datasets",
        help="Directory containing raw zip files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "processed_research",
        help="Output directory for processed CSV files and audit summaries.",
    )
    parser.add_argument(
        "--guoneng-target-policy",
        choices=["clip_zero", "raw"],
        default="clip_zero",
        help="How to handle negative Guoneng actual power.",
    )
    return parser.parse_args()


def find_zip(raw_dir: Path, required_suffixes: list[str]) -> Path:
    for path in sorted(raw_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
        except zipfile.BadZipFile:
            continue
        if all(any(name.endswith(suffix) for name in names) for suffix in required_suffixes):
            return path
    raise FileNotFoundError(f"No zip in {raw_dir} contains {required_suffixes}")


def wind_to_sincos(degrees: pd.Series) -> tuple[pd.Series, pd.Series]:
    radians = np.deg2rad(pd.to_numeric(degrees, errors="coerce"))
    return pd.Series(np.sin(radians), index=degrees.index), pd.Series(np.cos(radians), index=degrees.index)


def summarize_frame(df: pd.DataFrame, dataset: str, station: str, note: str) -> dict:
    time = pd.to_datetime(df["date"], errors="coerce")
    target = pd.to_numeric(df["OT"], errors="coerce")
    expected = pd.date_range(time.min(), time.max(), freq=FREQ) if len(time) else []
    return {
        "dataset": dataset,
        "station": station,
        "rows": int(len(df)),
        "start": str(time.min()),
        "end": str(time.max()),
        "expected_rows_at_15min": int(len(expected)),
        "missing_time_points_after_processing": int(len(expected) - time.nunique()) if len(time) else 0,
        "duplicate_times_after_processing": int(time.duplicated().sum()),
        "missing_cells_after_processing": int(df.isna().sum().sum()),
        "target_min": float(target.min()) if len(target) else math.nan,
        "target_max": float(target.max()) if len(target) else math.nan,
        "target_mean": float(target.mean()) if len(target) else math.nan,
        "target_zero_ratio": float((target == 0).mean()) if len(target) else math.nan,
        "target_negative_ratio": float((target < 0).mean()) if len(target) else math.nan,
        "note": note,
    }


def clean_duplicate_times(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    numeric_cols = [col for col in df.columns if col != "date"]
    df = df.groupby("date", as_index=False)[numeric_cols].mean()
    return df.sort_values("date").reset_index(drop=True), before - len(df)


def longest_contiguous_segment(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    time = pd.to_datetime(df["date"], errors="coerce")
    gaps = time.diff().ne(pd.Timedelta(minutes=15))
    segment_id = gaps.cumsum()
    sizes = segment_id.value_counts().sort_values(ascending=False)
    keep_id = sizes.index[0]
    segment = df.loc[segment_id == keep_id].copy().reset_index(drop=True)
    return segment, {
        "segment_count": int(segment_id.nunique()),
        "kept_segment_rows": int(len(segment)),
        "dropped_rows_outside_kept_segment": int(len(df) - len(segment)),
        "kept_segment_start": str(pd.to_datetime(segment["date"]).min()),
        "kept_segment_end": str(pd.to_datetime(segment["date"]).max()),
    }


def build_pvod(raw_dir: Path, out_dir: Path) -> list[dict]:
    zip_path = find_zip(raw_dir, ["metadata.csv", "station00.csv", "station09.csv"])
    output_dir = out_dir / "all_stations" / "PVOD"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        station_names = sorted(name for name in zf.namelist() if name.startswith("station") and name.endswith(".csv"))
        for name in station_names:
            raw = pd.read_csv(zf.open(name))
            station = Path(name).stem
            out = pd.DataFrame()
            # PVOD timestamps are in UTC; convert to Beijing time for solar-cycle features.
            out["date"] = pd.to_datetime(raw["date_time"], errors="coerce") + pd.Timedelta(hours=8)
            out["nwp_globalirrad"] = pd.to_numeric(raw["nwp_globalirrad"], errors="coerce")
            out["nwp_directirrad"] = pd.to_numeric(raw["nwp_directirrad"], errors="coerce")
            out["nwp_temperature"] = pd.to_numeric(raw["nwp_temperature"], errors="coerce")
            out["nwp_humidity"] = pd.to_numeric(raw["nwp_humidity"], errors="coerce")
            out["nwp_windspeed"] = pd.to_numeric(raw["nwp_windspeed"], errors="coerce")
            sin_col, cos_col = wind_to_sincos(raw["nwp_winddirection"])
            out["nwp_winddir_sin"] = sin_col
            out["nwp_winddir_cos"] = cos_col
            out["nwp_pressure"] = pd.to_numeric(raw["nwp_pressure"], errors="coerce")
            out["OT"] = pd.to_numeric(raw["power"], errors="coerce").clip(lower=0)
            out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            out, duplicate_count = clean_duplicate_times(out)
            out, segment_info = longest_contiguous_segment(out)
            out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

            file_name = f"PVOD_{station}_nwp.csv"
            out.to_csv(output_dir / file_name, index=False)
            summaries.append(
                summarize_frame(
                    out,
                    dataset="PVOD",
                    station=station,
                    note=(
                        "NWP-only; lmd observed weather excluded; PVOD timestamps converted UTC+8; "
                        f"duplicate timestamps averaged={duplicate_count}; "
                        f"kept longest contiguous 15min segment; {segment_info}."
                    ),
                )
            )
    return summaries


def build_guoneng(raw_dir: Path, out_dir: Path, target_policy: str) -> list[dict]:
    zip_path = find_zip(raw_dir, ["train_1.csv", "train_2.csv", "train_3.csv", "train_4.csv"])
    output_dir = out_dir / "all_stations" / "GuonengRixin"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        train_names = sorted(name for name in zf.namelist() if name.endswith(".csv") and "/train_" in name)
        for name in train_names:
            raw = pd.read_csv(zf.open(name), encoding="gb18030")
            station = f"site{Path(name).stem.split('_')[-1]}"

            # Column order in the released competition data:
            # time, forecast irradiance, wind speed, wind direction, temperature,
            # pressure, humidity, observed irradiance, actual power.
            out = pd.DataFrame()
            out["date"] = pd.to_datetime(raw.iloc[:, 0], errors="coerce", infer_datetime_format=True)
            out["forecast_irrad"] = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
            out["forecast_windspeed"] = pd.to_numeric(raw.iloc[:, 2], errors="coerce")
            sin_col, cos_col = wind_to_sincos(raw.iloc[:, 3])
            out["forecast_winddir_sin"] = sin_col
            out["forecast_winddir_cos"] = cos_col
            out["forecast_temperature"] = pd.to_numeric(raw.iloc[:, 4], errors="coerce")
            out["forecast_pressure"] = pd.to_numeric(raw.iloc[:, 5], errors="coerce")
            out["forecast_humidity"] = pd.to_numeric(raw.iloc[:, 6], errors="coerce")
            power_raw = pd.to_numeric(raw.iloc[:, 8], errors="coerce")
            out["OT"] = power_raw.clip(lower=0) if target_policy == "clip_zero" else power_raw

            out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            out, duplicate_count = clean_duplicate_times(out)
            out, segment_info = longest_contiguous_segment(out)
            out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

            file_name = f"Guoneng_{station}_forecast_longest.csv"
            out.to_csv(output_dir / file_name, index=False)
            note = (
                "Forecast weather only; observed irradiance excluded; "
                f"target_policy={target_policy}; duplicate timestamps averaged={duplicate_count}; "
                f"kept longest contiguous 15min segment; {segment_info}."
            )
            summaries.append(summarize_frame(out, dataset="GuonengRixin", station=station, note=note))
    return summaries


def copy_pilot_files(out_dir: Path) -> None:
    pilot_dir = out_dir / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        (out_dir / "all_stations" / "PVOD" / "PVOD_station01_nwp.csv", pilot_dir / "PVOD_station01_nwp.csv"),
        (
            out_dir / "all_stations" / "GuonengRixin" / "Guoneng_site2_forecast_longest.csv",
            pilot_dir / "Guoneng_site2_forecast_longest.csv",
        ),
    ]
    for src, dst in pairs:
        if src.exists():
            pd.read_csv(src).to_csv(dst, index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    summaries.extend(build_pvod(args.raw_dir, args.out_dir))
    summaries.extend(build_guoneng(args.raw_dir, args.out_dir, args.guoneng_target_policy))
    copy_pilot_files(args.out_dir)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.out_dir / "dataset_audit_summary.csv", index=False)
    with (args.out_dir / "dataset_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"Wrote processed datasets to: {args.out_dir}")
    print(summary_df[["dataset", "station", "rows", "start", "end", "target_zero_ratio", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
