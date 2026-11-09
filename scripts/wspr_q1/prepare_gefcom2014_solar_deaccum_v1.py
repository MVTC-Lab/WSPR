#!/usr/bin/env python3
"""Prepare a versioned, cycle-aware GEFCom2014-S Task 15 dataset.

The four ECMWF accumulated fields are converted to one-hour interval
accumulations within each documented midnight initialization cycle.  This
script never writes to the legacy processed-data directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT.parent / "datasets" / "PV"
DEFAULT_RAW_ZIP = DATA_ROOT / "raws_datasets" / "GEFCom2014" / "GEFCom2014.zip"
DEFAULT_OLD_DIR = DATA_ROOT / "processed_research" / "all_stations" / "GEFCom2014Solar"
DEFAULT_OUT_DIR = DATA_ROOT / "processed_research" / "corrected" / "GEFCom2014Solar_deaccum_v1"
DEFAULT_AUDIT_DIR = (
    PROJECT_ROOT / "experiments" / "wspr_q1" / "gefcom_deaccum_corrective_v1" / "01_data_audit"
)

NWP_COLUMNS = [
    "VAR78", "VAR79", "VAR134", "VAR157", "VAR164", "VAR165",
    "VAR166", "VAR167", "VAR169", "VAR175", "VAR178", "VAR228",
]
ACCUMULATED_COLUMNS = ["VAR169", "VAR175", "VAR178", "VAR228"]
NON_ACCUMULATED_COLUMNS = [c for c in NWP_COLUMNS if c not in ACCUMULATED_COLUMNS]
DESCRIPTIONS = {
    "VAR78": "ECMWF total column liquid water (tclw)",
    "VAR79": "ECMWF total column ice water (tciw)",
    "VAR134": "ECMWF surface pressure (SP)",
    "VAR157": "ECMWF relative humidity at 1000 mbar",
    "VAR164": "ECMWF total cloud cover (TCC)",
    "VAR165": "ECMWF 10 metre U wind component",
    "VAR166": "ECMWF 10 metre V wind component",
    "VAR167": "ECMWF 2 metre temperature",
    "VAR169": "ECMWF one-hour surface solar radiation downwards accumulation (SSRD)",
    "VAR175": "ECMWF one-hour surface thermal radiation downwards accumulation (STRD)",
    "VAR178": "ECMWF one-hour top net solar radiation accumulation (TSR)",
    "VAR228": "ECMWF one-hour total precipitation accumulation (TP)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_task15(raw_zip: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    with zipfile.ZipFile(raw_zip) as outer:
        nested = [n for n in outer.namelist() if n.endswith("GEFCom2014-S_V2.zip")]
        if len(nested) != 1:
            raise RuntimeError(f"expected one GEFCom2014-S_V2.zip, found {nested}")
        payload = outer.read(nested[0])
    with zipfile.ZipFile(io.BytesIO(payload)) as solar:
        instructions = solar.read("Solar/Instructions.txt").decode("utf-8", errors="replace")
        train = pd.read_csv(solar.open("Solar/Task 15/train15.csv"))
        predictors = pd.read_csv(solar.open("Solar/Task 15/predictors15.csv"))
    return train, predictors, instructions


def normalize_task15(train: pd.DataFrame, predictors: pd.DataFrame) -> pd.DataFrame:
    required_train = {"ZONEID", "TIMESTAMP", "POWER"}
    required_predictors = {"ZONEID", "TIMESTAMP", *NWP_COLUMNS}
    if missing := required_train.difference(train.columns):
        raise ValueError(f"train15.csv missing columns: {sorted(missing)}")
    if missing := required_predictors.difference(predictors.columns):
        raise ValueError(f"predictors15.csv missing columns: {sorted(missing)}")

    left = train[["ZONEID", "TIMESTAMP", "POWER"]].copy()
    right = predictors[["ZONEID", "TIMESTAMP", *NWP_COLUMNS]].copy()
    merged = left.merge(right, on=["ZONEID", "TIMESTAMP"], how="inner", validate="one_to_one")
    merged["date"] = pd.to_datetime(merged["TIMESTAMP"], format="%Y%m%d %H:%M", errors="raise")
    if not merged["date"].dt.minute.eq(0).all():
        raise ValueError("non-hourly timestamp minute found")
    for col in [*NWP_COLUMNS, "POWER"]:
        merged[col] = pd.to_numeric(merged[col], errors="raise")
    merged = merged.rename(columns={"POWER": "OT"})
    merged["OT"] = merged["OT"].clip(lower=0)
    return merged[["ZONEID", "date", *NWP_COLUMNS, "OT"]].sort_values(["ZONEID", "date"]).reset_index(drop=True)


def add_cycle_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    midnight = out["date"].dt.hour.eq(0)
    out["forecast_initialization"] = out["date"].dt.normalize() - pd.to_timedelta(midnight.astype(int), unit="D")
    out["lead_hour"] = out["date"].dt.hour.where(~midnight, 24).astype(int)
    return out


def validate_cycles(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (zone, initialization), group in frame.groupby(["ZONEID", "forecast_initialization"], sort=True):
        leads = group["lead_hour"].tolist()
        expected = list(range(1, 25))
        if leads != expected:
            raise ValueError(f"invalid 24-hour cycle zone={zone} init={initialization}: {leads}")
        valid = group["date"].tolist()
        expected_valid = [initialization + pd.Timedelta(hours=h) for h in expected]
        if valid != expected_valid:
            raise ValueError(f"valid-time mismatch zone={zone} init={initialization}")
        rows.append(
            {
                "zone": f"zone{int(zone):02d}",
                "forecast_initialization": initialization,
                "first_valid_time": valid[0],
                "last_valid_time": valid[-1],
                "rows": len(group),
                "first_lead_hour": leads[0],
                "last_lead_hour": leads[-1],
            }
        )
    return pd.DataFrame(rows)


def deaccumulate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    audit_rows: list[dict[str, object]] = []
    group_keys = [out["ZONEID"], out["forecast_initialization"]]
    for col in ACCUMULATED_COLUMNS:
        raw = out[col].astype(float)
        increments = raw.groupby(group_keys, sort=False).diff()
        first = out["lead_hour"].eq(1)
        increments.loc[first] = raw.loc[first]  # step 0 is zero by the locked convention
        if increments.isna().any():
            raise ValueError(f"unexpected missing increment in {col}")
        preclip = increments.copy()
        increments = increments.clip(lower=0.0)
        out[col] = increments
        for zone, idx in out.groupby("ZONEID", sort=True).groups.items():
            before = preclip.loc[idx]
            after = increments.loc[idx]
            audit_rows.append(
                {
                    "zone": f"zone{int(zone):02d}",
                    "field": col,
                    "rows": int(len(idx)),
                    "timestamp_start": out.loc[idx, "date"].min(),
                    "timestamp_end": out.loc[idx, "date"].max(),
                    "missing_count": int(after.isna().sum()),
                    "nan_count": int(np.isnan(after.to_numpy()).sum()),
                    "inf_count": int(np.isinf(after.to_numpy()).sum()),
                    "min": float(after.min()),
                    "max": float(after.max()),
                    "mean": float(after.mean()),
                    "p01": float(after.quantile(0.01)),
                    "p50": float(after.quantile(0.50)),
                    "p99": float(after.quantile(0.99)),
                    "identified_cycles": int(out.loc[idx, "forecast_initialization"].nunique()),
                    "identified_reset_boundaries": int(out.loc[idx, "forecast_initialization"].nunique() - 1),
                    "negative_increments_before_nonnegative_clip": int(before.lt(0).sum()),
                    "minimum_increment_before_clip": float(before.min()),
                    "negative_increments_after_handling": int(after.lt(0).sum()),
                }
            )
    return out, pd.DataFrame(audit_rows)


def compare_with_legacy(corrected: pd.DataFrame, old_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    checks: list[dict[str, object]] = []
    representative: list[pd.DataFrame] = []
    for zone in sorted(corrected["ZONEID"].unique()):
        zone_name = f"zone{int(zone):02d}"
        old_path = old_dir / f"GEFCom2014Solar_{zone_name}_nwp.csv"
        old = pd.read_csv(old_path)
        old["date"] = pd.to_datetime(old["date"], errors="raise")
        new = corrected[corrected["ZONEID"].eq(zone)].drop(columns=["ZONEID"]).copy()
        old = old[["date", *NWP_COLUMNS, "OT"]]
        new = new[["date", *NWP_COLUMNS, "OT"]]
        checks.extend(
            [
                {"zone": zone_name, "check": "row_count", "passed": len(old) == len(new), "detail": f"old={len(old)} new={len(new)}"},
                {"zone": zone_name, "check": "timestamps_unchanged", "passed": np.array_equal(old["date"].to_numpy(), new["date"].to_numpy()), "detail": "exact datetime value equality"},
                {"zone": zone_name, "check": "OT_unchanged", "passed": np.array_equal(old["OT"].to_numpy(), new["OT"].to_numpy()), "detail": "exact numeric equality"},
                {"zone": zone_name, "check": "nonaccumulated_predictors_unchanged", "passed": np.array_equal(old[NON_ACCUMULATED_COLUMNS].to_numpy(), new[NON_ACCUMULATED_COLUMNS].to_numpy()), "detail": "exact numeric equality"},
                {"zone": zone_name, "check": "intended_accumulated_fields_changed", "passed": all(not np.array_equal(old[c].to_numpy(), new[c].to_numpy()) for c in ACCUMULATED_COLUMNS), "detail": ",".join(ACCUMULATED_COLUMNS)},
            ]
        )
        sample = pd.DataFrame({"zone": zone_name, "date": new["date"].iloc[:72]})
        for col in ACCUMULATED_COLUMNS:
            sample[f"old_{col}"] = old[col].iloc[:72].to_numpy()
            sample[f"corrected_{col}"] = new[col].iloc[:72].to_numpy()
        representative.append(sample)
    return pd.DataFrame(checks), pd.concat(representative, ignore_index=True)


def write_comparison_plot(sample: pd.DataFrame, audit_dir: Path) -> None:
    zone = "zone01"
    part = sample[sample["zone"].eq(zone)].copy()
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
    for row, col in enumerate(ACCUMULATED_COLUMNS):
        axes[row, 0].plot(part["date"], part[f"old_{col}"], color="#777777", linewidth=1.2)
        axes[row, 0].set_ylabel(col)
        axes[row, 0].set_title(f"Legacy cumulative {col}")
        axes[row, 1].plot(part["date"], part[f"corrected_{col}"], color="#0072B2", linewidth=1.2)
        axes[row, 1].set_title(f"Corrected one-hour {col}")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("GEFCom2014-S Task 15 cycle-aware deaccumulation (zone01, first 72 h)")
    fig.tight_layout()
    fig.savefig(audit_dir / "legacy_vs_corrected_representative.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    columns = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.9g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_outputs(
    corrected: pd.DataFrame,
    cycle_audit: pd.DataFrame,
    field_audit: pd.DataFrame,
    comparison_checks: pd.DataFrame,
    comparison_sample: pd.DataFrame,
    raw_zip: Path,
    out_dir: Path,
    audit_dir: Path,
    instructions: str,
) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty corrected dataset directory: {out_dir}")
    if audit_dir.exists() and any(audit_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty data-audit directory: {audit_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for zone in sorted(corrected["ZONEID"].unique()):
        zone_name = f"zone{int(zone):02d}"
        zdf = corrected[corrected["ZONEID"].eq(zone)].drop(
            columns=["ZONEID", "forecast_initialization", "lead_hour"]
        ).copy()
        zdf["date"] = zdf["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        nwp_name = f"GEFCom2014Solar_{zone_name}_nwp.csv"
        history_name = f"GEFCom2014Solar_{zone_name}_history.csv"
        zdf[["date", *NWP_COLUMNS, "OT"]].to_csv(out_dir / nwp_name, index=False, encoding="utf-8-sig")
        zdf[["date", "OT"]].to_csv(out_dir / history_name, index=False, encoding="utf-8-sig")
        manifest_rows.append(
            {
                "dataset": "GEFCom2014Solar_deaccum_v1",
                "station": zone_name,
                "task": 15,
                "nwp_file": nwp_name,
                "history_file": history_name,
                "rows": len(zdf),
                "start": zdf["date"].iloc[0],
                "end": zdf["date"].iloc[-1],
                "freq": "1h",
                "target": "OT",
                "future_covariates": ",".join(NWP_COLUMNS),
                "source_zip": str(raw_zip),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest_gefcom2014solar_deaccum_v1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "column": col,
                "description": DESCRIPTIONS[col],
                "future_known": True,
                "transformation": "cycle-aware one-hour increment; negative numerical artifacts clipped to zero" if col in ACCUMULATED_COLUMNS else "unchanged",
            }
            for col in NWP_COLUMNS
        ]
        + [{"column": "OT", "description": "PV power target", "future_known": False, "transformation": "unchanged"}]
    ).to_csv(out_dir / "known_future_columns_gefcom2014solar_deaccum_v1.csv", index=False, encoding="utf-8-sig")

    cycle_audit.to_csv(audit_dir / "forecast_cycle_audit.csv", index=False, encoding="utf-8-sig")
    field_audit.to_csv(audit_dir / "accumulated_field_statistics.csv", index=False, encoding="utf-8-sig")
    comparison_checks.to_csv(audit_dir / "legacy_corrected_invariance_checks.csv", index=False, encoding="utf-8-sig")
    comparison_sample.to_csv(audit_dir / "legacy_vs_corrected_representative.csv", index=False, encoding="utf-8-sig")
    (audit_dir / "gefcom2014solar_instructions.txt").write_text(instructions, encoding="utf-8")
    write_comparison_plot(comparison_sample, audit_dir)

    if not comparison_checks["passed"].all():
        failed = comparison_checks.loc[~comparison_checks["passed"]]
        raise AssertionError(f"legacy/corrected invariance QA failed:\n{failed.to_string(index=False)}")
    if field_audit["negative_increments_after_handling"].sum() != 0:
        raise AssertionError("negative increments remain after locked handling")
    if field_audit[["missing_count", "nan_count", "inf_count"]].to_numpy().sum() != 0:
        raise AssertionError("missing/NaN/Inf values found in corrected fields")

    report_lines = [
        "# GEFCom2014 Solar corrected-data audit",
        "",
        "Status: **PASS**",
        "",
        "## Locked transformation",
        "",
        "- Source: GEFCom2014-S Task 15.",
        "- Each zone contains complete midnight-initialized 24-hour cycles with valid times at lead hours 1–24.",
        "- For each accumulated field, lead 1 equals the archived accumulated value because forecast step 0 is zero; leads 2–24 are adjacent differences within the same cycle.",
        "- Negative one-hour increments in physically nonnegative fields are clipped to zero before any model training. Counts before and after this handling are reported below.",
        "- Units remain interval accumulations in the archive units (J m^-2 for radiation and m for precipitation); no performance-dependent choice was made.",
        "",
        "## Structural checks",
        "",
        f"- Cycles: {len(cycle_audit)} total; all have 24 rows and leads 1–24.",
        f"- Corrected rows: {len(corrected)} total across {corrected['ZONEID'].nunique()} zones.",
        f"- Legacy invariance checks passed: {int(comparison_checks['passed'].sum())}/{len(comparison_checks)}.",
        "- OT, timestamps, and all eight non-accumulated predictors are exactly unchanged.",
        "- Only VAR169, VAR175, VAR178, and VAR228 are transformed.",
        "- No future target value is introduced; OT remains the target and is excluded from known-future covariates.",
        "",
        "## Accumulated-field summary",
        "",
        dataframe_to_markdown(field_audit),
    ]
    (audit_dir / "gefcom_corrected_data_audit.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    critical = sorted([p for p in out_dir.rglob("*") if p.is_file()] + [p for p in audit_dir.rglob("*") if p.is_file()])
    hash_path = audit_dir / "SHA256SUMS.csv"
    with hash_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "size_bytes", "sha256"])
        for path in critical:
            writer.writerow([str(path), path.stat().st_size, sha256(path)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_zip", type=Path, default=DEFAULT_RAW_ZIP)
    parser.add_argument("--old_dir", type=Path, default=DEFAULT_OLD_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--audit_dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train, predictors, instructions = read_task15(args.raw_zip)
    normalized = add_cycle_columns(normalize_task15(train, predictors))
    cycle_audit = validate_cycles(normalized)
    corrected, field_audit = deaccumulate(normalized)
    second, _ = deaccumulate(normalized)
    pd.testing.assert_frame_equal(corrected, second, check_exact=True)
    checks, representative = compare_with_legacy(corrected, args.old_dir)
    write_outputs(
        corrected, cycle_audit, field_audit, checks, representative,
        args.raw_zip, args.out_dir, args.audit_dir, instructions,
    )
    payload = {
        "status": "PASS",
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "rows": len(corrected),
        "cycles": len(cycle_audit),
        "zones": int(corrected["ZONEID"].nunique()),
        "output_dir": str(args.out_dir),
        "audit_dir": str(args.audit_dir),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
