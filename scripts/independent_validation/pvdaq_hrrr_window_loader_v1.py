"""Origin-indexed, numpy-only PVDAQ/HRRR loader; no model or training imports.

Preserves frozen dates and forecast vintages. The primary target is explicitly
zero-clipped hourly measured AC power. Historical meteorology is NOT provided.
Seven constant standardized-space padding slots preserve existing model shapes;
they are not weather observations or imputed forecasts.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FIELDS = ("TMP", "RH", "UGRD", "VGRD", "PRES", "TCDC", "DSWRF")
SITES = ("2107", "9068", "14200")
HORIZONS = (4, 24)
HISTORY = 96
PATHWAYS = ("history", "direct", "residual")
SPLITS = {
    "train": ("2020-01-01", "2021-10-01"),
    "validation": ("2021-10-01", "2022-01-01"),
    "test": ("2022-01-01", "2023-01-01"),
}
HOUR = pd.Timedelta(hours=1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def target_view(values, mode):
    raw = np.asarray(values, dtype=np.float64)
    require(np.isfinite(raw).all(), "Nonfinite power in retained window")
    if mode == "zero_clipped_ac":
        return np.maximum(raw, 0.0)
    if mode == "raw_ac":
        return raw.copy()
    raise ValueError("Unknown target convention: " + str(mode))


def fit_standardizer(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and len(values) > 0 and np.isfinite(values).all(), "Invalid fitting rows")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    constant = scale == 0.0
    scale[constant] = 1.0
    return dict(mean=mean.tolist(), scale=scale.tolist(), constant_columns=np.flatnonzero(constant).tolist(),
                rows=len(values), ddof=0)


def transform(values, scaler):
    return (np.asarray(values, dtype=np.float64) - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])


def inverse_transform(values, scaler):
    return np.asarray(values, dtype=np.float64) * np.asarray(scaler["scale"]) + np.asarray(scaler["mean"])


def fit_site_scalers(power, joined, mode):
    """Fit power once per complete TRAIN hour and NWP once per TRAIN vintage/lead.

    Not per rolling window: repeated hours/horizon views never reweight fitting.
    Hourly power timestamps are interval ends, hence the train test is (lo, hi].
    """
    lo, hi = [pd.Timestamp(x) for x in SPLITS["train"]]
    use = (power.index > lo) & (power.index <= hi) & np.isfinite(power.to_numpy())
    pv = power.loc[use]
    require(len(pv) > 0, "Empty training power set")
    train = joined.loc[joined["split"].eq("train")].copy()
    keys = ["init_time_utc", "lead_hours"]
    # Duplicate H=4/H=24 views must agree; never hide inconsistent values.
    unique_values = train[keys + list(FIELDS)].drop_duplicates()
    require(not unique_values.duplicated(keys).any(), "Conflicting train NWP duplicate")
    nwp = unique_values.sort_values(keys)
    valid = nwp.init_time_utc + pd.to_timedelta(nwp.lead_hours, unit="h")
    require((nwp.init_time_utc >= lo).all() and (valid <= hi).all(), "Non-train forecast reached scaler")
    result = dict(target_mode=mode, power=fit_standardizer(target_view(pv.values, mode)[:, None]),
                  nwp=fit_standardizer(nwp[list(FIELDS)].to_numpy()),
                  power_first_end=str(pv.index.min()), power_last_end=str(pv.index.max()),
                  nwp_first_init=str(nwp.init_time_utc.min()), nwp_last_valid=str(valid.max()),
                  fitting_partition="train_only", nwp_deduplication=keys)
    return result


def time_features_hourly(times):
    """Identical hourly UTC calendar convention to frozen utils/timefeatures.py."""
    idx = pd.DatetimeIndex(times)
    return np.column_stack([idx.hour / 23.0 - 0.5, idx.dayofweek / 6.0 - 0.5,
                            (idx.day - 1) / 30.0 - 0.5, (idx.dayofyear - 1) / 365.0 - 0.5]).astype(np.float32)


def validate_window_times(origin, horizon, split, rows):
    require(horizon in HORIZONS and split in SPLITS, "Unknown window task")
    origin = pd.Timestamp(origin)
    lo, hi = [pd.Timestamp(x) for x in SPLITS[split]]
    require(origin - HISTORY * HOUR >= lo and origin + horizon * HOUR <= hi,
            "Window crosses a frozen split boundary")
    require(origin.hour in (0, 6, 12, 18) and origin.minute == origin.second == 0,
            "Invalid initialization cycle")
    require(len(rows) == horizon and np.array_equal(rows.lead_hours.to_numpy(), np.arange(1, horizon + 1)),
            "Missing, duplicated or unordered forecast leads")
    require(rows.origin_utc.eq(origin).all() and rows.init_time_utc.eq(origin).all(), "Forecast vintage mismatch")
    valid = pd.date_range(origin + HOUR, periods=horizon, freq="h")
    require(np.array_equal(rows.valid_time_utc.to_numpy(), valid.to_numpy()), "Forecast valid-time mismatch")
    require(np.array_equal(rows.pv_interval_end_utc.to_numpy(), valid.to_numpy()) and
            np.array_equal(rows.pv_interval_start_utc.to_numpy(), (valid - HOUR).to_numpy()), "PV interval mismatch")
    return pd.date_range(origin - (HISTORY - 1) * HOUR, periods=HISTORY, freq="h"), valid


def pack_model_inputs(history_scaled, future_nwp_scaled, history_times, future_times, pathway):
    """No future power parameter exists in this API; labels stay outside inputs."""
    require(pathway in PATHWAYS, "Unknown pathway")
    power = np.asarray(history_scaled, dtype=np.float32).reshape(HISTORY, 1)
    nwp = np.asarray(future_nwp_scaled, dtype=np.float32)
    require(nwp.ndim == 2 and nwp.shape[1] == len(FIELDS), "Wrong NWP channel shape")
    require(len(history_times) == HISTORY and len(future_times) == len(nwp), "Calendar length mismatch")
    x = np.zeros((HISTORY, len(FIELDS) + 1), dtype=np.float32)
    x[:, -1:] = power
    decoder = np.zeros((len(nwp), len(FIELDS) + 1), dtype=np.float32)
    if pathway != "history":
        decoder[:, :-1] = nwp
    require(np.isfinite(x).all() and np.isfinite(decoder).all(), "Nonfinite model input")
    return dict(x_enc=x, x_mark_enc=time_features_hourly(history_times), x_dec=decoder,
                x_mark_dec=time_features_hourly(future_times))


class WindowStore:
    """Retained origins are the only index; generic percentage splits are forbidden."""
    def __init__(self, source_root, qc_dir, target_mode):
        self.root, self.qc = Path(source_root), Path(qc_dir)
        require(target_mode in ("zero_clipped_ac", "raw_ac"), "Explicit target mode required")
        self.target_mode = target_mode
        summary = read_json(self.qc / "summary.json")
        require(summary["status"] == "PASS" and summary["excluded_origins"] == 0, "Full QC gate not passed")
        self.input_hashes = {}
        for name, expected in summary["artifacts_sha256"].items():
            actual = sha(self.qc / name)
            require(actual == expected, "QC artifact changed: " + name)
            self.input_hashes[str(self.qc / name)] = actual
        self.input_hashes[str(self.qc / "summary.json")] = sha(self.qc / "summary.json")
        frozen = read_json(self.root / "FROZEN_CHRONOLOGICAL_SPLIT.json")
        plan_path = self.root / "hrrr_point_extraction_v2/plan.json"
        require(sha(plan_path) == summary["plan_sha256"] ==
                "bb46abe6c6eef9d04e7c8dfdd7e53379318cffca4a26c020ac9394209e67bff9", "Frozen plan changed")
        self.input_hashes[str(plan_path)] = sha(plan_path)
        plan = read_json(plan_path)
        for name in ("FROZEN_CHRONOLOGICAL_SPLIT.json", "FROZEN_CHRONOLOGICAL_SPLIT_origins.csv"):
            actual = sha(self.root / name)
            require(actual == plan["inputs_sha256"][name], "Frozen source changed: " + name)
            self.input_hashes[str(self.root / name)] = actual
        self.power = {}
        for site in SITES:
            path = self.root / ("pv_%s_2020_2022_hourly_diagnostic.csv" % site)
            actual = sha(path)
            require(actual == frozen["power_hourly_sha256"][site], "PV checksum mismatch")
            self.input_hashes[str(path)] = actual
            data = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            require(data.index.is_unique and data.index.is_monotonic_increasing, "Invalid power time index")
            self.power[site] = data
        self.origins = pd.read_csv(self.root / "FROZEN_CHRONOLOGICAL_SPLIT_origins.csv", dtype={"site": str})
        self.origins["origin_utc"] = pd.to_datetime(self.origins.origin_utc, errors="raise")
        self.joined = pd.read_csv(self.qc / "audited_joined_forecast_rows.csv", dtype={"site": str})
        for name in ("origin_utc", "init_time_utc", "valid_time_utc", "pv_interval_start_utc", "pv_interval_end_utc"):
            self.joined[name] = pd.to_datetime(self.joined[name], errors="raise")
        keys = ["site", "horizon_hours", "split", "origin_utc"]
        require(not self.origins.duplicated(keys).any(), "Duplicate origin")
        require(not self.joined.duplicated(keys + ["lead_hours"]).any(), "Duplicate joined lead")
        self.groups = self.joined.groupby(keys, sort=False).indices
        expected = set(self.origins[keys].itertuples(index=False, name=None))
        require(set(self.groups) == expected and len(expected) == 17635, "Joined/origin set differs")
        # Preserve different vintages at the same valid time; only repeated
        # horizon views of the SAME site/init/lead may be deduplicated.
        vintage_keys = ["site", "init_time_utc", "lead_hours"]
        distinct = self.joined[vintage_keys + ["valid_time_utc", "power_kw_raw"] + list(FIELDS)].drop_duplicates()
        require(not distinct.duplicated(vintage_keys).any(), "Inconsistent cross-horizon duplicate")
        self.scalers = {site: fit_site_scalers(self.power[site], self.joined.loc[self.joined.site.eq(site)], target_mode)
                        for site in SITES}

    def __len__(self):
        return len(self.origins)

    def window(self, index):
        r = self.origins.iloc[index]
        site, horizon, split, origin = str(r.site), int(r.horizon_hours), str(r.split), r.origin_utc
        rows = self.joined.iloc[self.groups[(site, horizon, split, origin)]].sort_values("lead_hours")
        history_times, future_times = validate_window_times(origin, horizon, split, rows)
        history_raw = self.power[site].reindex(history_times).to_numpy(dtype=np.float64)
        target_raw = self.power[site].reindex(future_times).to_numpy(dtype=np.float64)
        require(np.allclose(target_raw, rows.power_kw_raw.to_numpy(), rtol=0, atol=1e-10), "Joined target mismatch")
        scaler = self.scalers[site]
        history_scaled = transform(target_view(history_raw, self.target_mode)[:, None], scaler["power"])
        target_scaled = transform(target_view(target_raw, self.target_mode)[:, None], scaler["power"])
        nwp_scaled = transform(rows[list(FIELDS)].to_numpy(), scaler["nwp"])
        require(np.isfinite(nwp_scaled).all(), "Invalid forecast values")
        return dict(site=site, horizon=horizon, split=split, origin=str(origin),
                    history_times=history_times, future_times=future_times,
                    history_raw=history_raw, target_raw=target_raw,
                    history_scaled=history_scaled, target_scaled=target_scaled, nwp_scaled=nwp_scaled)

    def verify_sources_unchanged(self):
        for path, checksum in self.input_hashes.items():
            require(sha(path) == checksum, "Source changed during check: " + path)
