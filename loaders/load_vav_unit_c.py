"""Load BAS parquet/catalog data for VAV_UNIT_C (Zone C building) into the RC pipeline format.

Drop-in replacement for bas_loader.load_bas_data(); the returned DataFrame
matches the canonical pipeline column format (see rc_quiet_learner.py) and can be
passed downstream without changes.

Output columns (identical to bas_loader.py):
    Datetime              : index, tz-aware America/Los_Angeles, hourly
    Room Temp (F)         : VAV_UNIT_C zone air temperature
    Outdoor Temp (F)      : outdoor air temperature
    VAV Discharge Air Temp (F)            : VAV_UNIT_C discharge air temp
    AHU Discharge Air Temp (F)            : AHU04 supply air temp
    VAV Discharge Air Volume (ft^3 / min) : VAV_UNIT_C airflow CFM
    GHI (W/m²)            : solar (Open-Meteo, cached)
    DNI (W/m²)
    DHI (W/m²)

Usage:
    from loaders.load_vav_unit_c import load_vav_unit_c
    df = load_vav_unit_c()
"""

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOCAL_TZ = "America/Los_Angeles"
# Site coordinates for the Open-Meteo solar pull. Set these to your site.
SITE_LAT = 0.0
SITE_LON = 0.0

SOLAR_COLS = ["GHI (W/m²)", "DNI (W/m²)", "DHI (W/m²)"]
SOLAR_CACHE_FILENAME = "solar_cache_vav_unit_c.csv"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Fullwidth colon used in BAS filenames (U+FF1A, replaces ":" on Windows)
FWCOLON = "\uf03a"

# Output column names must match _resolve_cols aliases in rc_quiet_learner.py
OUT_ROOM_TEMP    = "Room Temp (F)"
OUT_OAT          = "Outdoor Temp (F)"
OUT_VAV_DAT      = "VAV Discharge Air Temp (F)"
OUT_AHU_SAT      = "AHU Discharge Air Temp (F)"
OUT_FLOW         = "VAV Discharge Air Volume (ft^3 / min)"


# ---------------------------------------------------------------------------
# Solar helpers (copied from bas_loader.py for self-containment)
# ---------------------------------------------------------------------------
def _fetch_solar(lat, lon, start_date, end_date):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": "shortwave_radiation,direct_radiation,diffuse_radiation",
        "timezone": LOCAL_TZ,
    }
    url = f"{OPEN_METEO_ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    hourly = payload["hourly"]
    idx = pd.to_datetime(hourly["time"])
    return pd.DataFrame({
        "GHI (W/m²)": hourly["shortwave_radiation"],
        "DNI (W/m²)": hourly["direct_radiation"],
        "DHI (W/m²)": hourly["diffuse_radiation"],
    }, index=idx)


def _load_or_fetch_solar(cache_dir, start_date, end_date, verbose):
    cache_path = Path(cache_dir) / SOLAR_CACHE_FILENAME
    if cache_path.exists():
        if verbose:
            print(f"  [cache] solar from {cache_path.name}")
        solar = pd.read_csv(cache_path, parse_dates=["Datetime"]).set_index("Datetime")
    else:
        if verbose:
            print(f"  [fetch] open-meteo {start_date} -> {end_date}")
        solar = _fetch_solar(SITE_LAT, SITE_LON, start_date, end_date)
        solar.index.name = "Datetime"
        solar.reset_index().to_csv(cache_path, index=False)
        if verbose:
            print(f"  [cache] wrote {len(solar)} rows to {cache_path.name}")
    return solar


# ---------------------------------------------------------------------------
# Find VAV_UNIT_C sensor files from catalog
# ---------------------------------------------------------------------------
def _find_vav_files(catalog_path: Path, data_dir: Path, vav_name: str,
                    verbose: bool) -> dict:
    """
    Look up vav_name in catalog.csv and return file paths for each
    field_type that exists on disk.

    Returns dict: {field_type: Path}
    """
    cat = pd.read_csv(catalog_path)
    vav_all = cat[cat["vav_name"] == vav_name]

    # Primary: OK / OK_SHARED rows.
    ok_rows = vav_all[vav_all["parse_status"].isin(["OK", "OK_SHARED"])]
    ok_field_types = set(ok_rows["field_type"])

    # Fallback: for any field_type with zero OK rows, accept WARN rows.
    warn_rows = vav_all[
        (vav_all["parse_status"] == "WARN") &
        (~vav_all["field_type"].isin(ok_field_types))
    ]

    # For vav_dat_sensor_id, if multiple WARN files exist, keep the one with
    # the highest n_rows from the catalog.
    dat_warn = warn_rows[warn_rows["field_type"] == "vav_dat_sensor_id"]
    if len(dat_warn) > 1:
        keep_idx = dat_warn["n_rows"].idxmax()
        warn_rows = warn_rows.drop(dat_warn.index.difference([keep_idx]))

    vav_rows = pd.concat([ok_rows, warn_rows])

    if vav_rows.empty:
        raise ValueError(
            f"VAV '{vav_name}' not found in catalog with OK or WARN status. "
            f"Run data_catalog.py first."
        )

    found = {}
    for _, row in vav_rows.iterrows():
        fpath = data_dir / row["filename"]
        if fpath.exists():
            found[row["field_type"]] = fpath
            if verbose:
                tag = "ok" if row["parse_status"] in ("OK", "OK_SHARED") else "warn"
                print(f"  [{tag}] {row['field_type']:35s} <- {row['filename'].replace(chr(0xf03a), ':')}")
        else:
            if verbose:
                print(f"  [missing] {row['filename'].replace(chr(0xf03a), ':')}")

    return found


# ---------------------------------------------------------------------------
# Single-file loader: raw CSV -> hourly Series
# ---------------------------------------------------------------------------
def _load_csv_hourly(fpath: Path, col_name: str,
                     unit: str = "F") -> pd.Series:
    """Load a BAS timeseries CSV, resample hourly (unit: "C" converts °C->°F, "F"/"cfm" keep as-is)."""
    df = pd.read_csv(fpath, usecols=["ts", "val_mag"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").set_index("ts")

    s = df["val_mag"]

    # Mask physically impossible values before conversion
    if unit == "C":
        s = s.where(s.between(0, 60))          # valid zone/supply temp range
        s = s * 9 / 5 + 32                     # °C -> °F
    elif unit == "F":
        s = s.where(s.between(40, 95))         # valid °F discharge air range
    elif unit == "cfm":
        s = s.where(s >= 0)                    # flow can't be negative

    hourly = s.resample("1h").mean()
    hourly.name = col_name
    return hourly


# ---------------------------------------------------------------------------
# Parquet loader for pre-processed RC input
# ---------------------------------------------------------------------------
def _load_parquet_hourly(parquet_path: Path, verbose: bool) -> pd.DataFrame:
    """
    Load vav_unit_c_rc_input.parquet (ZAT, AHU SAT, OAT in °C, 5-min UTC),
    convert temperatures to °F, resample to hourly.
    """
    df = pd.read_parquet(parquet_path)
    if verbose:
        print(f"  [parquet] {parquet_path.name}  "
              f"{len(df)} rows  {df.index.min()} -> {df.index.max()}")

    # Convert °C -> °F for temperature columns
    temp_cols = ["zat_VAV_UNIT_C", "sat_AHU04", "oat"]
    for col in temp_cols:
        df[col] = df[col] * 9 / 5 + 32

    # Resample to hourly
    hourly = df.resample("1h").mean()
    return hourly


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------
def load_vav_unit_c(
    parquet_path: str | Path = "output/vav_unit_c_rc_input.parquet",
    catalog_path: str | Path = "catalog.csv",
    data_dir: str | Path = "data",
    output_dir: str | Path = "output",
    verbose: bool = True,
) -> pd.DataFrame:
    """Load VAV_UNIT_C data into the canonical pipeline column format (hourly, local tz).

    ZAT, AHU SAT, and OAT come from the pre-processed parquet; DAT and flow are
    loaded fresh from the raw BAS CSVs via catalog lookup; solar is fetched from
    Open-Meteo (cached).
    """
    parquet_path = Path(parquet_path)
    catalog_path = Path(catalog_path)
    data_dir     = Path(data_dir)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Loading VAV_UNIT_C data for the RC pipeline...\n")

    # ------------------------------------------------------------------
    # 1. Load ZAT, AHU SAT, OAT from parquet (already cleaned)
    # ------------------------------------------------------------------
    hourly_base = _load_parquet_hourly(parquet_path, verbose)

    series_list = [
        hourly_base["zat_VAV_UNIT_C"].rename(OUT_ROOM_TEMP),
        hourly_base["oat"].rename(OUT_OAT),
        hourly_base["sat_AHU04"].rename(OUT_AHU_SAT),
    ]

    # ------------------------------------------------------------------
    # 2. Load DAT and flow from raw CSV via catalog
    # ------------------------------------------------------------------
    if verbose:
        print("\nLooking up VAV_UNIT_C sensor files in catalog...")

    vav_files = _find_vav_files(catalog_path, data_dir, "VAV_UNIT_C", verbose)

    # DAT
    if "vav_dat_sensor_id" in vav_files:
        dat_series = _load_csv_hourly(
            vav_files["vav_dat_sensor_id"], OUT_VAV_DAT, unit="F"
        )
        series_list.append(dat_series)
        if verbose:
            print(f"  [ok] {OUT_VAV_DAT:45s}  {dat_series.notna().sum()} non-NaN rows")
    else:
        if verbose:
            print(f"  [warn] vav_dat_sensor_id not found for VAV_UNIT_C — "
                  f"column will be NaN")
        series_list.append(pd.Series(name=OUT_VAV_DAT, dtype=float))

    # Flow
    if "vav_flow_sensor_id" in vav_files:
        flow_series = _load_csv_hourly(
            vav_files["vav_flow_sensor_id"], OUT_FLOW, unit="cfm"
        )
        series_list.append(flow_series)
        if verbose:
            print(f"  [ok] {OUT_FLOW:45s}  {flow_series.notna().sum()} non-NaN rows")
    else:
        if verbose:
            print(f"  [warn] vav_flow_sensor_id not found for VAV_UNIT_C — "
                  f"column will be NaN")
        series_list.append(pd.Series(name=OUT_FLOW, dtype=float))

    # ------------------------------------------------------------------
    # 3. Merge all series on hourly UTC index
    # ------------------------------------------------------------------
    df = pd.concat(series_list, axis=1)

    # Convert to local timezone. A missing sensor adds an empty placeholder
    # series whose non-datetime index can downgrade df.index to a plain Index
    # after concat, so coerce back to a DatetimeIndex first.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(LOCAL_TZ)
    else:
        df.index = df.index.tz_convert(LOCAL_TZ)
    df.index.name = "Datetime"

    # Drop rows with NaN in core columns (ZAT, OAT, AHU SAT)
    core_cols = [OUT_ROOM_TEMP, OUT_OAT, OUT_AHU_SAT]
    n_before = len(df)
    df = df.dropna(subset=core_cols)
    if verbose:
        print(f"\n  Merged: {n_before} rows -> {len(df)} complete rows "
              f"({n_before - len(df)} dropped for NaN in core columns)")
        print(f"  Date range: {df.index.min()}  ->  {df.index.max()}")

    # ------------------------------------------------------------------
    # 4. Solar irradiance (Open-Meteo, cached)
    # ------------------------------------------------------------------
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date   = df.index.max().strftime("%Y-%m-%d")

    solar = _load_or_fetch_solar(output_dir, start_date, end_date, verbose)
    naive_idx = df.index.tz_localize(None)
    solar_aligned = solar.reindex(naive_idx)
    solar_aligned.index = df.index
    df = pd.concat([df, solar_aligned], axis=1)

    for c in SOLAR_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n  Final columns: {list(df.columns)}")
        print(f"\n  Column stats:")
        print(df.describe().round(2).to_string())

    return df


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_vav_unit_c(verbose=True)
    print(f"\nFirst 5 rows:\n{df.head()}")

    out_path = Path("output") / "vav_unit_c_pipeline_preview.csv"
    df.head(200).to_csv(out_path)
    print(f"\nPreview saved -> {out_path}")