r"""Load the 7 BAS sensor CSVs, join on timestamp, resample hourly into the RC pipeline format.

Expected output columns (matching _resolve_cols aliases in rc_quiet_learner.py):
    Datetime              : timezone-aware, America/Los_Angeles
    Room Temp (F)         : zone air temperature
    Outdoor Temp (F)      : outdoor air temperature
    VAV Discharge Air Temp (F)   : VAV discharge air temp (T_vav)
    AHU Discharge Air Temp (F)   : AHU supply air temp (T_ahu)
    VAV Discharge Air Volume (ft^3 / min) : airflow CFM

Usage:
    from loaders.bas_loader import load_bas_data
    df = load_bas_data(data_dir="data/zone_c/bldg_data")
"""

import os
import glob
import json
import urllib.parse
import urllib.request
import pandas as pd

# ---------------------------------------------------------------------------
# Column name mapping: output name → substring to match in filename
# ---------------------------------------------------------------------------
FILE_SIGNAL_MAP = {
    "Room Temp (F)":                       "vav_zat",
    "Outdoor Temp (F)":                    "outdoor_air_temperature",
    "VAV Discharge Air Temp (F)":          "vav_dat",
    "AHU Discharge Air Temp (F)":          "ahu_sat",
    "VAV Discharge Air Volume (ft^3 / min)": "vav_flow",
}

LOCAL_TZ = "America/Los_Angeles"

# Site coordinates for the Open-Meteo solar pull. Set these to your site.
SITE_LAT = 0.0
SITE_LON = 0.0

SOLAR_COLS = ["GHI (W/m²)", "DNI (W/m²)", "DHI (W/m²)"]
SOLAR_CACHE_FILENAME = "solar_cache.csv"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_solar_openmeteo(lat: float, lon: float,
                          start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly GHI / DNI / DHI from the Open-Meteo historical archive.

    Returns a DataFrame indexed by naive local hourly timestamps with columns
    'GHI (W/m²)', 'DNI (W/m²)', 'DHI (W/m²)'.
    """
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":   "shortwave_radiation,direct_radiation,diffuse_radiation",
        "timezone": LOCAL_TZ,
    }
    url = f"{OPEN_METEO_ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    hourly = payload["hourly"]
    idx = pd.to_datetime(hourly["time"])  # naive local timestamps
    df = pd.DataFrame({
        "GHI (W/m²)": hourly["shortwave_radiation"],
        "DNI (W/m²)": hourly["direct_radiation"],
        "DHI (W/m²)": hourly["diffuse_radiation"],
    }, index=idx)
    df.index.name = "Datetime"
    return df


def _load_or_fetch_solar(data_dir: str, start_date: str, end_date: str,
                         verbose: bool) -> pd.DataFrame:
    """Return a solar DataFrame, fetching once and caching to CSV in data_dir."""
    cache_path = os.path.join(data_dir, SOLAR_CACHE_FILENAME)
    if os.path.exists(cache_path):
        if verbose:
            print(f"  [cache] reading solar from {SOLAR_CACHE_FILENAME}")
        solar = pd.read_csv(cache_path, parse_dates=["Datetime"]).set_index("Datetime")
    else:
        if verbose:
            print(f"  [fetch] open-meteo archive {start_date} -> {end_date}")
        solar = fetch_solar_openmeteo(SITE_LAT, SITE_LON, start_date, end_date)
        solar.reset_index().to_csv(cache_path, index=False)
        if verbose:
            print(f"  [cache] wrote {len(solar)} rows to {SOLAR_CACHE_FILENAME}")
    return solar


def _find_csv(data_dir: str, keyword: str) -> str:
    """Find a CSV file in data_dir whose name contains keyword."""
    pattern = os.path.join(data_dir, f"*{keyword}*.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No CSV found matching '*{keyword}*.csv' in {data_dir}"
        )
    if len(matches) > 1:
        print(f"  [warn] Multiple files match '{keyword}', using: {os.path.basename(matches[0])}")
    return matches[0]


def _load_single(filepath: str, col_name: str) -> pd.Series:
    """
    Load one sensor CSV, parse timestamps, resample to hourly mean.
    Returns a named Series indexed by hourly UTC timestamps.
    """
    df = pd.read_csv(filepath, usecols=["ts", "val_mag"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").set_index("ts")
    hourly = df["val_mag"].resample("1h").mean()
    hourly.name = col_name
    return hourly


def load_bas_data(data_dir: str, verbose: bool = True) -> pd.DataFrame:
    """Load and merge all BAS CSVs into one hourly DataFrame in the canonical pipeline format.

    Index is 'Datetime' (tz-aware, America/Los_Angeles); rows with any NaN are dropped.
    """
    series_list = []

    for col_name, keyword in FILE_SIGNAL_MAP.items():
        try:
            fpath = _find_csv(data_dir, keyword)
            s = _load_single(fpath, col_name)
            series_list.append(s)
            if verbose:
                print(f"  [ok] {col_name:45s}  {len(s)} hourly rows  "
                      f"({s.isna().sum()} NaN)  ← {os.path.basename(fpath)}")
        except FileNotFoundError as e:
            print(f"  [missing] {e}")

    if not series_list:
        raise RuntimeError("No signal files found. Check data_dir path.")

    # Merge on shared hourly UTC index
    df = pd.concat(series_list, axis=1)

    # Convert index to local timezone and name it 'Datetime'
    df.index = df.index.tz_convert(LOCAL_TZ)
    df.index.name = "Datetime"

    # Drop rows where any signal is NaN
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)

    if verbose:
        print(f"\n  Merged: {n_before} rows → {len(df)} complete rows "
              f"({n_dropped} dropped for NaN)")
        print(f"  Date range: {df.index.min()}  →  {df.index.max()}")

    # ----- Solar irradiance merge (open-meteo, cached) -----
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date   = df.index.max().strftime("%Y-%m-%d")
    solar = _load_or_fetch_solar(data_dir, start_date, end_date, verbose=verbose)

    # solar's index is naive local. Strip tz on df's index so the join aligns
    # by wall-clock, then restore the tz-aware index afterwards.
    naive_index = df.index.tz_localize(None)
    solar_aligned = solar.reindex(naive_index)
    solar_aligned.index = df.index
    df = pd.concat([df, solar_aligned], axis=1)

    # Nighttime / out-of-range gaps: zero is a fine fill for irradiance.
    for c in SOLAR_COLS:
        df[c] = df[c].fillna(0.0)

    if verbose:
        print(f"\n  Column stats:")
        print(df.describe().round(2).to_string())

    return df


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "zone_c", "bldg_data")
    print(f"Loading BAS data from: {data_dir}\n")
    df = load_bas_data(data_dir, verbose=True)
    print(f"\nFirst 5 rows:\n{df.head()}")

    # Save a preview Excel for inspection
    out_path = os.path.join(data_dir, "ZONE_C_hourly_merged_preview.xlsx")
    df.head(200).to_excel(out_path)
    print(f"\nPreview saved → {out_path}")
