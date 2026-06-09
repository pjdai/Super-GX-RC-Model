"""Merge a building zone's BAS data with open-meteo weather data on hourly timestamps.

Usage:
    python scripts/merge_weather.py ZONE_A
    python scripts/merge_weather.py ZONE_B
"""
from pathlib import Path
import argparse
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
WEATHER = DATA_DIR / "weather_hourly.xlsx"

# Per-zone BAS source basename (xlsx preferred, csv fallback).
ZONES = {
    "ZONE_A": "BLDG.ZONE_A_hourly",
    "ZONE_B": "BLDG.ZONE_B_hourly",
}

WEATHER_COLS = [
    "temperature_2m",
    "shortwave_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
]
INTERP_LIMIT = 3


def load_bas(zone: str) -> pd.DataFrame:
    base = DATA_DIR / ZONES[zone]
    xlsx, csv = base.with_suffix(".xlsx"), base.with_suffix(".csv")
    if xlsx.exists():
        df, src = pd.read_excel(xlsx), xlsx.name
    elif csv.exists():
        df, src = pd.read_csv(csv), csv.name
    else:
        raise FileNotFoundError(f"BAS file not found: {xlsx} or {csv}")
    print(f"Loaded BAS:     {src}  rows={len(df)}  cols={len(df.columns)}")
    if "datetime" not in df.columns:
        raise KeyError(f"'datetime' not in BAS columns: {list(df.columns)[:10]}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"])
    return df


def load_weather() -> pd.DataFrame:
    df = pd.read_excel(WEATHER)
    print(f"Loaded weather: {WEATHER.name}  rows={len(df)}  cols={len(df.columns)}")
    if "time" not in df.columns:
        raise KeyError(f"'time' not in weather columns: {list(df.columns)}")

    # Weather columns may include units, e.g. "temperature_2m (°C)". Match by prefix.
    rename_map: dict[str, str] = {}
    for want in WEATHER_COLS:
        matches = [c for c in df.columns if c == want or c.startswith(want + " ") or c.startswith(want + "(")]
        if not matches:
            raise KeyError(f"Weather file missing column starting with '{want}'. Have: {list(df.columns)}")
        rename_map[matches[0]] = want
    df = df.rename(columns=rename_map)

    df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"])
    return df[["datetime", *WEATHER_COLS]]


def count_oversized_gaps(series: pd.Series, limit: int) -> int:
    is_na = series.isna().to_numpy()
    n_gaps = 0
    run = 0
    for v in is_na:
        if v:
            run += 1
        else:
            if run > limit:
                n_gaps += 1
            run = 0
    if run > limit:
        n_gaps += 1
    return n_gaps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zone", choices=sorted(ZONES), help="Zone to merge")
    args = ap.parse_args()

    bas = load_bas(args.zone)
    wx = load_weather()

    merged = bas.merge(wx, on="datetime", how="outer").sort_values("datetime").reset_index(drop=True)
    print(f"\nMerged (pre-interp): rows={len(merged)}")

    numeric_cols = merged.select_dtypes(include="number").columns
    gap_counts = {c: count_oversized_gaps(merged[c], INTERP_LIMIT) for c in numeric_cols}

    merged[numeric_cols] = merged[numeric_cols].interpolate(
        method="linear", limit=INTERP_LIMIT, limit_direction="both"
    )

    post_nan = merged.isna().sum()

    print("\n=== SUMMARY ===")
    print(f"Total rows        : {len(merged)}")
    print(f"Date range        : {merged['datetime'].min()}  ->  {merged['datetime'].max()}")
    print(f"Interpolation     : linear, limit={INTERP_LIMIT} consecutive hours")
    print(f"\nRemaining NaN per column (after interpolation):")
    for col, n in post_nan.items():
        if n:
            print(f"  {col:50s} {int(n):>8d}")
    no_nans = int((post_nan == 0).sum())
    print(f"  ({no_nans} columns have zero remaining NaN)")

    print(f"\nGaps too large to fill (>{INTERP_LIMIT} consecutive NaN hours), per column:")
    total_big = 0
    for col, n in gap_counts.items():
        if n:
            print(f"  {col:50s} {n:>8d}")
            total_big += n
    print(f"  TOTAL oversized gaps across numeric columns: {total_big}")

    output = DATA_DIR / f"merged_{args.zone}_weather.xlsx"
    merged.to_excel(output, index=False)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
