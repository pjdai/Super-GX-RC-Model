"""
load_building.py
----------------
Given a building name (or "all"), loads VAV zone temp, AHU supply air temp,
and outdoor air temp timeseries into a clean, resampled DataFrame ready for
RC model analysis.

Usage (CLI)
-----------
    # List available buildings
    python load_building.py --list

    # Load ZAT + AHU SAT for one building, save to parquet
    python load_building.py --building "Zone C building"

    # Load specific field types only
    python load_building.py --building "Zone C building" \
        --fields vav_zat_sensor_id ahu_sat_sensor_id outdoor_air_temp

    # Load all buildings (slow)
    python load_building.py --building all

Usage (import)
--------------
    from load_building import load_building

    df_long, df_wide = load_building(
        building="Zone C building",
        catalog_path="catalog.csv",
        data_dir="data",
        fields=["vav_zat_sensor_id", "ahu_sat_sensor_id", "outdoor_air_temp"],
        resample_freq="5min",
    )

Output
------
    df_long  : long-format DataFrame
               columns: ts (UTC), val_mag, unit, field_type,
                        vav_name, ahu_sat_sensor_name, site_name, sensor_hex

    df_wide  : wide-format DataFrame, index=ts, one column per sensor
               column names: "{vav_name}__{field_type}"
               e.g. "VAV_UNIT_X__vav_zat_sensor_id", "BLDG_AHU1__ahu_sat_sensor_id"
               Resampled to resample_freq, forward-filled up to 30 min gaps.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Full-width colon used in filenames (U+FF1A)
FWCOLON = "\uf03a"

DEFAULT_FIELDS = [
    "vav_zat_sensor_id",
    "ahu_sat_sensor_id",
    "outdoor_air_temp",
]

# Forward-fill limit: don't gap-fill across more than this many 5-min periods
FFILL_LIMIT = 6   # = 30 minutes


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------
def load_building(
    building: str,
    catalog_path: str | Path = "catalog.csv",
    data_dir: str | Path = "data",
    fields: list[str] | None = None,
    resample_freq: str = "5min",
    unit_convert_f_to_c: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters
    ----------
    building        : building name substring to match (case-insensitive),
                      or "all" to load everything in catalog.
    catalog_path    : path to catalog.csv produced by data_catalog.py
    data_dir        : directory containing the raw timeseries CSVs
    fields          : list of field_type values to load;
                      defaults to ZAT + AHU SAT + OAT
    resample_freq   : pandas offset string for resampling; "5min" recommended
    unit_convert_f_to_c : if True, convert °F columns to °C in df_wide

    Returns
    -------
    (df_long, df_wide)
    """
    if fields is None:
        fields = DEFAULT_FIELDS

    catalog_path = Path(catalog_path)
    data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # 1. Filter catalog
    # ------------------------------------------------------------------
    cat = pd.read_csv(catalog_path)

    # Building filter
    if building.lower() != "all":
        mask_site = (
            cat["site_name"]
            .fillna("")
            .str.contains(building, case=False, regex=False)
        )
        # OAT files have no site_name — include them always when OAT requested
        mask_oat = (cat["field_type"] == "outdoor_air_temp") & ("outdoor_air_temp" in fields)
        cat_filtered = cat[mask_site | mask_oat].copy()
    else:
        cat_filtered = cat.copy()

    # Field type filter
    cat_filtered = cat_filtered[cat_filtered["field_type"].isin(fields)]

    # Skip unresolvable files
    cat_filtered = cat_filtered[
        ~cat_filtered["parse_status"].isin(["SKIP", "ERROR"])
    ]

    if cat_filtered.empty:
        print(f"No files found for building='{building}', fields={fields}")
        return pd.DataFrame(), pd.DataFrame()

    # Deduplicate: AHU SAT sensor files can appear once per VAV in catalog
    # (same physical file, different vav_name rows) — load each file once
    cat_unique = cat_filtered.drop_duplicates(subset=["filename"])

    print(
        f"Loading {len(cat_unique)} files for '{building}' "
        f"[{', '.join(fields)}] ..."
    )

    # ------------------------------------------------------------------
    # 2. Load each file and tag with metadata
    # ------------------------------------------------------------------
    chunks = []
    skipped = 0

    for _, row in cat_unique.iterrows():
        fpath = data_dir / row["filename"]
        if not fpath.exists():
            print(f"  MISSING: {row['filename']}")
            skipped += 1
            continue

        try:
            df = pd.read_csv(fpath, usecols=["ts", "val_mag", "unit"])
        except Exception as exc:
            print(f"  READ ERROR {row['filename']}: {exc}")
            skipped += 1
            continue

        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df.dropna(subset=["ts", "val_mag"], inplace=True)

        # Tag with catalog metadata
        df["field_type"]          = row["field_type"]
        df["sensor_hex"]          = row["sensor_hex"]
        df["vav_name"]            = row.get("vav_name", None)
        df["ahu_sat_sensor_name"] = row.get("ahu_sat_sensor_name", None)
        df["site_name"]           = row.get("site_name", None)

        chunks.append(df)

    if not chunks:
        print("No data loaded.")
        return pd.DataFrame(), pd.DataFrame()

    df_long = pd.concat(chunks, ignore_index=True)
    df_long.sort_values("ts", inplace=True)

    if skipped:
        print(f"  ({skipped} files skipped)")

    # ------------------------------------------------------------------
    # 3. Build column labels for wide format
    # ------------------------------------------------------------------
    # Priority: vav_name > ahu_sat_sensor_name > sensor_hex (fallback)
    def make_label(r: pd.Series) -> str:
        name = r["vav_name"]
        if pd.isna(name) or str(name).startswith("["):
            name = r["ahu_sat_sensor_name"]
        if pd.isna(name):
            name = r["sensor_hex"]
        # sanitise for column name
        name = str(name).replace(" ", "_").replace("/", "_").replace("&", "and")
        return f"{name}__{r['field_type']}"

    label_map = (
        cat_unique[["sensor_hex", "field_type", "vav_name", "ahu_sat_sensor_name"]]
        .drop_duplicates(subset=["sensor_hex", "field_type"])
        .assign(col_label=lambda d: d.apply(make_label, axis=1))
        .set_index(["sensor_hex", "field_type"])["col_label"]
    )

    df_long["col_label"] = df_long.apply(
        lambda r: label_map.get((r["sensor_hex"], r["field_type"]), r["sensor_hex"]),
        axis=1,
    )

    # ------------------------------------------------------------------
    # 4. Pivot to wide format + resample
    # ------------------------------------------------------------------
    df_wide = (
        df_long
        .groupby(["ts", "col_label"])["val_mag"]
        .mean()                      # average if duplicate timestamps per sensor
        .unstack("col_label")
    )

    # Resample to regular grid
    df_wide = df_wide.resample(resample_freq).mean()

    # Forward-fill short gaps only (BAS dropouts, not true missing data)
    df_wide = df_wide.ffill(limit=FFILL_LIMIT)

    # ------------------------------------------------------------------
    # 5. Optional unit conversion: °F → °C
    # ------------------------------------------------------------------
    if unit_convert_f_to_c:
        temp_cols = [
            c for c in df_wide.columns
            if any(k in c for k in ["zat", "sat", "dat", "oat", "outdoor"])
        ]
        df_wide[temp_cols] = (df_wide[temp_cols] - 32) * 5 / 9
        print(f"  Converted {len(temp_cols)} temp columns from °F to °C")

    n_sensors = df_wide.shape[1]
    n_rows = len(df_wide)
    pct_valid = df_wide.notna().mean().mean() * 100
    print(
        f"Wide DataFrame: {n_sensors} sensors × {n_rows} timesteps  "
        f"({pct_valid:.1f}% non-NaN)"
    )
    print(f"  Date range : {df_wide.index.min()}  →  {df_wide.index.max()}")
    print(f"  Columns    : {list(df_wide.columns)[:6]}{'...' if n_sensors > 6 else ''}")

    return df_long, df_wide


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_outputs(
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    building: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = building.lower().replace(" ", "_")

    long_path = output_dir / f"{slug}_long.parquet"
    wide_path = output_dir / f"{slug}_wide.parquet"

    df_long.to_parquet(long_path, index=False)
    df_wide.to_parquet(wide_path)

    print(f"\nSaved:")
    print(f"  long format : {long_path}  ({len(df_long):,} rows)")
    print(f"  wide format : {wide_path}  ({df_wide.shape[0]:,} rows × {df_wide.shape[1]} cols)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def list_buildings(catalog_path: str | Path = "catalog.csv") -> None:
    cat = pd.read_csv(catalog_path)
    sites = cat["site_name"].dropna().unique()
    print(f"\nBuildings in catalog ({len(sites)}):")
    for s in sorted(sites):
        count = (cat["site_name"] == s).sum()
        print(f"  {str(s):<45} {count:>5} files")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load BAS timeseries for a building into a DataFrame."
    )
    p.add_argument(
        "--building",
        type=str,
        default="Zone C building",
        help="Building name substring (case-insensitive), or 'all'",
    )
    p.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help="field_type values to load (space-separated)",
    )
    p.add_argument("--catalog",    type=Path, default=Path("catalog.csv"))
    p.add_argument("--data_dir",   type=Path, default=Path("data"))
    p.add_argument("--output_dir", type=Path, default=Path("output"))
    p.add_argument("--resample",   type=str,  default="5min")
    p.add_argument(
        "--celsius",
        action="store_true",
        help="Convert temperature columns from °F to °C",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available buildings and exit",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.catalog.exists():
        print(f"ERROR: catalog not found: {args.catalog}", file=sys.stderr)
        print("Run data_catalog.py first.", file=sys.stderr)
        sys.exit(1)

    if args.list:
        list_buildings(args.catalog)
        return

    df_long, df_wide = load_building(
        building=args.building,
        catalog_path=args.catalog,
        data_dir=args.data_dir,
        fields=args.fields,
        resample_freq=args.resample,
        unit_convert_f_to_c=args.celsius,
    )

    if df_long.empty:
        sys.exit(1)

    save_outputs(df_long, df_wide, args.building, args.output_dir)


if __name__ == "__main__":
    main()