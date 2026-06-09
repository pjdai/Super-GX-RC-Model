"""
data_catalog.py
---------------
Scans a directory of BAS timeseries CSVs, parses each filename into
structured metadata, joins against the VAV master sheet, and outputs a
catalog CSV plus a console summary.

Usage
-----
    python data_catalog.py \
        --data_dir   ./data                    \
        --master     bldg_vav_info.csv         \
        --output     catalog.csv

Filename conventions handled
-----------------------------
  VAV sensor file:
    p_zone_c_r_{record_hex}__{field_type}__{sensor_hex}.csv
    e.g. p_zone_c_r_1e3a7bdd-8a34804f__vav_zat_sensor_id__1e3a7b14-abc.csv

  OAT (Outdoor Air Temp) file:
    oat_p_zone_c_r_{site_hex}.csv
    e.g. oat_p_zone_c_r_2c3afbe8-da6a1097.csv

Output columns
--------------
    filename, file_type, parse_status, parse_note,
    record_hex, field_type, sensor_hex,
    vav_id, vav_name, site_name, vav_zat_location_name,
    ahu_sat_sensor_name,
    clg_min_flow, clg_max_flow, htg_min_flow, htg_max_flow,
    ts_min, ts_max, duration_days, n_rows, unit
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SITE_PREFIX = "p:zone_c:r:"

# VAV sensor file: p_zone_c_r_{record_hex}__{field_type}__{sensor_hex}.csv
VAV_RE = re.compile(
    r"^p\uf03azone_c\uf03ar\uf03a(?P<record_hex>[0-9a-f]+-[0-9a-f]+)"
    r"__(?P<field_type>[a-z_]+)"
    r"__(?P<sensor_hex>[0-9a-f]+-[0-9a-f]+)\.csv$"
)

# OAT file: oat_p_zone_c_r_{site_hex}.csv
OAT_RE = re.compile(
    r"^oat_p\uf03azone_c\uf03ar\uf03a(?P<site_hex>[0-9a-f]+-[0-9a-f]+)\.csv$"
)

SENSOR_ID_COLS = [
    "vav_flow_sensor_id",
    "vav_dat_sensor_id",
    "vav_vpos_sensor_id",
    "vav_dpos_sensor_id",
    "vav_zat_sensor_id",
    "ahu_sat_sensor_id",
]

MASTER_KEEP = [
    "vav_id",
    "vav_name",
    "site_name",
    "vav_zat_location_name",
    "ahu_sat_sensor_name",
    "clg_min_flow",
    "clg_max_flow",
    "htg_min_flow",
    "htg_max_flow",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_ts_meta(path: Path) -> dict:
    """Read ts and unit columns only to get date range efficiently."""
    try:
        df = pd.read_csv(path, usecols=["ts", "unit"])
        if df.empty:
            return dict(ts_min=None, ts_max=None, duration_days=None,
                        unit=None, n_rows=0)
        ts = pd.to_datetime(df["ts"], utc=True)
        ts_min = ts.min()
        ts_max = ts.max()
        unit = df["unit"].dropna().iloc[0] if not df["unit"].dropna().empty else None
        return dict(
            ts_min=ts_min.isoformat(),
            ts_max=ts_max.isoformat(),
            duration_days=int((ts_max - ts_min).days),
            unit=unit,
            n_rows=len(df),
        )
    except Exception as exc:
        return dict(ts_min=None, ts_max=None, duration_days=None,
                    unit=None, n_rows=-1, _read_error=str(exc))


# ---------------------------------------------------------------------------
# Per-file parsers
# ---------------------------------------------------------------------------
def parse_vav_file(path: Path, m: re.Match, master_by_vav: pd.DataFrame) -> dict:
    row = dict(
        file_type="vav_sensor",
        record_hex=m.group("record_hex"),
        field_type=m.group("field_type"),
        sensor_hex=m.group("sensor_hex"),
    )

    vav_full_id = f"{SITE_PREFIX}{row['record_hex']}"
    sensor_full_id = f"{SITE_PREFIX}{row['sensor_hex']}"

    if vav_full_id in master_by_vav.index:
        vav_row = master_by_vav.loc[vav_full_id]
        if isinstance(vav_row, pd.DataFrame):
            vav_row = vav_row.iloc[0]
        for col in MASTER_KEEP:
            row[col] = vav_row.get(col, None)

        if row["field_type"] in SENSOR_ID_COLS:
            expected = vav_row.get(row["field_type"], None)
            if pd.isna(expected) or expected != sensor_full_id:
                row["parse_status"] = "WARN"
                row["parse_note"] = (
                    f"sensor_id mismatch: file has {sensor_full_id}, "
                    f"master has {expected}"
                )
            else:
                row["parse_status"] = "OK"
                row["parse_note"] = ""
        else:
            row["parse_status"] = "WARN"
            row["parse_note"] = f"field_type '{row['field_type']}' not a known sensor_id column"

    else:
        # record_hex not a direct vav_id — try reverse lookup by sensor
        if row["field_type"] in SENSOR_ID_COLS:
            matches = master_by_vav[master_by_vav[row["field_type"]] == sensor_full_id]
            if not matches.empty:
                first = matches.iloc[0]
                row["vav_id"] = None
                row["vav_name"] = f"[{len(matches)} VAVs share sensor]"
                row["site_name"] = first.get("site_name", None)
                row["ahu_sat_sensor_name"] = first.get("ahu_sat_sensor_name", None)
                row["parse_status"] = "OK_SHARED"
                row["parse_note"] = (
                    f"record_hex not a VAV; sensor shared by "
                    f"{len(matches)} VAVs"
                )
            else:
                row["parse_status"] = "WARN"
                row["parse_note"] = "record_hex and sensor_hex not found in master"
        else:
            row["parse_status"] = "WARN"
            row["parse_note"] = "record_hex not found in master vav_id"

    return row


def parse_oat_file(path: Path, m: re.Match, master: pd.DataFrame) -> dict:
    site_hex = m.group("site_hex")
    site_full_id = f"{SITE_PREFIX}{site_hex}"

    # Try to find a matching site in master (may not always match)
    site_match = master[master["site_id"] == site_full_id]
    if not site_match.empty:
        site_name = site_match.iloc[0]["site_name"]
        parse_status = "OK"
        parse_note = ""
    else:
        site_name = None
        parse_status = "OK_NO_SITE"
        parse_note = "site_hex not found in master site_id; standalone OAT sensor"

    return dict(
        file_type="oat",
        record_hex=site_hex,
        field_type="outdoor_air_temp",
        sensor_hex=site_hex,
        vav_id=None,
        vav_name=None,
        site_name=site_name,
        vav_zat_location_name=None,
        ahu_sat_sensor_name=None,
        clg_min_flow=None,
        clg_max_flow=None,
        htg_min_flow=None,
        htg_max_flow=None,
        parse_status=parse_status,
        parse_note=parse_note,
    )


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def build_catalog(data_dir: Path, master_path: Path) -> pd.DataFrame:
    print(f"Loading master sheet: {master_path}")
    master = pd.read_csv(master_path)

    for col in ["clg_min_flow", "clg_max_flow", "htg_min_flow", "htg_max_flow"]:
        master[col] = pd.to_numeric(
            master[col].astype(str).str.replace(" cfm", "", regex=False),
            errors="coerce",
        )

    master_by_vav = master.set_index("vav_id")

    # Glob ALL csvs, then classify by regex
    csv_files = sorted(data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {data_dir}\n")

    records = []
    for i, path in enumerate(csv_files, 1):
        row = {"filename": path.name}

        vav_m = VAV_RE.match(path.name)
        oat_m = OAT_RE.match(path.name)

        if vav_m:
            row.update(parse_vav_file(path, vav_m, master_by_vav))
        elif oat_m:
            row.update(parse_oat_file(path, oat_m, master))
        else:
            row["file_type"] = "unknown"
            row["parse_status"] = "SKIP"
            row["parse_note"] = "filename did not match VAV or OAT pattern"
            records.append(row)
            continue

        ts_meta = read_ts_meta(path)
        if "_read_error" in ts_meta:
            row["parse_status"] = "ERROR"
            row["parse_note"] = str(row.get("parse_note", "")) + " | " + ts_meta["_read_error"]
        row.update({k: v for k, v in ts_meta.items() if not k.startswith("_")})

        records.append(row)

        if i % 100 == 0:
            print(f"  processed {i}/{len(csv_files)} files...")

    catalog = pd.DataFrame(records)

    col_order = [
        "filename", "file_type", "parse_status", "parse_note",
        "record_hex", "field_type", "sensor_hex",
        "vav_id", "vav_name", "site_name", "vav_zat_location_name",
        "ahu_sat_sensor_name",
        "clg_min_flow", "clg_max_flow", "htg_min_flow", "htg_max_flow",
        "ts_min", "ts_max", "duration_days", "n_rows", "unit",
    ]
    existing = [c for c in col_order if c in catalog.columns]
    extra = [c for c in catalog.columns if c not in col_order]
    return catalog[existing + extra]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(catalog: pd.DataFrame) -> None:
    if catalog.empty or "parse_status" not in catalog.columns:
        print("\nNo files were processed. Check --data_dir path.\n")
        return

    print("\n" + "=" * 60)
    print("CATALOG SUMMARY")
    print("=" * 60)

    print(f"\nTotal files scanned : {len(catalog)}")
    for status, count in catalog["parse_status"].value_counts().items():
        print(f"  {status:<20} : {count}")

    ok = catalog[catalog["parse_status"].isin(["OK", "OK_SHARED", "OK_NO_SITE"])]

    if "file_type" in ok.columns:
        print(f"\nFile types ({ok['file_type'].nunique()} unique):")
        for ft, cnt in ok["file_type"].value_counts().items():
            print(f"  {ft:<30} : {cnt}")

    if "field_type" in ok.columns:
        print(f"\nField types ({ok['field_type'].nunique()} unique):")
        for ft, cnt in ok["field_type"].value_counts().items():
            print(f"  {ft:<35} : {cnt}")

    if "site_name" in ok.columns:
        print(f"\nBuildings ({ok['site_name'].nunique()} unique):")
        for site, cnt in ok["site_name"].value_counts().items():
            print(f"  {str(site):<45} : {cnt}")

    if ok["ts_min"].notna().any():
        ts_min_all = pd.to_datetime(ok["ts_min"], utc=True)
        ts_max_all = pd.to_datetime(ok["ts_max"], utc=True)
        print(f"\nDate range across all files:")
        print(f"  Earliest start : {ts_min_all.min()}")
        print(f"  Latest end     : {ts_max_all.max()}")

    dur = ok["duration_days"].dropna()
    rows = ok["n_rows"].dropna()
    if not dur.empty:
        print(f"\nFile duration (days): mean {dur.mean():.1f} | min {dur.min():.0f} | max {dur.max():.0f}")
    if not rows.empty:
        print(f"Rows per file      : mean {rows.mean():,.0f} | total {rows.sum():,.0f}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Build BAS timeseries catalog.")
    p.add_argument("--data_dir", type=Path, default=Path("./data"))
    p.add_argument("--master",   type=Path, default=Path("bldg_vav_info.csv"))
    p.add_argument("--output",   type=Path, default=Path("catalog.csv"))
    args = p.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: data_dir not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.master.exists():
        print(f"ERROR: master sheet not found: {args.master}", file=sys.stderr)
        sys.exit(1)

    catalog = build_catalog(args.data_dir, args.master)
    catalog.to_csv(args.output, index=False)
    print(f"Catalog saved to: {args.output}  ({len(catalog)} rows)")
    print_summary(catalog)


if __name__ == "__main__":
    main()