#!/usr/bin/env python3
"""Sweep training-data length and season composition for one zone.

Runs run_rolling_forecast.main() once per config (forecasting June 2025) and
prints a single consolidated summary table at the end.
"""
import os
import sys
import traceback
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import run_rolling_forecast as program_rolling

ROOM = "BLDG.ZONE_A"
FORECAST_END = "2025-06-30 23:00"
FORECAST_DAYS = 30

# (tag, train_period_label, length_label, train_start, train_end)
CONFIGS = [
    ("A1_1mo",    "2025-05 only",   "1 mo",  "2025-05-01", "2025-05-31 23:00"),
    ("A2_3mo",    "2025-03 to 05",  "3 mo",  "2025-03-01", "2025-05-31 23:00"),
    ("A3_6mo",    "2024-12 to 25-05","6 mo", "2024-12-01", "2025-05-31 23:00"),
    ("A4_12mo",   "2024-06 to 25-05","12 mo","2024-06-01", "2025-05-31 23:00"),
    ("B1_spring", "2025-03 to 05",  "3 mo",  "2025-03-01", "2025-05-31 23:00"),
    ("B2_winter", "2024-12 to 25-02","3 mo", "2024-12-01", "2025-02-28 23:00"),
    ("B3_summer", "2024-06 to 08",  "3 mo",  "2024-06-01", "2024-08-31 23:00"),
    ("B4_autumn", "2024-09 to 11",  "3 mo",  "2024-09-01", "2024-11-30 23:00"),
]


def main():
    # Force forecast configuration that we want for every run.
    program_rolling.ROOM_ID = ROOM
    program_rolling.DATA_SOURCE = "MERGED"
    program_rolling.RUN_PHASE_2 = True
    program_rolling.FIXED_CUA_OVERRIDE = None
    program_rolling.MERGED_FORECAST_END_DATE = FORECAST_END
    program_rolling.MERGED_FORECAST_DAYS = FORECAST_DAYS
    program_rolling.HUNTING_MONTHS_FILTER = None

    rows = []
    for tag, period, length, ts, te in CONFIGS:
        run_tag = f"sensitivity/{tag}"
        print("\n" + "=" * 78)
        print(f"### RUN {tag}  train=[{ts} .. {te}]  ({length})")
        print("=" * 78)
        try:
            res = program_rolling.main(
                run_tag=run_tag,
                train_start=ts,
                train_end=te,
                merged_forecast_end_date=FORECAST_END,
                merged_forecast_days=FORECAST_DAYS,
                hunting_months_filter=None,
            )
        except Exception as e:
            print(f"[ERROR] Run {tag} crashed: {e}")
            traceback.print_exc()
            res = None

        cua = (res or {}).get("fixed_CUA") or {}
        mae = (res or {}).get("mae_by_horizon") or {}
        rows.append({
            "Run":         tag.split("_", 1)[0],
            "Train":       period,
            "Length":      length,
            "Champion":    str((res or {}).get("champion_window_start") or "-"),
            "C":           cua.get("C_Btu_per_F"),
            "UA":          cua.get("UA_Btu_per_hrF"),
            "Q_base":      cua.get("Q_base_BtuHr"),
            "MAE_1h":      mae.get("1h Gated"),
            "MAE_24h":     mae.get("24h"),
        })

    # === Final summary table ===
    print("\n\n" + "=" * 92)
    print("SENSITIVITY EXPERIMENT SUMMARY - BLDG.ZONE_A, forecast = June 2025")
    print("=" * 92)
    header = ["Run", "Train period", "Length", "Champion window",
             "C", "UA", "Q_base", "1h MAE", "24h MAE"]
    widths = [4, 19, 6, 16, 8, 8, 10, 8, 8]

    def fmt(val, w, prec=None):
        if val is None or (isinstance(val, float) and (val != val)):
            s = "-"
        elif prec is not None and isinstance(val, (int, float)):
            s = f"{val:.{prec}f}"
        else:
            s = str(val)
        return s.ljust(w)

    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        line = " | ".join([
            fmt(r["Run"],      widths[0]),
            fmt(r["Train"],    widths[1]),
            fmt(r["Length"],   widths[2]),
            fmt(r["Champion"], widths[3]),
            fmt(r["C"],        widths[4], 1),
            fmt(r["UA"],       widths[5], 2),
            fmt(r["Q_base"],   widths[6], 1),
            fmt(r["MAE_1h"],   widths[7], 3),
            fmt(r["MAE_24h"],  widths[8], 3),
        ])
        print(line)

    # Persist the table for downstream use.
    try:
        out_csv = os.path.join(REPO_ROOT, "Room_Temp_Rolling", "output",
                               "BLDG.ZONE_A", "sensitivity", "summary.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nSaved summary -> {out_csv}")
    except Exception as e:
        print(f"[WARN] Could not write summary csv: {e}")


if __name__ == "__main__":
    sys.exit(main())
