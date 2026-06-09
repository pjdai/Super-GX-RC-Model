#!/usr/bin/env python3
"""Rolling forecast orchestrator.

Phase 1 (Parameter Hunting): scan training windows, for each one run the quiet
learner and pick the (C, UA, Q_base) triple whose Physics-only one-step MAE
is best across the database.

Phase 2 (Locked Rolling Forecast): re-anchor the chosen (C, UA, Q_base) and
roll the hybrid model forward day-by-day across the forecast window.
"""
import os
import sys
import json
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow flat imports of the core modules / loaders package from the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rc_forecast_pipeline import (
    run_pipeline_quiet, run_forecast_quiet, add_time_keys,
)

# === Path configuration (all paths derived from the repo root) ===
DATA_DIR     = os.path.join(REPO_ROOT, "data")
ROLLING_BASE = os.path.join(REPO_ROOT, "Room_Temp_Rolling")
OUTDIR_BASE  = os.path.join(ROLLING_BASE, "output")
FIGDIR_BASE  = os.path.join(ROLLING_BASE, "figures")

# Data source selector:
#   "EXCEL"  - original Excel triplet (Jun-Jul / Jul-Aug / Aug-Sep)
#   "ZONE_C" - BAS CSV bundle via bas_loader.load_bas_data
#   "MERGED" - single merged xlsx (BAS + open-meteo) for one zone
DATA_SOURCE  = "ZONE_C"

# Switch ROOM_ID to forecast a different zone - the three xlsx file names below
# are templated off it (Jun-Jul training, Jul-Aug training, Aug-Sep forecast).
ROOM_ID      = "BLDG.ZONE_A"
INPUT_TRAIN1 = os.path.join(DATA_DIR, f"{ROOM_ID}_from_Jun-17_Jul-17_2025_Hourly.xlsx")
INPUT_TRAIN2 = os.path.join(DATA_DIR, f"{ROOM_ID}_from_Jul-17_Aug-17_2025_Hourly.xlsx")
INPUT_FORE   = os.path.join(DATA_DIR, f"{ROOM_ID}_from_Aug-17_Sep-17_2025_Hourly.xlsx")

# ZONE_C configuration (only used when DATA_SOURCE == "ZONE_C").
ZONE_C_DATA_DIR       = os.path.join(DATA_DIR, "zone_c", "bldg_data")
ZONE_C_FORECAST_DAYS  = 30   # tail of the ZONE_C series treated as the forecast window

# MERGED configuration (only used when DATA_SOURCE == "MERGED").
MERGED_FILE        = os.path.join(DATA_DIR, "merged_ZONE_A_weather.xlsx")
MERGED_FORECAST_DAYS = 30  # span of forecast window in days
# If None, forecast window ends at the last timestamp in the merged file
# (default tail behavior). Set to a parseable date string to anchor the
# end of the forecast window elsewhere - useful for evaluating a model
# in a specific season.
MERGED_FORECAST_END_DATE: object = "2026-02-28 23:00"

# Phase 2 toggle - set False to stop after Phase 1 (parameter hunt) only.
RUN_PHASE_2  = True

# Optional Phase 1 bypass: if set to a dict {C_Btu_per_F, UA_Btu_per_hrF, Q_base_BtuHr},
# skip the parameter hunt and use these values directly as the locked CUA for Phase 2.
FIXED_CUA_OVERRIDE = None

# Per-room output isolation: append ROOM_ID to output/figure dirs so successive
# runs for different rooms don't overwrite each other's hunt summary.
# RUN_TAG further isolates this run (e.g. "warm_only") from the full-year run.
RUN_TAG      = "cool_only"
OUTDIR_BASE  = os.path.join(OUTDIR_BASE, ROOM_ID, RUN_TAG)
FIGDIR_BASE  = os.path.join(FIGDIR_BASE, ROOM_ID, RUN_TAG)

# Training-data slice (Phase 1 hunt + Phase 2 rolling training window).
# Set to a parseable date string to restrict df_all to >= TRAIN_START and/or
# <= TRAIN_END. Phase 2 still evaluates the forecast window against actuals
# from the unfiltered data, so the restriction only affects what the learner
# sees, not what's being forecast.
TRAIN_START: object = None
TRAIN_END:   object = None

# === Tuning constants ===
WINDOW_DAYS               = 60     # rolling-forecast training window length [days]
# Per-source hunt tuning. Each source can override window length and month
# filter (months refer to the candidate window's START date; None = no filter).
HUNTING_CONFIG = {
    "EXCEL":  {"window_days": 30, "months_filter": {10, 11, 12, 1, 2, 3}},
    "MERGED": {"window_days": 30, "months_filter": {10, 11, 12, 1, 2, 3}},
    "ZONE_C": {"window_days": 20, "months_filter": {9, 10, 11, 12, 1, 2, 3}},
}
HUNTING_WINDOW_DAYS   = HUNTING_CONFIG[DATA_SOURCE]["window_days"]
HUNTING_MONTHS_FILTER = HUNTING_CONFIG[DATA_SOURCE]["months_filter"]
MIN_SIGNAL_STRENGTH_F     = 3.5    # require avg |T_out - T_room| above this to bother fitting [deg-F]
HUNTING_MAE_ACCEPT        = 10.0   # ignore candidates worse than this even if best so far [deg-F]
SIGMA_PHYS_DEFAULT        = 0.5    # fallback gating sigma if training residuals are too few [deg-F]
SIGMA_PHYS_CLIP           = (0.1, 2.0)  # clamp sigma_phys to this range [deg-F]
MIN_SIGMA_SAMPLES         = 20     # need this many residuals to compute sigma_phys
MIN_TRAINING_HOURS        = 24 * 7 # minimum rolling-window size to bother forecasting a day


def load_and_fix(path: str) -> pd.DataFrame:
    """Read an Excel file and patch the typo `Room Temp(F)` -> `Room Temp (F)` if present."""
    df = pd.read_excel(path)
    df = df.rename(columns={"Room Temp(F)": "Room Temp (F)"})
    return df


def validate_data(df: pd.DataFrame, name: str) -> None:
    """Print row count, columns, and the first three rows - catches column-naming mismatches early."""
    print(f"\n=== Validating: {name} ===")
    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(df.head(3).to_string(index=False))


def _classify_runtime_error(msg: str):
    """Map a learner RuntimeError message to (ua_valid, c_valid) flags for the hunt summary."""
    if "Not enough steady-state snapshots" in msg or "WLS regression" in msg or "UA still invalid" in msg:
        return False, None  # never got past UA step
    if "Not enough quiet samples" in msg or "1/C invalid" in msg:
        return True, False
    return None, None


def main(run_tag=None, train_start=None, train_end=None,
         merged_forecast_end_date=None, merged_forecast_days=None,
         hunting_months_filter="__use_module__"):
    """Run Phase 1 hunt + (optional) Phase 2 rolling forecast.

    All parameters default to module-level constants when None / unset.
    `hunting_months_filter` uses a sentinel because None is a meaningful value
    (no month filter) different from "fall back to module default".

    Returns a results dict with champion CUA, champion window start, and
    horizon-resolved anchor MAEs (or None if Phase 2 was skipped / failed).
    """
    run_tag = RUN_TAG if run_tag is None else run_tag
    train_start = TRAIN_START if train_start is None else train_start
    train_end = TRAIN_END if train_end is None else train_end
    merged_forecast_end_date = (
        MERGED_FORECAST_END_DATE if merged_forecast_end_date is None
        else merged_forecast_end_date
    )
    merged_forecast_days = (
        MERGED_FORECAST_DAYS if merged_forecast_days is None
        else merged_forecast_days
    )
    if hunting_months_filter == "__use_module__":
        hunting_months_filter = HUNTING_MONTHS_FILTER

    outdir_base = os.path.join(ROLLING_BASE, "output", ROOM_ID, run_tag)
    figdir_base = os.path.join(ROLLING_BASE, "figures", ROOM_ID, run_tag)
    os.makedirs(outdir_base, exist_ok=True)
    os.makedirs(figdir_base, exist_ok=True)

    # --- 1. Load and prepare data ---
    # Both branches converge on a single `df` whose index is named 'Datetime'
    # and whose columns match the canonical schema. Downstream code consumes
    # `df_all` (Date column + time keys) and `df_f` (forecast window).
    if DATA_SOURCE == "EXCEL":
        df_t1 = load_and_fix(INPUT_TRAIN1)
        df_t2 = load_and_fix(INPUT_TRAIN2)
        df_f  = load_and_fix(INPUT_FORE)

        validate_data(df_t1, os.path.basename(INPUT_TRAIN1))
        validate_data(df_t2, os.path.basename(INPUT_TRAIN2))
        validate_data(df_f,  os.path.basename(INPUT_FORE))

        df_t1 = add_time_keys(df_t1); df_t2 = add_time_keys(df_t2); df_f = add_time_keys(df_f)
        df_all = pd.concat([df_t1, df_t2, df_f], ignore_index=True).sort_values("Date").reset_index(drop=True)

        # Unified Datetime-indexed view for the cross-source validation print.
        df = df_all.drop(columns=["hour", "dow", "how"], errors="ignore").set_index("Date")
        df.index.name = "Datetime"

    elif DATA_SOURCE == "ZONE_C":
        from loaders.load_vav_unit_c import load_vav_unit_c
        df = load_vav_unit_c()
        # Convert to the (Date column, naive timestamps) form the rest of the
        # pipeline assumes. add_time_keys() reads the "Date" column.
        df_naive = df.copy()
        df_naive.index = df_naive.index.tz_localize(None)
        df_all = df_naive.reset_index().rename(columns={"Datetime": "Date"})
        df_all = add_time_keys(df_all).sort_values("Date").reset_index(drop=True)

        # Treat the tail of the ZONE_C series as the forecast window.
        last_date = df_all["Date"].max()
        fore_start = last_date - pd.Timedelta(days=ZONE_C_FORECAST_DAYS)
        df_f = df_all[df_all["Date"] >= fore_start].copy()

    elif DATA_SOURCE == "MERGED":
        from rc_quiet_learner import _resolve_cols
        df_raw = pd.read_excel(MERGED_FILE)
        cols = _resolve_cols(df_raw)

        # Convert outdoor temp from open-meteo °C to °F if needed (mirrors the
        # conversion already wired inside TOWQuietLearner._prep).
        out_src = cols["t_out_f"]
        if out_src == "temperature_2m":
            df_raw[out_src] = df_raw[out_src] * 9.0 / 5.0 + 32.0

        # Rename resolved columns -> canonical names that downstream pipeline
        # functions (run_pipeline_quiet, compute_q_mech_btu_per_hr, etc.) expect.
        rename = {
            cols["ts"]:         "Date",
            cols["t_room_f"]:   "Room Temp (F)",
            cols["t_out_f"]:    "Outdoor Temp (F)",
            cols["t_supply_f"]: "VAV Discharge Air Temp (F)",
            cols["t_ahu_f"]:    "AHU Discharge Air Temp (F)",
            cols["cfm"]:        "VAV Discharge Air Volume (ft^3 / min)",
        }
        if "ghi" in cols: rename[cols["ghi"]] = "GHI (W/m²)"
        if "dni" in cols: rename[cols["dni"]] = "DNI (W/m²)"
        df_norm = df_raw.rename(columns=rename)

        # Drop rows where any required pipeline input is still NaN (gaps too
        # large to interpolate at merge time). The downstream pipeline reads
        # these columns by hard-coded name and does not tolerate NaN.
        required = [
            "Room Temp (F)", "Outdoor Temp (F)",
            "VAV Discharge Air Temp (F)", "AHU Discharge Air Temp (F)",
            "VAV Discharge Air Volume (ft^3 / min)",
            "GHI (W/m²)", "DNI (W/m²)",
        ]
        before = len(df_norm)
        df_norm = df_norm.dropna(subset=required).reset_index(drop=True)
        print(f"[MERGED] Dropped {before - len(df_norm)} rows with NaN in required cols (kept {len(df_norm)}).")

        df_all = add_time_keys(df_norm).sort_values("Date").reset_index(drop=True)
        df = df_all.drop(columns=["hour", "dow", "how"], errors="ignore").set_index("Date")
        df.index.name = "Datetime"

        if merged_forecast_end_date is not None:
            end_date = pd.Timestamp(merged_forecast_end_date)
        else:
            end_date = pd.to_datetime(df_all["Date"]).max()
        fore_start = end_date - pd.Timedelta(days=merged_forecast_days)
        df_f = df_all[(df_all["Date"] >= fore_start) & (df_all["Date"] <= end_date)].copy()

    else:
        raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE!r}. Use 'EXCEL', 'ZONE_C', or 'MERGED'.")

    # --- 1b. Source-agnostic validation summary ---
    print(f"\n=== Data validation ({DATA_SOURCE}) ===")
    print(f"  Rows:       {len(df)}")
    print(f"  Columns:    {list(df.columns)}")
    print(f"  Date range: {df.index.min()}  ->  {df.index.max()}")

    # Preserve unfiltered df_all so Phase 2 can still pull per-day actuals
    # from the forecast window when the training slice is restricted.
    df_all_full = df_all.copy()

    if train_start is not None or train_end is not None:
        before_rows = len(df_all)
        if train_start is not None:
            df_all = df_all[df_all["Date"] >= pd.Timestamp(train_start)]
        if train_end is not None:
            df_all = df_all[df_all["Date"] <= pd.Timestamp(train_end)]
        df_all = df_all.reset_index(drop=True)
        print(
            f"[TRAIN-SLICE] train_start={train_start}, train_end={train_end}: "
            f"{before_rows} -> {len(df_all)} rows "
            f"({df_all['Date'].min()} -> {df_all['Date'].max()})"
        )

    # --- PHASE 1: PARAMETER HUNTING (skipped if FIXED_CUA_OVERRIDE is set) ---
    best_mae = 999.0
    final_fixed_CUA = None
    champion_window_start = None

    if FIXED_CUA_OVERRIDE is not None:
        final_fixed_CUA = dict(FIXED_CUA_OVERRIDE)
        print(
            f"\nStep 1: Phase 1 bypassed via FIXED_CUA_OVERRIDE -> "
            f"C={final_fixed_CUA['C_Btu_per_F']:.1f}, "
            f"UA={final_fixed_CUA['UA_Btu_per_hrF']:.2f}, "
            f"Q_base={final_fixed_CUA['Q_base_BtuHr']:.2f} BTU/hr"
        )
        unique_days = []
        hunt_records = []
    else:
        print("\nStep 1: Hunting for the best physical C / UA / Q_base across the database...")
        unique_days = pd.to_datetime(df_all["Date"]).dt.date.unique()
        hunt_records = []  # rows for the end-of-Phase-1 summary table

    for i in range(len(unique_days) - HUNTING_WINDOW_DAYS):
        d_start = pd.Timestamp(unique_days[i])
        if hunting_months_filter is not None and d_start.month not in hunting_months_filter:
            continue
        d_end   = d_start + pd.Timedelta(days=HUNTING_WINDOW_DAYS)
        df_win  = df_all[(df_all["Date"] >= d_start) & (df_all["Date"] < d_end)].copy()

        rec = {
            "window_start":  d_start.date(),
            "signal_pass":   True,
            "ua_valid":      None,
            "c_valid":       None,
            "mae":           None,
            "Q_base_BtuHr":  None,
            "reason":        "",
        }

        # Signal-strength filter
        avg_dT = (df_win["Outdoor Temp (F)"] - df_win["Room Temp (F)"]).abs().mean()
        if avg_dT < MIN_SIGNAL_STRENGTH_F:
            rec["signal_pass"] = False
            rec["reason"] = f"avg|dT|={avg_dT:.2f}F < {MIN_SIGNAL_STRENGTH_F}F"
            hunt_records.append(rec)
            continue

        try:
            outdir_hunt = os.path.join(outdir_base, "hunting_temp")
            tmp_path = os.path.join(outdir_hunt, "hunt_win.xlsx")
            os.makedirs(outdir_hunt, exist_ok=True)
            df_win.to_excel(tmp_path, index=False)

            res = run_pipeline_quiet(tmp_path, outdir_hunt, outdir_hunt)
            curr_mae = res["metrics_dict"]["Physics-only"]["MAE"]
            q_base = float(res["fixed_CUA"].get("Q_base_BtuHr", 0.0))

            rec["ua_valid"]     = True
            rec["c_valid"]      = True
            rec["mae"]          = round(float(curr_mae), 4)
            rec["Q_base_BtuHr"] = round(q_base, 2)

            if curr_mae < HUNTING_MAE_ACCEPT and curr_mae < best_mae:
                best_mae = curr_mae
                final_fixed_CUA = res["fixed_CUA"]
                champion_window_start = d_start.date()
                print(
                    f"[HUNT] New Champion: MAE={best_mae:.3f} on {d_start.date()} "
                    f"with C={final_fixed_CUA['C_Btu_per_F']:.1f}, "
                    f"UA={final_fixed_CUA['UA_Btu_per_hrF']:.2f}, "
                    f"Q_base={final_fixed_CUA['Q_base_BtuHr']:.2f} BTU/hr"
                )
        except RuntimeError as re:
            ua_v, c_v = _classify_runtime_error(str(re))
            rec["ua_valid"] = ua_v
            rec["c_valid"]  = c_v
            rec["reason"]   = str(re)
            print(f"[SKIP] Window {d_start.date()} invalid physics: {re}")
        except Exception as e:
            rec["reason"] = f"CRASH: {e}"
            print(f"[DEBUG] Window {d_start.date()} code crashed:")
            traceback.print_exc()
            hunt_records.append(rec)
            break

        hunt_records.append(rec)

    # --- Phase 1 summary table ---
    if hunt_records:
        summary_df = pd.DataFrame(hunt_records)
        # Truncate long reason strings for table display
        summary_df["reason"] = summary_df["reason"].astype(str).str.slice(0, 80)
        print("\n=== Phase 1 Hunting Summary ===")
        print(summary_df.to_string(index=False))
        try:
            summary_df.to_csv(os.path.join(outdir_base, "hunting_summary.csv"), index=False)
        except Exception as e:
            print(f"[WARN] Could not write hunting_summary.csv: {e}")

    if final_fixed_CUA is None:
        print("\nCRITICAL: No parameters survived the hunt. Loosen MIN_SIGNAL_STRENGTH_F or extend the data window.")
        return {
            "run_tag": run_tag,
            "champion_window_start": None,
            "fixed_CUA": None,
            "mae_by_horizon": {},
            "best_hunt_mae": None,
        }

    if not RUN_PHASE_2:
        print("\nRUN_PHASE_2 is False. Stopping after Phase 1.")
        return {
            "run_tag": run_tag,
            "champion_window_start": champion_window_start,
            "fixed_CUA": dict(final_fixed_CUA),
            "mae_by_horizon": {},
            "best_hunt_mae": float(best_mae) if best_mae < 999.0 else None,
        }

    # --- PHASE 2: LOCKED ROLLING FORECAST ---
    print(
        f"\nStep 2: Locked CUA at C={final_fixed_CUA['C_Btu_per_F']:.1f}, "
        f"UA={final_fixed_CUA['UA_Btu_per_hrF']:.2f}, "
        f"Q_base={final_fixed_CUA['Q_base_BtuHr']:.2f} BTU/hr. Running rolling forecast..."
    )

    fore_dates = pd.to_datetime(df_f["Date"])
    forecast_days = np.sort(pd.to_datetime(fore_dates.dt.date.unique()))

    all_preds = []
    success_days = 0

    for day in forecast_days:
        day_ts = pd.Timestamp(day)
        try:
            start_win = day_ts - pd.Timedelta(days=WINDOW_DAYS)
            # Phase 2 rolling lead-up uses the unfiltered series so off-season
            # training slices (e.g. winter-only) still have recent data to anchor
            # the locked-CUA model when forecasting June 2025.
            df_win = df_all_full[(df_all_full["Date"] >= start_win) & (df_all_full["Date"] < day_ts)].copy()
            df_day = df_all_full[(df_all_full["Date"] >= day_ts) & (df_all_full["Date"] < (day_ts + pd.Timedelta(days=1)))].copy()

            if len(df_win) < MIN_TRAINING_HOURS or len(df_day) < 2:
                continue

            outdir = os.path.join(outdir_base, f"forecast_day_{day_ts.date()}")
            figdir = os.path.join(figdir_base, f"forecast_day_{day_ts.date()}")
            os.makedirs(outdir, exist_ok=True); os.makedirs(figdir, exist_ok=True)

            tmp_train_path = os.path.join(outdir, "training_window.xlsx")
            df_win.to_excel(tmp_train_path, index=False)

            result = run_pipeline_quiet(tmp_train_path, outdir, figdir, fixed_CUA=final_fixed_CUA)
            params_path = result["params_path"]

            # Compute sigma_phys for gating from training residuals
            train_xlsx = os.path.join(outdir, "predictions_and_metrics_TOW_QUIET.xlsx")
            sigma_phys = SIGMA_PHYS_DEFAULT
            if os.path.exists(train_xlsx):
                df_res = pd.read_excel(train_xlsx, sheet_name="predictions")
                if "T_actual_next" in df_res.columns and "T_phys_next" in df_res.columns:
                    err = df_res["T_actual_next"] - df_res["T_phys_next"]
                    clean_err = err[np.isfinite(err)].dropna()
                    if len(clean_err) > MIN_SIGMA_SAMPLES:
                        sigma_phys = float(np.std(clean_err))
                        sigma_phys = float(np.clip(sigma_phys, *SIGMA_PHYS_CLIP))

            last_actual_T = float(df_win["Room Temp (F)"].dropna().iloc[-1])
            tmp_fore_path = os.path.join(outdir, "predict_day.xlsx")
            df_day.to_excel(tmp_fore_path, index=False)

            forecast_path = run_forecast_quiet(
                params_path, tmp_fore_path, outdir,
                init_temp=last_actual_T,
                sigma_phys=sigma_phys,
            )

            # Second pass: propagating rollout (same params, different output dir).
            outdir_prop = os.path.join(outdir, "propagating")
            os.makedirs(outdir_prop, exist_ok=True)
            forecast_path_prop = run_forecast_quiet(
                params_path, tmp_fore_path, outdir_prop,
                init_temp=last_actual_T,
                sigma_phys=sigma_phys,
                propagating_rollout=True,
            )

            df_pred = pd.read_excel(forecast_path, sheet_name="forecast")
            cols_needed = [
                "Date", "T_phys_only (F)", "T_full_gated (F)",
                "T_rollout_3h (F)", "T_rollout_6h (F)",
                "T_rollout_12h (F)", "T_rollout_24h (F)",
                "Gating_Factor", "D_solar",
            ]
            existing_cols = [c for c in cols_needed if c in df_pred.columns]
            keep = df_pred[existing_cols].copy()

            df_pred_prop = pd.read_excel(forecast_path_prop, sheet_name="forecast")
            prop_rollout_cols = [c for c in df_pred_prop.columns if c.startswith("T_rollout_")]
            prop_renamed = df_pred_prop[["Date"] + prop_rollout_cols].rename(
                columns={c: c.replace(" (F)", "_prop (F)") for c in prop_rollout_cols}
            )
            keep = keep.merge(prop_renamed, on="Date", how="left")

            if "Room Temp (F)" in df_day.columns:
                keep = keep.merge(df_day[["Date", "Room Temp (F)"]], on="Date", how="left")

            keep["day"] = day_ts
            all_preds.append(keep)
            success_days += 1

        except Exception as e:
            print(f"[ROLLING] Error on {day_ts.date()}: {e}")
            continue

    # --- 3. Result Collection ---
    mae_by_horizon: dict = {}
    if all_preds:
        big = pd.concat(all_preds, ignore_index=True).sort_values("Date")
        big.to_excel(os.path.join(outdir_base, "rolling_predictions_vs_actual.xlsx"), index=False)

        fig, ax1 = plt.subplots(figsize=(15, 8))
        ax1.plot(pd.to_datetime(big["Date"]), big["T_full_gated (F)"], label="1h Gated", alpha=0.8)
        ax1.plot(pd.to_datetime(big["Date"]), big["T_rollout_6h (F)"], "--", label="6h Rollout", alpha=0.7)
        if "T_rollout_12h (F)" in big.columns:
            ax1.plot(pd.to_datetime(big["Date"]), big["T_rollout_12h (F)"], ":", label="12h Rollout", alpha=0.6)
        if "T_rollout_24h (F)" in big.columns:
            ax1.plot(pd.to_datetime(big["Date"]), big["T_rollout_24h (F)"], "-.", label="24h Rollout", alpha=0.9, color="red")
        if "Room Temp (F)" in big.columns:
            ax1.plot(pd.to_datetime(big["Date"]), big["Room Temp (F)"], "k", label="Actual Room Temp", alpha=0.4, linewidth=2)
        ax1.set_ylabel("Temperature (F)")
        ax1.legend(loc="upper left")

        if "D_solar" in big.columns:
            ax2 = ax1.twinx()
            ax2.fill_between(pd.to_datetime(big["Date"]), 0, big["D_solar"], color="orange", alpha=0.15, label="Solar Temp Gain (F)")
            ax2.set_ylabel("Solar Temperature Gain (F)", color="orange")
            if big["D_solar"].max() > 0:
                ax2.set_ylim(0, big["D_solar"].max() * 4)
            ax2.legend(loc="upper right")

        plt.title(f"Multi-Scale Rollout Comparison with Solar Gain (Locked CUA, Best MAE={best_mae:.3f})")
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(figdir_base, "final_long_term_comparison.png"))
        plt.close()
        print(f"\nRolling Forecast Success: {success_days} days. Output saved to {outdir_base}.")

        # --- Side-by-side anchor vs propagating MAE ---
        if "Room Temp (F)" in big.columns:
            y = big["Room Temp (F)"].values
            comparison = [
                ("1h Gated", "T_full_gated (F)", None),
                ("3h",       "T_rollout_3h (F)",  "T_rollout_3h_prop (F)"),
                ("6h",       "T_rollout_6h (F)",  "T_rollout_6h_prop (F)"),
                ("12h",      "T_rollout_12h (F)", "T_rollout_12h_prop (F)"),
                ("24h",      "T_rollout_24h (F)", "T_rollout_24h_prop (F)"),
            ]
            print("\n=== Anchor-based vs Propagating Rollout MAE ===")
            print(f"{'Horizon':<10} | {'Anchor MAE':>12} | {'Propagating MAE':>16}")
            print("-" * 46)
            for name, anchor_col, prop_col in comparison:
                if anchor_col not in big.columns:
                    continue
                anchor_mae = float(np.nanmean(np.abs(y - big[anchor_col].values)))
                mae_by_horizon[name] = anchor_mae
                if prop_col is None:
                    prop_str = "(same, N/A)"
                elif prop_col in big.columns:
                    prop_mae = float(np.nanmean(np.abs(y - big[prop_col].values)))
                    prop_str = f"{prop_mae:>16.3f}"
                else:
                    prop_str = "n/a"
                print(f"{name:<10} | {anchor_mae:>12.3f} | {prop_str}")

    return {
        "run_tag": run_tag,
        "champion_window_start": champion_window_start,
        "fixed_CUA": dict(final_fixed_CUA) if final_fixed_CUA is not None else None,
        "mae_by_horizon": mae_by_horizon,
        "best_hunt_mae": float(best_mae) if best_mae < 999.0 else None,
        "success_days": success_days if all_preds else 0,
    }


if __name__ == "__main__":
    main()
