#!/usr/bin/env python3
"""Visualize how C, UA, MAE vary across all candidate 30-day hunting windows per zone.

Per-room:
  - Zone rooms (ZONE_A, ZONE_B): re-run sweep_wls_windows() with stride_days=1 on
    the concatenated dataset reference_n4sid.load_room_full() returns; compute a
    physics-only one-step MAE per window (the same proxy find_champion uses) and
    the signal_pass / ua_valid / c_valid flags consistent with the hunting logic.
  - ZONE_C: re-run sweep_wls_windows() to recover C / UA per window, then merge the
    existing hunting_summary.csv flags onto window_start.

Figure: 3 cols (rooms) x 3 rows (C, UA, MAE), saved to
Room_Temp_Rolling/output/window_parameter_sweep.png at 18x14 inches, DPI 150.
"""
from __future__ import annotations

import os
import sys
import warnings
from contextlib import redirect_stdout
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Allow flat imports of the core modules / loaders package from the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Local imports — reuse the existing pipeline.
from reference_n4sid import load_room_full
from rc_forecast_pipeline import (
    AIR_HEAT_CAPACITY_FACTOR,
    _learn_C_UA_quiet,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_DIR = os.path.join(REPO_ROOT, "Room_Temp_Rolling", "output")
HUNTING_CSV = os.path.join(OUT_DIR, "hunting_summary.csv")
OUT_PNG = os.path.join(OUT_DIR, "window_parameter_sweep.png")

WINDOW_DAYS = 30
STRIDE_DAYS = 1
MIN_SIGNAL_STRENGTH_F = 3.5  # matches run_rolling_forecast.MIN_SIGNAL_STRENGTH_F

# Physically plausible bands for an office zone (BTU/F, BTU/hr/F).
C_BAND = (10_000.0, 100_000.0)
UA_BAND = (0.5, 50.0)


# ---------------------------------------------------------------------------
# Per-window sweep that mirrors run_rolling_forecast.py's hunting flags.
# ---------------------------------------------------------------------------

def _physics_one_step_mae(win: pd.DataFrame, C: float, UA: float,
                          Q_base: float) -> float:
    """Physics-only one-step prediction MAE on the entire window.

    Matches the proxy used in reference_n4sid.find_sdh_champion (forward-Euler,
    Q_mech = 1.08 * CFM * (T_vav - T_ahu)).
    """
    T = win["Room Temp (F)"].astype(float).values
    if len(T) < 2:
        return float("nan")
    Tn = T[1:]
    T0 = T[:-1]
    Tout = win["Outdoor Temp (F)"].astype(float).values[:-1]
    cfm = win["VAV Discharge Air Volume (ft^3 / min)"].astype(float).values[:-1]
    t_vav = win["VAV Discharge Air Temp (F)"].astype(float).values[:-1]
    t_ahu = win["AHU Discharge Air Temp (F)"].astype(float).values[:-1]
    Q = AIR_HEAT_CAPACITY_FACTOR * cfm * (t_vav - t_ahu)
    if not np.isfinite(C) or C == 0:
        return float("nan")
    T_phys = T0 + 1.0 * ((UA / C) * (Tout - T0) + (Q + Q_base) / C)
    return float(np.nanmean(np.abs(Tn - T_phys)))


def sweep_all_windows(df_full: pd.DataFrame,
                      window_days: int = WINDOW_DAYS,
                      stride_days: int = STRIDE_DAYS) -> pd.DataFrame:
    """Return one row per candidate 30-day window with the full hunting flags.

    Columns:
        window_start, C_BtuF, UA_BtuHrF, Q_base_BtuHr, mae,
        signal_pass, ua_valid, c_valid, reason
    """
    rows = []
    if df_full.empty:
        return pd.DataFrame(rows)
    days = pd.to_datetime(df_full["Date"]).dt.normalize().unique()
    days = np.sort(pd.to_datetime(days))
    for i in range(0, max(0, len(days) - window_days), stride_days):
        d_start = pd.Timestamp(days[i])
        d_end = d_start + pd.Timedelta(days=window_days)
        win = df_full[(df_full["Date"] >= d_start) & (df_full["Date"] < d_end)]

        rec = {
            "window_start": d_start,
            "C_BtuF": np.nan,
            "UA_BtuHrF": np.nan,
            "Q_base_BtuHr": np.nan,
            "mae": np.nan,
            "signal_pass": True,
            "ua_valid": False,
            "c_valid": False,
            "reason": "",
        }

        if len(win) < 24 * 7:
            rec["signal_pass"] = False
            rec["reason"] = f"too few rows ({len(win)})"
            rows.append(rec)
            continue

        avg_dT = (win["Outdoor Temp (F)"] - win["Room Temp (F)"]).abs().mean()
        if not np.isfinite(avg_dT) or avg_dT < MIN_SIGNAL_STRENGTH_F:
            rec["signal_pass"] = False
            rec["reason"] = f"avg|dT|={avg_dT:.2f}F < {MIN_SIGNAL_STRENGTH_F}F"
            rows.append(rec)
            continue

        try:
            with open(os.devnull, "w") as f:
                with redirect_stdout(f):
                    cua = _learn_C_UA_quiet(win, verbose_quiet=False)
            C = float(cua["C_Btu_per_F"])
            UA = float(cua["UA_Btu_per_hrF"])
            Qb = float(cua.get("Q_base_BtuHr", 0.0))
            rec["C_BtuF"] = C
            rec["UA_BtuHrF"] = UA
            rec["Q_base_BtuHr"] = Qb
            rec["ua_valid"] = bool(np.isfinite(UA) and UA > 0)
            rec["c_valid"] = bool(np.isfinite(C) and C > 0)
            rec["mae"] = _physics_one_step_mae(win, C, UA, Qb)
        except Exception as exc:
            rec["reason"] = f"learner failed: {exc}"
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-room data preparation
# ---------------------------------------------------------------------------

def prepare_room(room: str) -> pd.DataFrame:
    """Run the sweep for one room. For ZONE_C, overlay hunting_summary.csv flags."""
    print(f"  loading {room} ...", flush=True)
    df_full = load_room_full(room)
    print(f"    {len(df_full)} rows; "
          f"{df_full['Date'].min()} -> {df_full['Date'].max()}", flush=True)

    print(f"  sweeping windows (stride={STRIDE_DAYS}d, "
          f"win={WINDOW_DAYS}d) ...", flush=True)
    sweep = sweep_all_windows(df_full, WINDOW_DAYS, STRIDE_DAYS)
    print(f"    {len(sweep)} candidate windows", flush=True)

    if room == "ZONE_C" and os.path.exists(HUNTING_CSV):
        # The CSV's mae / signal_pass / ua_valid / c_valid / Q_base override
        # our re-computed proxies for ZONE_C (they came from run_pipeline_quiet
        # so they are the authoritative numbers for that room).
        hunt = pd.read_csv(HUNTING_CSV, parse_dates=["window_start"])
        hunt = hunt[[
            "window_start", "signal_pass", "ua_valid", "c_valid",
            "mae", "Q_base_BtuHr", "reason",
        ]].rename(columns={
            "signal_pass": "signal_pass_csv",
            "ua_valid": "ua_valid_csv",
            "c_valid": "c_valid_csv",
            "mae": "mae_csv",
            "Q_base_BtuHr": "Q_base_csv",
            "reason": "reason_csv",
        })
        sweep = sweep.merge(hunt, on="window_start", how="left")
        # Cast numeric columns first so loc assignment doesn't trip the
        # bool/float dtype-mixing FutureWarning.
        sweep["signal_pass"] = sweep["signal_pass"].astype(object)
        sweep["ua_valid"] = sweep["ua_valid"].astype(object)
        sweep["c_valid"] = sweep["c_valid"].astype(object)
        for src, dst in [
            ("signal_pass_csv", "signal_pass"),
            ("ua_valid_csv", "ua_valid"),
            ("c_valid_csv", "c_valid"),
            ("mae_csv", "mae"),
            ("Q_base_csv", "Q_base_BtuHr"),
            ("reason_csv", "reason"),
        ]:
            mask = sweep[src].notna()
            sweep.loc[mask, dst] = sweep.loc[mask, src]
        sweep = sweep.drop(columns=[c for c in sweep.columns if c.endswith("_csv")])
        # Cast back after the dtype-mixing merge.
        for c in ("signal_pass", "ua_valid", "c_valid"):
            sweep[c] = sweep[c].astype(bool)
        print(f"    merged hunting_summary.csv flags onto ZONE_C windows", flush=True)

    return sweep


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _add_month_axis(ax) -> None:
    """Add a secondary x-axis on top with month labels."""
    secax = ax.secondary_xaxis("top")
    secax.xaxis.set_major_locator(mdates.MonthLocator())
    secax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    secax.tick_params(axis="x", labelsize=8, pad=2)
    for lbl in secax.get_xticklabels():
        lbl.set_rotation(0)


def _format_date_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(35)
        lbl.set_ha("right")


def make_figure(per_room: Dict[str, pd.DataFrame], outpath: str) -> None:
    rooms = list(per_room.keys())
    n_rooms = len(rooms)

    # Shared MAE colorscale across all valid windows in all rooms.
    valid_mae_all = []
    for df in per_room.values():
        v = df[(df["ua_valid"] == True) & (df["c_valid"] == True)]  # noqa: E712
        valid_mae_all.extend(v["mae"].dropna().tolist())
    if valid_mae_all:
        mae_lo = float(np.nanpercentile(valid_mae_all, 2))
        mae_hi = float(np.nanpercentile(valid_mae_all, 98))
        if mae_hi <= mae_lo:
            mae_hi = mae_lo + 1e-3
    else:
        mae_lo, mae_hi = 0.0, 1.0
    norm = Normalize(vmin=mae_lo, vmax=mae_hi)
    cmap = plt.get_cmap("RdYlGn_r")

    fig, axes = plt.subplots(3, n_rooms, figsize=(18, 14), sharex="col")
    if n_rooms == 1:
        axes = axes.reshape(3, 1)

    for col, room in enumerate(rooms):
        df = per_room[room].copy()
        df = df.sort_values("window_start").reset_index(drop=True)
        df["window_start"] = pd.to_datetime(df["window_start"])

        candidates = df[df["signal_pass"] == True]  # noqa: E712
        valid = df[(df["ua_valid"] == True) & (df["c_valid"] == True)]  # noqa: E712
        n_total = len(candidates)
        n_valid = len(valid)

        # Champion = lowest MAE among valid.
        if len(valid) > 0 and valid["mae"].notna().any():
            champ = valid.loc[valid["mae"].idxmin()]
        else:
            champ = None

        ax_c = axes[0, col]
        ax_ua = axes[1, col]
        ax_mae = axes[2, col]

        # ----- Row 1: C distribution -----
        ax_c.axhspan(C_BAND[0], C_BAND[1], color="#cfe4ff", alpha=0.5, zorder=0,
                     label=f"plausible range {C_BAND[0]:,.0f}-{C_BAND[1]:,.0f}")
        if len(candidates):
            ax_c.scatter(candidates["window_start"],
                         np.abs(candidates["C_BtuF"]).replace(0, np.nan),
                         s=10, color="#bdbdbd", alpha=0.6, zorder=2,
                         label=f"all candidates (n={n_total})")
        if len(valid):
            sc = ax_c.scatter(valid["window_start"], valid["C_BtuF"],
                              c=valid["mae"], cmap=cmap, norm=norm,
                              s=40, edgecolor="black", linewidth=0.5,
                              zorder=3, label=f"valid (n={n_valid})")
        if champ is not None:
            ax_c.axvline(champ["window_start"], ls="--", color="#444", lw=1, zorder=1)
            ax_c.scatter([champ["window_start"]], [champ["C_BtuF"]],
                         marker="*", s=300, color="gold",
                         edgecolor="black", linewidth=1.0, zorder=5,
                         label="champion")
            ax_c.annotate(
                f"C* = {champ['C_BtuF']:,.0f} BTU/°F",
                xy=(champ["window_start"], champ["C_BtuF"]),
                xytext=(8, 10), textcoords="offset points",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="black", alpha=0.85),
            )
        ax_c.set_yscale("log")
        ax_c.set_ylim(bottom=max(1.0, 0.5 * min(C_BAND[0], (valid["C_BtuF"].min()
                                                            if len(valid) else C_BAND[0]))),
                      top=10 * max(C_BAND[1], (valid["C_BtuF"].max()
                                               if len(valid) else C_BAND[1])))
        ax_c.set_ylabel("C (BTU/°F, log)", fontsize=10)
        ax_c.set_title(f"{room}   (valid: {n_valid}/{n_total} windows)",
                       fontsize=12, fontweight="bold")
        ax_c.grid(True, which="both", alpha=0.25)
        ax_c.legend(loc="upper left", fontsize=7, framealpha=0.85)
        _add_month_axis(ax_c)

        # ----- Row 2: UA distribution -----
        ax_ua.axhspan(UA_BAND[0], UA_BAND[1], color="#d4f0d4", alpha=0.5, zorder=0,
                      label=f"plausible range {UA_BAND[0]}-{UA_BAND[1]:.0f}")
        ax_ua.axhline(0.0, color="black", ls="--", lw=1, alpha=0.7, zorder=1)
        if len(candidates):
            ax_ua.scatter(candidates["window_start"], candidates["UA_BtuHrF"],
                          s=10, color="#bdbdbd", alpha=0.6, zorder=2,
                          label=f"all candidates (n={n_total})")
        if len(valid):
            ax_ua.scatter(valid["window_start"], valid["UA_BtuHrF"],
                          c=valid["mae"], cmap=cmap, norm=norm,
                          s=40, edgecolor="black", linewidth=0.5, zorder=3,
                          label=f"valid (n={n_valid})")
        if champ is not None:
            ax_ua.scatter([champ["window_start"]], [champ["UA_BtuHrF"]],
                          marker="*", s=300, color="gold",
                          edgecolor="black", linewidth=1.0, zorder=5,
                          label="champion")
            ax_ua.annotate(
                f"UA* = {champ['UA_BtuHrF']:.2f} BTU/hr/°F",
                xy=(champ["window_start"], champ["UA_BtuHrF"]),
                xytext=(8, 10), textcoords="offset points",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="black", alpha=0.85),
            )
        ax_ua.set_ylabel("UA (BTU/hr/°F)", fontsize=10)
        ax_ua.grid(True, alpha=0.25)
        ax_ua.legend(loc="upper left", fontsize=7, framealpha=0.85)

        # ----- Row 3: MAE across valid windows -----
        if len(valid):
            v = valid.dropna(subset=["mae"]).sort_values("window_start")
            ax_mae.fill_between(v["window_start"], v["mae"], alpha=0.18,
                                color="#1f77b4", step=None)
            ax_mae.plot(v["window_start"], v["mae"], lw=1.4, color="#1f77b4",
                        zorder=2)
            ax_mae.scatter(v["window_start"], v["mae"],
                           c=v["mae"], cmap=cmap, norm=norm,
                           s=35, edgecolor="black", linewidth=0.4, zorder=3)
        if champ is not None and np.isfinite(champ["mae"]):
            ax_mae.axhline(champ["mae"], ls="--", color="#444", lw=1, zorder=1,
                           label=f"champion MAE = {champ['mae']:.3f}°F")
            ax_mae.scatter([champ["window_start"]], [champ["mae"]],
                           marker="*", s=300, color="gold",
                           edgecolor="black", linewidth=1.0, zorder=5)
            cs = champ["window_start"]
            cs_str = cs.strftime("%Y-%m-%d") if hasattr(cs, "strftime") else str(cs)
            txt = (
                f"champion window: {cs_str}\n"
                f"C   = {champ['C_BtuF']:,.0f} BTU/°F\n"
                f"UA  = {champ['UA_BtuHrF']:.2f} BTU/hr/°F\n"
                f"Q_b = {champ['Q_base_BtuHr']:.1f} BTU/hr\n"
                f"MAE = {champ['mae']:.3f} °F"
            )
            ax_mae.text(0.02, 0.97, txt, transform=ax_mae.transAxes,
                        fontsize=8.5, va="top", ha="left", family="monospace",
                        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                                  ec="black", alpha=0.95))
            ax_mae.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax_mae.set_ylabel("MAE (°F, physics-only 1-step)", fontsize=10)
        ax_mae.set_xlabel("window start date", fontsize=10)
        ax_mae.grid(True, alpha=0.25)
        _format_date_axis(ax_mae)

    # Shared colorbar.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(left=0.06, right=0.92, top=0.93, bottom=0.07,
                        hspace=0.28, wspace=0.22)
    cax = fig.add_axes([0.935, 0.10, 0.012, 0.78])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("MAE (°F) on valid windows", fontsize=10)

    fig.suptitle(
        "Parameter sweep across all candidate 30-day identification windows  "
        "(stride = 1 day)",
        fontsize=14, fontweight="bold", y=0.985,
    )
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  saved: {outpath}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rooms = ["ZONE_A", "ZONE_B", "ZONE_C"]
    per_room: Dict[str, pd.DataFrame] = {}
    for room in rooms:
        print(f"\n=== {room} ===", flush=True)
        try:
            per_room[room] = prepare_room(room)
        except Exception as exc:
            print(f"  [{room}] FAILED: {exc}", flush=True)
            import traceback; traceback.print_exc()
            per_room[room] = pd.DataFrame(columns=[
                "window_start", "C_BtuF", "UA_BtuHrF", "Q_base_BtuHr",
                "mae", "signal_pass", "ua_valid", "c_valid", "reason",
            ])

    print(f"\nBuilding figure ...", flush=True)
    make_figure(per_room, OUT_PNG)


if __name__ == "__main__":
    main()
