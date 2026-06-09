#!/usr/bin/env python3
"""N4SID 1R1C identification reference baseline.

Implements the subspace identification method described in
Jiang et al. 2025 (Building and Environment, "Development of a grey-box heat
load prediction model by subspace identification method for heating building")
as a baseline for comparison against the WLS-with-intercept method in rc_quiet_learner.py.

Discrete-time 1R1C (forward-Euler, dt = 1 hr):
    T[k+1] = A0*T[k] + b1*Q_mech[k] + b2*T_out[k]
where  A0 = 1 - UA*dt/C,  b1 = dt/C,  b2 = UA*dt/C
=>     C  = dt / b1,        UA = b2 / b1   (no intercept term)

For each room: load data, pick an identification window (champion from
hunting_summary.csv where available), restrict to quiet hours, run N4SID with
M=2 (per-night Hankels concatenated), then forecast over a 31-day window and
capture 1h gated MAE and 24h rollout MAE. Repeats for the WLS_intercept baseline.
Writes reference_method_comparison.csv and parameter_confidence_boxplot.png.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rc_forecast_pipeline import (
    AIR_HEAT_CAPACITY_FACTOR,
    add_time_keys,
    run_forecast_quiet,
    run_pipeline_quiet,
    _learn_C_UA_quiet,
)
from rc_quiet_learner import QuietConfig, TOWQuietLearner

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "Room_Temp_Rolling", "output")
REF_OUT_DIR = os.path.join(BASE_DIR, "Room_Temp_Rolling", "reference_n4sid")
os.makedirs(REF_OUT_DIR, exist_ok=True)

ZONE_C_DATA_DIR = os.path.join(DATA_DIR, "zone_c", "bldg_data")
HUNTING_CSV = os.path.join(OUT_DIR, "hunting_summary.csv")

ID_WINDOW_DAYS = 30
FORECAST_DAYS = 31
W_PER_BTUHR = 1.0 / 3.412141633


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_room_full(room: str) -> pd.DataFrame:
    """Load full hourly time series for a room with the columns the pipeline expects."""
    if room == "ZONE_C":
        from loaders.load_vav_unit_c import load_vav_unit_c
        df = load_vav_unit_c(verbose=False)
        df_naive = df.copy()
        df_naive.index = df_naive.index.tz_localize(None)
        df = df_naive.reset_index().rename(columns={"Datetime": "Date"})
        return add_time_keys(df).sort_values("Date").reset_index(drop=True)

    if room in ("ZONE_A", "ZONE_B"):
        token = "BLDG.ZONE_A" if room == "ZONE_A" else "BLDG.ZONE_B"
        files = [
            os.path.join(DATA_DIR, f"{token}_from_Jun-17_Jul-17_2025_Hourly.xlsx"),
            os.path.join(DATA_DIR, f"{token}_from_Jul-17_Aug-17_2025_Hourly.xlsx"),
            os.path.join(DATA_DIR, f"{token}_from_Aug-17_Sep-17_2025_Hourly.xlsx"),
        ]
        frames = []
        for f in files:
            if os.path.exists(f):
                d = pd.read_excel(f, sheet_name=0)
                d = d.rename(columns={"Room Temp(F)": "Room Temp (F)"})
                frames.append(d)
        if not frames:
            raise FileNotFoundError(f"No data files found for {room}")
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        return add_time_keys(df).reset_index(drop=True)

    raise ValueError(f"Unknown room: {room}")


def champion_window_from_csv(csv_path: str) -> Optional[pd.Timestamp]:
    """Return window_start of the lowest-MAE valid row in hunting_summary.csv."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, parse_dates=["window_start"])
    valid = df[(df["ua_valid"] == True) & (df["c_valid"] == True)]  # noqa: E712
    if valid.empty:
        return None
    return pd.Timestamp(valid.sort_values("mae").iloc[0]["window_start"])


def slice_window(df: pd.DataFrame, start: pd.Timestamp, days: int) -> pd.DataFrame:
    end = start + pd.Timedelta(days=days)
    out = df[(df["Date"] >= start) & (df["Date"] < end)].copy().reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# N4SID identification
# ---------------------------------------------------------------------------

@dataclass
class N4SIDResult:
    A0: float
    b1: float
    b2: float
    C_BtuF: float
    UA_BtuHrF: float
    tau_hr: float
    n_runs: int
    J_total: int
    sigma1: float
    sigma2: float


def _build_block_hankel(seq: np.ndarray, M: int) -> np.ndarray:
    """Block-Hankel matrix with M block-rows; row block r covers samples r .. r+(J-1)."""
    seq = np.atleast_2d(seq)
    if seq.shape[0] != 1 and seq.shape[1] == 1:
        seq = seq.T
    if seq.ndim == 1:
        seq = seq.reshape(1, -1)
    if seq.shape[0] < seq.shape[1]:
        # already (d, N)
        d, N = seq.shape
    else:
        seq = seq.T
        d, N = seq.shape
    J = N - M + 1
    H = np.empty((M * d, J), dtype=float)
    for r in range(M):
        H[r * d:(r + 1) * d, :] = seq[:, r:r + J]
    return H


def _split_contiguous_runs(quiet_df: pd.DataFrame, max_gap_hours: float = 1.5) -> List[pd.DataFrame]:
    """Split a quiet-hour dataframe into contiguous runs (gap <= max_gap_hours)."""
    if quiet_df.empty:
        return []
    ts = quiet_df["ts"].values
    gaps = np.diff(ts).astype("timedelta64[s]").astype(float) / 3600.0
    breaks = np.where(gaps > max_gap_hours)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(quiet_df)]])
    return [quiet_df.iloc[s:e].reset_index(drop=True) for s, e in zip(starts, ends)]


def n4sid_1r1c_imperial(
    train_df: pd.DataFrame,
    M: int = 2,
    quiet_only: bool = True,
) -> N4SIDResult:
    """Identify a 1R1C model in imperial units via N4SID (M=2); returns C, UA from b1, b2.

    Builds a block-Hankel from each contiguous quiet-hour run, concatenates columns
    across runs, then LQ / SVD / projection. quiet_only restricts to quiet hours first.
    """
    learner = TOWQuietLearner(cfg=QuietConfig())
    from contextlib import redirect_stdout
    with open(os.devnull, "w") as f:
        with redirect_stdout(f):
            prepped = learner._prep(train_df)

    BTUHR_PER_W = 3.412141633
    prepped["q_mech_btuhr"] = prepped["Q_mech_W"] * BTUHR_PER_W

    if quiet_only:
        wk = (prepped["dow"] <= 4) & (prepped["hour"] >= 0) & (prepped["hour"] < 6)
        we = (prepped["dow"] >= 5) & (prepped["hour"] >= 0) & (prepped["hour"] < 8)
        q = prepped.loc[wk | we].copy()
    else:
        q = prepped.copy()
    q = q[np.isfinite(q["t_room_f"]) & np.isfinite(q["t_out_f"]) & np.isfinite(q["q_mech_btuhr"])]
    q = q.reset_index(drop=True)

    runs = _split_contiguous_runs(q, max_gap_hours=1.5)
    Up_list, Yp_list, Uf_list, Yf_list = [], [], [], []
    nu = 2  # Q_mech, T_out
    for run in runs:
        if len(run) < 2 * M + 1:
            continue
        y = run["t_room_f"].astype(float).values  # (N,)
        u = np.column_stack([
            run["q_mech_btuhr"].astype(float).values,  # Q_mech [BTU/hr]
            run["t_out_f"].astype(float).values,       # T_out [F]
        ])  # (N, nu)
        N = len(y)
        J_run = N - 2 * M + 1
        # Past: rows 0..M-1, columns 0..J_run-1
        Yp = np.array([y[r:r + J_run] for r in range(M)])                      # (M, J_run)
        Yf = np.array([y[r + M:r + M + J_run] for r in range(M)])              # (M, J_run)
        Up = np.vstack([np.array([u[r:r + J_run, c] for r in range(M)])         # past inputs
                        for c in range(nu)])                                    # (M*nu, J_run)
        Uf = np.vstack([np.array([u[r + M:r + M + J_run, c] for r in range(M)])
                        for c in range(nu)])                                    # (M*nu, J_run)
        Up_list.append(Up); Yp_list.append(Yp); Uf_list.append(Uf); Yf_list.append(Yf)

    if not Up_list:
        raise RuntimeError(
            f"Insufficient contiguous quiet-hour samples to build Hankel matrices "
            f"(have {len(runs)} runs, none with >= {2*M+1} rows)."
        )

    Up = np.hstack(Up_list)
    Yp = np.hstack(Yp_list)
    Uf = np.hstack(Uf_list)
    Yf = np.hstack(Yf_list)
    J = Up.shape[1]

    # Wp = [Up; Yp]
    Wp = np.vstack([Up, Yp])  # (M*nu + M, J)

    # LQ decomposition of Z = [Uf; Wp; Yf]
    Z = np.vstack([Uf, Wp, Yf])
    # numpy QR is on Z.T -> Z.T = Q_qr * R_qr; so Z = R_qr.T * Q_qr.T = L * Q
    Q_qr, R_qr = np.linalg.qr(Z.T, mode="reduced")
    L = R_qr.T

    n_uf = M * nu
    n_wp = Wp.shape[0]
    R22 = L[n_uf:n_uf + n_wp, n_uf:n_uf + n_wp]
    R32 = L[n_uf + n_wp:, n_uf:n_uf + n_wp]

    # SVD of R32 (size M x n_wp). Order n=1.
    U1, S1, _Vt = np.linalg.svd(R32, full_matrices=False)
    n_order = 1
    Gamma2 = U1[:, :n_order]                                                    # (M, 1)

    # State sequence: Xf = pinv(Gamma2) @ R32 @ pinv(R22) @ Wp     (1, J)
    Xf = np.linalg.pinv(Gamma2) @ R32 @ np.linalg.pinv(R22) @ Wp

    # Solve x[k+1] = A0 x[k] + B0 u[k] (no intercept)
    Xf_M = Xf[:, :J - 1]
    Xf_M1 = Xf[:, 1:J]
    # First time-step block of Uf is rows for u(M); but with the column-stack
    # convention above, Uf rows are [Q at lag 0 .. lag M-1, T_out at lag 0 .. M-1]
    # We need u(M) which is the M-th input sample of each column => row 0 of each
    # input's M-row block.
    Uf_M = np.vstack([Uf[0:1, :J - 1], Uf[M:M + 1, :J - 1]])                    # (nu, J-1)

    Z_lhs = np.vstack([Xf_M, Uf_M])                                             # (1+nu, J-1)
    theta, *_ = np.linalg.lstsq(Z_lhs.T, Xf_M1.T, rcond=None)                  # (1+nu, 1)
    A0 = float(theta[0, 0])
    b1 = float(theta[1, 0])
    b2 = float(theta[2, 0])

    # The state sequence Xf returned by the projection is in an arbitrary basis;
    # for n=1 it is x = c * y for some scalar c. Recover c from the *projected*
    # state at the same column indices that line up with y(M..M+J-1).
    # We have y(M+k) for k=0..J-1 in Yf row 0 (= y[M], y[M+1], ...).
    y_aligned = Yf[0, :]                                                        # (J,)
    # least-squares scaling: y = c * x  =>  c = (x . y) / (x . x)
    x_flat = Xf[0, :]
    if np.dot(x_flat, x_flat) > 1e-12:
        c_scale = float(np.dot(x_flat, y_aligned) / np.dot(x_flat, x_flat))
    else:
        c_scale = 1.0
    # In the y-basis, A is invariant (scalar). B coefficients scale with c.
    # x[k+1] = A0 x[k] + B0 u[k]   with x = y/c (since y = c*x => x = y/c)
    # rewrite for y: y[k+1]/c = A0 (y[k]/c) + B0 u[k]
    # =>  y[k+1] = A0 y[k] + (c * B0) u[k]
    # so the physical b1, b2 are c * (subspace b1, b2):
    b1_phys = c_scale * b1
    b2_phys = c_scale * b2

    dt = 1.0  # hour
    if abs(b1_phys) < 1e-12:
        C_BtuF = np.nan
        UA = np.nan
        tau = np.nan
    else:
        C_BtuF = dt / b1_phys                                                  # BTU/F
        UA = b2_phys / b1_phys                                                 # BTU/(hr*F)
        tau = C_BtuF / UA if abs(UA) > 1e-12 else np.nan

    return N4SIDResult(
        A0=A0, b1=b1_phys, b2=b2_phys,
        C_BtuF=float(C_BtuF), UA_BtuHrF=float(UA), tau_hr=float(tau),
        n_runs=len(Up_list), J_total=int(J),
        sigma1=float(S1[0]) if len(S1) >= 1 else np.nan,
        sigma2=float(S1[1]) if len(S1) >= 2 else np.nan,
    )


# ---------------------------------------------------------------------------
# Forecasting helpers
# ---------------------------------------------------------------------------

def run_forecast_with_params(
    df_window: pd.DataFrame,
    df_forecast: pd.DataFrame,
    fixed_CUA: dict,
    workdir: str,
) -> dict:
    """Train residual-correction layers on `df_window`, then forecast `df_forecast`.

    Uses run_pipeline_quiet with `fixed_CUA` so the physics layer uses the
    supplied (C, UA, Q_base) instead of re-fitting them. Returns 1h gated MAE
    and 24h rollout MAE on the forecast window.
    """
    os.makedirs(workdir, exist_ok=True)
    train_path = os.path.join(workdir, "train_window.xlsx")
    fore_path = os.path.join(workdir, "forecast_window.xlsx")
    df_window.to_excel(train_path, index=False)
    df_forecast.to_excel(fore_path, index=False)

    from contextlib import redirect_stdout
    with open(os.path.join(workdir, "_train_log.txt"), "w") as f:
        with redirect_stdout(f):
            res = run_pipeline_quiet(train_path, workdir, workdir, fixed_CUA=fixed_CUA)
    params_path = res["params_path"]

    init_temp = float(df_window["Room Temp (F)"].dropna().iloc[-1])
    with open(os.path.join(workdir, "_fore_log.txt"), "w") as f:
        with redirect_stdout(f):
            forecast_path = run_forecast_quiet(
                params_path, fore_path, workdir,
                init_temp=init_temp,
                sigma_phys=0.5,  # neutral default
            )
    pred = pd.read_excel(forecast_path, sheet_name="forecast")
    if "Room Temp (F)" not in pred.columns:
        return {"MAE_1h": float("nan"), "MAE_24h": float("nan")}
    y = pred["Room Temp (F)"].values
    mae_1h = float(np.nanmean(np.abs(y - pred["T_full_gated (F)"].values)))
    mae_24h = float(np.nanmean(np.abs(y - pred["T_rollout_24h (F)"].values)))
    return {"MAE_1h": mae_1h, "MAE_24h": mae_24h}


# ---------------------------------------------------------------------------
# Per-window WLS sweep for the boxplot
# ---------------------------------------------------------------------------

def sweep_wls_windows(
    df_full: pd.DataFrame,
    window_days: int = 30,
    stride_days: int = 2,
) -> pd.DataFrame:
    """Walk every `stride_days`-step 30-day window and run the WLS learner.

    Returns a DataFrame with columns window_start, C_BtuF, UA_BtuHrF, Q_base_BtuHr,
    valid (True if the learner returned finite positive C and UA).
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
        if len(win) < 24 * 7:
            continue
        try:
            from contextlib import redirect_stdout
            with open(os.devnull, "w") as f:
                with redirect_stdout(f):
                    cua = _learn_C_UA_quiet(win, verbose_quiet=False)
            C = float(cua["C_Btu_per_F"])
            UA = float(cua["UA_Btu_per_hrF"])
            Qb = float(cua.get("Q_base_BtuHr", 0.0))
            ok = np.isfinite(C) and np.isfinite(UA) and C > 0 and UA > 0
            rows.append(dict(window_start=d_start.date(), C_BtuF=C, UA_BtuHrF=UA,
                             Q_base_BtuHr=Qb, valid=bool(ok)))
        except Exception as e:
            rows.append(dict(window_start=d_start.date(), C_BtuF=np.nan,
                             UA_BtuHrF=np.nan, Q_base_BtuHr=np.nan, valid=False))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-room driver
# ---------------------------------------------------------------------------

def find_champion(df_full: pd.DataFrame, window_days: int = 30,
                  stride_days: int = 2) -> Optional[pd.Timestamp]:
    """Walk every stride_days-step 30-day window; return start of lowest-MAE WLS-valid window."""
    days = pd.to_datetime(df_full["Date"]).dt.normalize().unique()
    days = np.sort(pd.to_datetime(days))
    best = None
    best_mae = np.inf
    from contextlib import redirect_stdout
    for i in range(0, max(0, len(days) - window_days), stride_days):
        d_start = pd.Timestamp(days[i])
        d_end = d_start + pd.Timedelta(days=window_days)
        win = df_full[(df_full["Date"] >= d_start) & (df_full["Date"] < d_end)]
        if len(win) < 24 * 7:
            continue
        try:
            with open(os.devnull, "w") as f:
                with redirect_stdout(f):
                    cua = _learn_C_UA_quiet(win, verbose_quiet=False)
            C = float(cua["C_Btu_per_F"]); UA = float(cua["UA_Btu_per_hrF"])
            if not (np.isfinite(C) and np.isfinite(UA) and C > 0 and UA > 0):
                continue
            # quick MAE proxy: physics-only one-step on this window
            T = win["Room Temp (F)"].astype(float).values
            Tn = T[1:]; T0 = T[:-1]
            Tout = win["Outdoor Temp (F)"].astype(float).values[:-1]
            cfm = win["VAV Discharge Air Volume (ft^3 / min)"].astype(float).values[:-1]
            t_vav = win["VAV Discharge Air Temp (F)"].astype(float).values[:-1]
            t_ahu = win["AHU Discharge Air Temp (F)"].astype(float).values[:-1]
            Q = AIR_HEAT_CAPACITY_FACTOR * cfm * (t_vav - t_ahu)
            Q_base = float(cua.get("Q_base_BtuHr", 0.0))
            T_phys = T0 + 1.0 * ((UA / C) * (Tout - T0) + (Q + Q_base) / C)
            mae = float(np.nanmean(np.abs(Tn - T_phys)))
            if mae < best_mae:
                best_mae = mae
                best = d_start
        except Exception:
            continue
    return best


def run_room(room: str) -> dict:
    print(f"\n========== {room} ==========")

    df_full = load_room_full(room)
    print(f"  Loaded: {len(df_full):>5d} rows  ({df_full['Date'].min()} -> {df_full['Date'].max()})")

    # --- Champion / identification window ---
    if room == "ZONE_C":
        champ = champion_window_from_csv(HUNTING_CSV)
        if champ is None:
            champ = pd.Timestamp(df_full["Date"].iloc[0]).normalize()
        print(f"  Champion window start (from hunting_summary.csv): {champ.date()}")
    else:
        # hunt a champion window across the full concatenated data
        # (no per-room hunting_summary exists, so re-run the WLS learner here)
        champ = find_champion(df_full, window_days=ID_WINDOW_DAYS, stride_days=2)
        if champ is None:
            d0 = pd.Timestamp(df_full["Date"].iloc[0]).normalize()
            print(f"  [warn] no valid WLS window found, falling back to {d0.date()}")
            champ = d0
        else:
            print(f"  Champion window start (per-room WLS hunt): {champ.date()}")

    id_window = slice_window(df_full, champ, ID_WINDOW_DAYS)
    print(f"  Identification rows: {len(id_window)}")

    # --- WLS_intercept identification ---
    cua_wls = _learn_C_UA_quiet(id_window, verbose_quiet=False)
    print(f"  WLS:    C={cua_wls['C_Btu_per_F']:>10.1f} BTU/F, "
          f"UA={cua_wls['UA_Btu_per_hrF']:>7.2f} BTU/hr/F, "
          f"Q_base={cua_wls['Q_base_BtuHr']:>8.1f} BTU/hr")

    # --- N4SID identification ---
    try:
        n4 = n4sid_1r1c_imperial(id_window, M=2, quiet_only=True)
        print(f"  N4SID:  C={n4.C_BtuF:>10.1f} BTU/F, "
              f"UA={n4.UA_BtuHrF:>7.2f} BTU/hr/F  (A0={n4.A0:.4f}, "
              f"sv1/sv2={n4.sigma1:.2f}/{n4.sigma2:.2f}, J={n4.J_total}, runs={n4.n_runs})")
    except RuntimeError as e:
        print(f"  N4SID FAILED: {e}")
        n4 = N4SIDResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan, np.nan)

    # --- Forecast window (last 31 days of available data) ---
    last_day = pd.Timestamp(df_full["Date"].max()).normalize()
    fore_start = last_day - pd.Timedelta(days=FORECAST_DAYS)
    train_start = fore_start - pd.Timedelta(days=ID_WINDOW_DAYS)
    df_train = df_full[(df_full["Date"] >= train_start) & (df_full["Date"] < fore_start)].copy()
    df_fore = df_full[(df_full["Date"] >= fore_start) & (df_full["Date"] <= last_day)].copy()

    workdir_w = os.path.join(REF_OUT_DIR, f"{room}_WLS")
    workdir_n = os.path.join(REF_OUT_DIR, f"{room}_N4SID")

    if len(df_train) < 24 * 7 or len(df_fore) < 24:
        print(f"  [warn] not enough data for forecast (train={len(df_train)}, fore={len(df_fore)})")
        wls_mae = {"MAE_1h": np.nan, "MAE_24h": np.nan}
        n4_mae = {"MAE_1h": np.nan, "MAE_24h": np.nan}
    else:
        wls_mae = run_forecast_with_params(
            df_train, df_fore,
            fixed_CUA={
                "C_Btu_per_F": cua_wls["C_Btu_per_F"],
                "UA_Btu_per_hrF": cua_wls["UA_Btu_per_hrF"],
                "Q_base_BtuHr": cua_wls["Q_base_BtuHr"],
            },
            workdir=workdir_w,
        )
        if np.isfinite(n4.C_BtuF) and np.isfinite(n4.UA_BtuHrF) and n4.C_BtuF != 0 and n4.UA_BtuHrF != 0:
            n4_mae = run_forecast_with_params(
                df_train, df_fore,
                fixed_CUA={
                    "C_Btu_per_F": n4.C_BtuF,
                    "UA_Btu_per_hrF": n4.UA_BtuHrF,
                    "Q_base_BtuHr": 0.0,
                },
                workdir=workdir_n,
            )
        else:
            n4_mae = {"MAE_1h": np.nan, "MAE_24h": np.nan}
        print(f"  WLS forecast:    1h MAE={wls_mae['MAE_1h']:.3f}, 24h MAE={wls_mae['MAE_24h']:.3f}")
        print(f"  N4SID forecast:  1h MAE={n4_mae['MAE_1h']:.3f}, 24h MAE={n4_mae['MAE_24h']:.3f}")

    # --- Supplementary: N4SID on the first 30-day window (zone rooms only).
    # Demonstrates that N4SID's parameter sign is window-sensitive: on the first
    # window ZONE_B exhibits the negative-UA failure mode that the hunted
    # champion window avoids.
    n4_jun17 = None
    if room in ("ZONE_A", "ZONE_B"):
        d0 = pd.Timestamp(df_full["Date"].iloc[0]).normalize()
        win_jun17 = slice_window(df_full, d0, ID_WINDOW_DAYS)
        try:
            n4j = n4sid_1r1c_imperial(win_jun17, M=2, quiet_only=True)
            print(f"  [supp] N4SID on Jun-17 file:  C={n4j.C_BtuF:>10.1f}, "
                  f"UA={n4j.UA_BtuHrF:>7.2f}  (A0={n4j.A0:.4f})")
            n4_jun17 = dict(C=n4j.C_BtuF, UA=n4j.UA_BtuHrF, A0=n4j.A0)
        except RuntimeError as e:
            print(f"  [supp] N4SID on Jun-17 file FAILED: {e}")

    return dict(
        room=room,
        wls=dict(C=cua_wls["C_Btu_per_F"], UA=cua_wls["UA_Btu_per_hrF"],
                 Qb=cua_wls["Q_base_BtuHr"], mae=wls_mae),
        n4sid=dict(C=n4.C_BtuF, UA=n4.UA_BtuHrF, mae=n4_mae,
                   A0=n4.A0, sigma1=n4.sigma1, sigma2=n4.sigma2),
        n4sid_jun17=n4_jun17,
        df_full=df_full,
    )


# ---------------------------------------------------------------------------
# Boxplot of C and UA across hunting windows
# ---------------------------------------------------------------------------

def make_param_boxplot(room_results: List[dict], outpath: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for i, r in enumerate(room_results):
        room = r["room"]
        df_full = r["df_full"]
        # smaller stride for the shorter zone series so we see more candidates
        stride = 4 if room == "ZONE_C" else 1
        sweep = sweep_wls_windows(df_full, window_days=30, stride_days=stride)
        n_total = len(sweep)
        valid = sweep[sweep["valid"]]
        n_valid = len(valid)
        ax_c = axes[0, i]
        ax_ua = axes[1, i]

        if n_valid >= 3:
            ax_c.boxplot(valid["C_BtuF"].values, widths=0.55,
                         patch_artist=True,
                         boxprops=dict(facecolor="#9ecae1", alpha=0.85),
                         medianprops=dict(color="black", lw=2))
            ax_ua.boxplot(valid["UA_BtuHrF"].values, widths=0.55,
                          patch_artist=True,
                          boxprops=dict(facecolor="#a1d99b", alpha=0.85),
                          medianprops=dict(color="black", lw=2))
        elif n_valid >= 1:
            # too few for a boxplot - show as scatter at x=1
            ax_c.scatter(np.ones(n_valid), valid["C_BtuF"].values,
                         s=80, color="#3182bd", edgecolor="black", zorder=3,
                         label=f"WLS samples (n={n_valid})")
            ax_ua.scatter(np.ones(n_valid), valid["UA_BtuHrF"].values,
                          s=80, color="#31a354", edgecolor="black", zorder=3,
                          label=f"WLS samples (n={n_valid})")
        else:
            ax_c.text(0.5, 0.5, "no valid WLS windows",
                      transform=ax_c.transAxes, ha="center", va="center", fontsize=11)
            ax_ua.text(0.5, 0.5, "no valid WLS windows",
                       transform=ax_ua.transAxes, ha="center", va="center", fontsize=11)

        # WLS champion (the single value reported in the comparison table)
        wls_C = r["wls"]["C"]; wls_UA = r["wls"]["UA"]
        if np.isfinite(wls_C):
            ax_c.axhline(wls_C, color="#08519c", lw=1.5, ls="-",
                         label=f"WLS champion = {wls_C:,.0f}")
        if np.isfinite(wls_UA):
            ax_ua.axhline(wls_UA, color="#006d2c", lw=1.5, ls="-",
                          label=f"WLS champion = {wls_UA:.2f}")

        # N4SID horizontal markers
        n4_C = r["n4sid"]["C"]; n4_UA = r["n4sid"]["UA"]
        if np.isfinite(n4_C):
            ax_c.axhline(n4_C, color="red", lw=2, ls="--",
                         label=f"N4SID = {n4_C:,.0f}")
        if np.isfinite(n4_UA):
            ax_ua.axhline(n4_UA, color="red", lw=2, ls="--",
                          label=f"N4SID = {n4_UA:.2f}")

        ax_c.set_yscale("log")
        ax_c.set_title(f"{room}  (valid: {n_valid}/{n_total} windows)")
        ax_c.set_xticks([])
        ax_ua.set_xticks([])
        if i == 0:
            ax_c.set_ylabel("C  (BTU/°F, log scale)")
            ax_ua.set_ylabel("UA  (BTU/hr/°F)")
        ax_ua.axhline(0, color="black", lw=1, ls=":")
        ax_c.legend(loc="best", fontsize=8)
        ax_ua.legend(loc="best", fontsize=8)
        ax_c.grid(True, alpha=0.3, which="both")
        ax_ua.grid(True, alpha=0.3)

    fig.suptitle("WLS parameter spread across all valid 30-day hunting windows  "
                 "(red dashed = N4SID single-shot estimate; solid = WLS champion)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rooms = ["ZONE_A", "ZONE_B", "ZONE_C"]
    results = []
    for room in rooms:
        try:
            results.append(run_room(room))
        except Exception as e:
            print(f"[{room}] FAILED: {e}")
            import traceback; traceback.print_exc()

    # --- Save comparison CSV ---
    rows = []
    for r in results:
        rows.append(dict(
            room=r["room"], method="WLS_intercept",
            C_BtuF=r["wls"]["C"], UA_BtuHrF=r["wls"]["UA"],
            Q_base_W=r["wls"]["Qb"] * W_PER_BTUHR,
            tau_hr=r["wls"]["C"] / r["wls"]["UA"] if r["wls"]["UA"] != 0 else np.nan,
            UA_negative=bool(r["wls"]["UA"] < 0),
            MAE_1h=r["wls"]["mae"]["MAE_1h"], MAE_24h=r["wls"]["mae"]["MAE_24h"],
        ))
        rows.append(dict(
            room=r["room"], method="N4SID",
            C_BtuF=r["n4sid"]["C"], UA_BtuHrF=r["n4sid"]["UA"],
            Q_base_W=0.0,
            tau_hr=r["n4sid"]["C"] / r["n4sid"]["UA"] if r["n4sid"]["UA"] not in (0, np.nan) else np.nan,
            UA_negative=bool(r["n4sid"]["UA"] < 0) if np.isfinite(r["n4sid"]["UA"]) else False,
            MAE_1h=r["n4sid"]["mae"]["MAE_1h"], MAE_24h=r["n4sid"]["mae"]["MAE_24h"],
        ))
    # Supplementary rows for zone rooms: N4SID on the first 30-day window,
    # exposing the negative-UA failure that motivates the comparison.
    for r in results:
        nj = r.get("n4sid_jun17")
        if nj is None:
            continue
        rows.append(dict(
            room=r["room"], method="N4SID_Jun17_window",
            C_BtuF=nj["C"], UA_BtuHrF=nj["UA"], Q_base_W=0.0,
            tau_hr=nj["C"] / nj["UA"] if (np.isfinite(nj["UA"]) and nj["UA"] != 0) else np.nan,
            UA_negative=bool(nj["UA"] < 0) if np.isfinite(nj["UA"]) else False,
            MAE_1h=np.nan, MAE_24h=np.nan,  # not forecast in supplementary path
        ))

    out_csv = os.path.join(REF_OUT_DIR, "reference_method_comparison.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # --- Boxplot ---
    out_png = os.path.join(REF_OUT_DIR, "parameter_confidence_boxplot.png")
    print(f"Building per-room WLS sweep for boxplot (this can take a minute)...")
    make_param_boxplot(results, out_png)
    print(f"Saved: {out_png}")

    # --- Final summary table ---
    print("\n" + "=" * 100)
    header = (f"{'Room':<10}{'Method':<16}{'C [BTU/F]':>14}{'UA [BTU/hr/F]':>16}"
              f"{'Q_base [W]':>12}{'tau [hr]':>10}{'1h MAE':>10}{'24h MAE':>10}")
    print(header)
    print("-" * 100)
    for row in rows:
        flag = "  NEG-UA" if row["UA_negative"] else ""
        c_str = "n/a" if not np.isfinite(row["C_BtuF"]) else f"{row['C_BtuF']:>14,.1f}"
        ua_str = "n/a" if not np.isfinite(row["UA_BtuHrF"]) else f"{row['UA_BtuHrF']:>16.2f}"
        tau_str = "n/a" if not np.isfinite(row["tau_hr"]) else f"{row['tau_hr']:>10.2f}"
        m1 = "n/a" if not np.isfinite(row["MAE_1h"]) else f"{row['MAE_1h']:>10.3f}"
        m24 = "n/a" if not np.isfinite(row["MAE_24h"]) else f"{row['MAE_24h']:>10.3f}"
        print(
            f"{row['room']:<10}{row['method']:<16}"
            f"{c_str:>14}{ua_str:>16}"
            f"{row['Q_base_W']:>12.1f}{tau_str:>10}{m1:>10}{m24:>10}{flag}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
