#!/usr/bin/env python3
"""Hybrid grey-box room-temperature pipeline.

  - Physics: 1st-order RC model on the room air node, parameters (C, UA, Q_base)
    learned from the quiet-hours regression in rc_quiet_learner.py
  - Statistical correction: Time-of-Week schedule (D_tow), bifurcated solar
    (D_solar, clear vs cloudy), VAV-flow-anomaly internal load (Z_internal)
  - Adaptive Gaussian gating with exponential decay across the rollout horizon

Public entry points:
  run_pipeline_quiet()  - learn parameters from a training window, write params + metrics
  run_forecast_quiet()  - roll forward into a forecast window using saved params
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Tuple, Optional

import importlib.util, sys, pathlib

# === Named constants ===
AIR_HEAT_CAPACITY_FACTOR = 1.08      # BTU / (hr * CFM * deg-F): Q_mech = 1.08 * CFM * dT
DEFAULT_DECAY_RATE = 0.90            # lambda: exponential decay of statistical correction
                                     # (D_tow + Z_internal) across rollout horizon

# Solar regime thresholds
DEFAULT_GHI_DAY_THRESHOLD = 10.0     # W/m^2: below this we treat as nighttime
DEFAULT_CLEAR_CUT = 0.35             # DNI/GHI ratio above which we call sky "clear"

# === Unit conversions (SI -> imperial) ===
BTU_PER_J = 1.0 / 1055.06
BTUHR_PER_W = 3.412141633
K_PER_F = 5.0 / 9.0


# === Live-import the quiet learner ===
QUIET_MOD = "rc_quiet_learner"
QUIET_PATH_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "rc_quiet_learner.py"),
    os.path.join(os.getcwd(), "rc_quiet_learner.py"),
]
_found = None
for p in QUIET_PATH_CANDIDATES:
    if os.path.exists(p):
        _found = p
        break
if _found is None:
    raise FileNotFoundError("rc_quiet_learner.py not found nearby.")
spec = importlib.util.spec_from_file_location(QUIET_MOD, _found)
quiet_mod = importlib.util.module_from_spec(spec)
sys.modules[QUIET_MOD] = quiet_mod
spec.loader.exec_module(quiet_mod)
QuietConfig = quiet_mod.QuietConfig
TOWQuietLearner = quiet_mod.TOWQuietLearner


def C_JperK_to_Btu_per_F(C_J_per_K: float) -> float:
    """Convert thermal capacitance from J/K to BTU/deg-F."""
    return float(C_J_per_K * BTU_PER_J * K_PER_F)


def UA_WperK_to_Btu_per_hrF(UA_W_per_K: float) -> float:
    """Convert envelope conductance from W/K to BTU/(hr*deg-F)."""
    return float(UA_W_per_K * BTUHR_PER_W * K_PER_F)


def Q_W_to_Btu_per_hr(Q_W: float) -> float:
    """Convert a steady heat-load rate from W to BTU/hr."""
    return float(Q_W * BTUHR_PER_W)


def add_time_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Attach hour, day-of-week, and hour-of-week (0-167) columns from "Date"."""
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["hour"] = d["Date"].dt.hour
    d["dow"] = d["Date"].dt.dayofweek
    d["how"] = d["dow"] * 24 + d["hour"]
    return d


def infer_dt_hours(dates) -> float:
    """Median sample spacing of a timestamp series, in hours."""
    s = pd.Series(dates).sort_values().diff().dt.total_seconds().dropna()
    if len(s) == 0:
        return 1.0
    return float(s.median()) / 3600.0


def compute_q_mech_btu_per_hr(df: pd.DataFrame) -> np.ndarray:
    """VAV-box mechanical load [BTU/hr]: Q_mech = 1.08 * CFM * (T_VAV - T_AHU)."""
    cfm = df["VAV Discharge Air Volume (ft^3 / min)"].astype(float).values
    t_vav = df["VAV Discharge Air Temp (F)"].astype(float).values
    t_ahu = df["AHU Discharge Air Temp (F)"].astype(float).values
    delta_t = t_vav - t_ahu
    return AIR_HEAT_CAPACITY_FACTOR * cfm * delta_t


def metrics(y_true, y_pred):
    """MAE, MAPE (%), RMSE between two same-length arrays."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(1e-6, np.abs(y_true))))) * 100.0
    return {"MAE": mae, "MAPE": mape, "RMSE": rmse}


def ls_slope(x, y) -> float:
    """OLS slope of y on x with no intercept (returns 0 on empty input)."""
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) == 0:
        return 0.0
    beta = np.linalg.lstsq(x.reshape(-1, 1), y, rcond=None)[0][0]
    return float(beta)


def _learn_C_UA_quiet(df: pd.DataFrame, quiet_cfg: Optional[QuietConfig] = None,
                     verbose_quiet: bool = False) -> Dict[str, float]:
    """Run the quiet-hours learner and convert SI outputs to imperial (C, UA, Q_base, tau)."""
    learner = TOWQuietLearner(cfg=quiet_cfg or QuietConfig(recency_weight_gain=2.0))
    learned = learner.learn_C_UA(df, verbose_quiet=verbose_quiet)
    return {
        "C_Btu_per_F":    C_JperK_to_Btu_per_F(learned["C_J_per_K"]),
        "UA_Btu_per_hrF": UA_WperK_to_Btu_per_hrF(learned["UA_W_per_K"]),
        "Q_base_BtuHr":   Q_W_to_Btu_per_hr(learned["Q_base_W"]),
        "tau_hours":      float(learned["tau_hours"]),
    }


def enforce_fixed_CUA_in_params(params_path: str, fixed_CUA: Optional[dict]) -> None:
    """Overwrite the C / UA / Q_base entries inside a saved params JSON (no-op if None).

    Lets Phase-1 hunting lock in physics that later per-window re-fits don't override.
    """
    if fixed_CUA is None:
        return
    with open(params_path, "r") as f:
        p = json.load(f)
    p["C_Btu_per_F"] = float(fixed_CUA["C_Btu_per_F"])
    p["UA_Btu_per_hrF"] = float(fixed_CUA["UA_Btu_per_hrF"])
    if "Q_base_BtuHr" in fixed_CUA:
        p["Q_base_BtuHr"] = float(fixed_CUA["Q_base_BtuHr"])
    with open(params_path, "w") as f:
        json.dump(p, f, indent=2)


def compute_gating_factor(resid_instant: float, sigma_phys: Optional[float]) -> float:
    """Gaussian gating weight gamma in [0,1] on the statistical correction layer.

    gamma -> 1 when the physics residual is small vs sigma_phys (trust the correction);
    gamma -> 0 when residual >> sigma_phys (anomaly suspected, fall back to physics).
    """
    if sigma_phys is None or sigma_phys <= 0:
        return 1.0
    gamma = np.exp(-0.5 * (resid_instant / sigma_phys) ** 2)
    return float(gamma)


def run_pipeline_quiet(input_path: str, outdir: str, figdir: str,
                       ghi_day_threshold: float = DEFAULT_GHI_DAY_THRESHOLD,
                       clear_cut: float = DEFAULT_CLEAR_CUT,
                       quiet_cfg: Optional[QuietConfig] = None,
                       fixed_CUA: Optional[dict] = None,
                       verbose_quiet: bool = False):
    """Train the hybrid grey-box pipeline on a single window.

    Learns C/UA/Q_base via quiet-hours regression, then fits the D_tow / D_solar /
    Z_internal correction layers on the first 80% of the window. fixed_CUA bypasses
    learning. Returns dict with excel_path, params_path, metrics_df, metrics_dict, fixed_CUA.
    """
    os.makedirs(outdir, exist_ok=True); os.makedirs(figdir, exist_ok=True)
    xl = pd.ExcelFile(input_path)
    df = xl.parse(xl.sheet_names[0])
    df = add_time_keys(df)
    dates = pd.to_datetime(df["Date"], errors="coerce").values
    dt_hr = infer_dt_hours(dates)

    # --- C / UA / Q_base ---
    if fixed_CUA is None:
        CUA = _learn_C_UA_quiet(df, quiet_cfg=quiet_cfg, verbose_quiet=verbose_quiet)
        fixed_CUA = {
            "C_Btu_per_F":    float(CUA["C_Btu_per_F"]),
            "UA_Btu_per_hrF": float(CUA["UA_Btu_per_hrF"]),
            "Q_base_BtuHr":   float(CUA["Q_base_BtuHr"]),
        }
    else:
        CUA = {
            "C_Btu_per_F":    float(fixed_CUA["C_Btu_per_F"]),
            "UA_Btu_per_hrF": float(fixed_CUA["UA_Btu_per_hrF"]),
            "Q_base_BtuHr":   float(fixed_CUA.get("Q_base_BtuHr", 0.0)),
            "tau_hours":      float(fixed_CUA["C_Btu_per_F"] / fixed_CUA["UA_Btu_per_hrF"]),
        }

    C_train       = float(CUA["C_Btu_per_F"])
    UA_train      = float(CUA["UA_Btu_per_hrF"])
    Q_base_train  = float(CUA.get("Q_base_BtuHr", 0.0))
    tau_train_hrs = float(CUA["tau_hours"]) if "tau_hours" in CUA else float(C_train / UA_train)

    # --- Pull all the per-step columns ---
    room = df["Room Temp (F)"].astype(float).values
    outt = df["Outdoor Temp (F)"].astype(float).values
    ghi  = df["GHI (W/m²)"].astype(float).values
    dni  = df["DNI (W/m²)"].astype(float).values
    flow = df["VAV Discharge Air Volume (ft^3 / min)"].astype(float).values
    how  = df["how"].values
    q_mech = compute_q_mech_btu_per_hr(df)

    ua_over_C = UA_train / C_train
    alpha_mech = 1.0 / C_train
    print(
        f"[TRAIN-QUIET] dt_hr={dt_hr:.3f}, C~{C_train:.3f}, UA~{UA_train:.3f}, "
        f"Q_base~{Q_base_train:.2f} BTU/hr, tau~{tau_train_hrs:.2f} h"
    )

    # --- Physics-only one-step prediction (with Q_base) ---
    T = room[:-1]; Tn = room[1:]; Tout = outt[:-1]; Q = q_mech[:-1]
    T_phys_next = T + dt_hr * (
        ua_over_C * (Tout - T) + alpha_mech * (Q + Q_base_train)
    )

    # --- Residual decomposition on first 80% of window ---
    resid_phys = Tn - T_phys_next
    n = len(T); split = int(n * 0.8) if n > 10 else n
    resid_train = resid_phys[:split]; how_train = how[1:][:split]

    # Time-of-week residual schedule (168-bin hourly)
    B_tow = np.zeros(168); counts = np.zeros(168)
    for r, h in zip(resid_train, how_train):
        hbin = int(h) % 168; B_tow[hbin] += r; counts[hbin] += 1.0
    counts[counts == 0] = 1.0; B_tow = B_tow / counts
    D_tow = np.array([B_tow[int(h) % 168] for h in how[1:]])

    # Solar split
    is_day = ghi[1:] >= ghi_day_threshold
    ci = np.divide(dni[1:], np.maximum(ghi[1:], 1e-6))
    is_clear = (ci >= clear_cut) & is_day
    resid_after_tow = resid_phys - D_tow
    ghi_used = ghi[1:]
    beta_s_clear  = ls_slope(ghi_used[:split][is_clear[:split]],  resid_after_tow[:split][is_clear[:split]])  if np.any(is_clear[:split])    else 0.0
    beta_s_cloudy = ls_slope(ghi_used[:split][~is_clear[:split]], resid_after_tow[:split][~is_clear[:split]]) if np.any(~is_clear[:split])   else 0.0
    D_solar = np.zeros_like(ghi_used)
    D_solar[is_day & is_clear]  = beta_s_clear  * ghi_used[is_day & is_clear]
    D_solar[is_day & ~is_clear] = beta_s_cloudy * ghi_used[is_day & ~is_clear]

    # VAV-flow-anomaly internal correction
    Z_proxy = (flow[1:] - np.nanmean(flow[:split])) / (np.nanstd(flow[:split]) + 1e-6)
    resid_after_tow_solar = resid_after_tow - D_solar
    beta_i = ls_slope(Z_proxy[:split], resid_after_tow_solar[:split]) if split > 0 else 0.0
    Z_internal = beta_i * Z_proxy

    D_total = D_tow + D_solar + Z_internal
    T_TOW_next        = T_phys_next + D_tow
    T_TOW_solar2_next = T_phys_next + D_tow + D_solar
    T_full_next       = T_phys_next + D_total

    def M(y, yh): return metrics(y, yh)
    m_pers = M(Tn, T); m_phys = M(Tn, T_phys_next); m_tow = M(Tn, T_TOW_next)
    m_sol2 = M(Tn, T_TOW_solar2_next); m_full = M(Tn, T_full_next)

    out_df = pd.DataFrame({
        "Date": pd.to_datetime(df["Date"]).values[1:],
        "T_actual_next": Tn,
        "T_phys_next": T_phys_next,
        "T_TOW_next": T_TOW_next,
        "T_TOW_solar2_next": T_TOW_solar2_next,
        "T_full_next": T_full_next,
        "D_tow": D_tow, "D_solar": D_solar, "Z_internal": Z_internal, "D_total": D_total,
    })
    metrics_df = pd.DataFrame([
        {"Model": "Persistence",                       **m_pers},
        {"Model": "Physics-only",                      **m_phys},
        {"Model": "Physics + TOW",                     **m_tow},
        {"Model": "Physics + TOW + solar (2 slopes)",  **m_sol2},
        {"Model": "Full (TOW + solar + internal)",     **m_full},
    ])[["Model", "MAE", "MAPE", "RMSE"]]
    metrics_dict = {
        "Persistence": m_pers,
        "Physics-only": m_phys,
        "Physics + TOW": m_tow,
        "Physics + TOW + solar (2 slopes)": m_sol2,
        "Full (TOW + solar + internal)": m_full,
    }

    excel_path = os.path.join(outdir, "predictions_and_metrics_TOW_QUIET.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, sheet_name="predictions", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)

    params = {
        "dt_hr":          float(dt_hr),
        "B_tow":          list(map(float, B_tow)),
        "beta_s_clear":   float(beta_s_clear),
        "beta_s_cloudy":  float(beta_s_cloudy),
        "beta_i":         float(beta_i),
        "C_Btu_per_F":    float(C_train),
        "UA_Btu_per_hrF": float(UA_train),
        "Q_base_BtuHr":   float(Q_base_train),
        "tau_hours":      float(tau_train_hrs),
        "thresholds": {
            "ghi_day_threshold": float(ghi_day_threshold),
            "clear_cut":         float(clear_cut),
        },
    }
    params_path = os.path.join(outdir, "params_TOW_QUIET.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    # Ablation chart
    plt.figure()
    labels = metrics_df["Model"].tolist(); maes = metrics_df["MAE"].tolist()
    plt.bar(range(len(labels)), maes); plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("MAE (F)"); plt.title("Ablation - MAE by variant (QUIET C/UA)"); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "ablation_bar_quiet.png"), bbox_inches="tight"); plt.close()

    # If a fixed CUA was supplied, ensure it (including Q_base) is the value left on disk
    enforce_fixed_CUA_in_params(params_path, fixed_CUA)

    print("[TRAIN-QUIET] Saved:", excel_path)
    print("[TRAIN-QUIET] Saved:", params_path)

    return {
        "excel_path":   excel_path,
        "params_path":  params_path,
        "metrics_df":   metrics_df,
        "metrics_dict": metrics_dict,
        "fixed_CUA":    fixed_CUA,
    }


def run_forecast_quiet(params_path: str, future_input_path: str, outdir: str,
                       init_temp: float, CUA_override: Optional[dict] = None,
                       sigma_phys: Optional[float] = None,
                       decay_rate: float = DEFAULT_DECAY_RATE,
                       propagating_rollout: bool = False):
    """Roll the trained model forward into a forecast window; returns the workbook path.

    Produces five trajectories: a 1h gated forecast (anchored to actual every step),
    3h/6h/12h/24h rollouts (anchor refreshed from actual every N hours), and a pure
    physics one-step forecast. propagating_rollout=True feeds each rollout's own
    previous prediction back as the anchor between resets (errors compound); the
    default False holds the anchor stale between resets. sigma_phys drives gating
    (None disables it); decay_rate is the per-step gating decay across the horizon.
    """
    os.makedirs(outdir, exist_ok=True)

    with open(params_path, "r") as f:
        params = json.load(f)

    B_tow = np.array(params["B_tow"], dtype=float)
    beta_s_clear  = float(params.get("beta_s_clear", 0.0))
    beta_s_cloudy = float(params.get("beta_s_cloudy", 0.0))
    beta_i        = float(params.get("beta_i", 0.0))
    th = params.get("thresholds", {})
    ghi_day_threshold = float(th.get("ghi_day_threshold", DEFAULT_GHI_DAY_THRESHOLD))
    clear_cut         = float(th.get("clear_cut", DEFAULT_CLEAR_CUT))

    # Load forecast inputs
    df_fut = pd.read_excel(future_input_path, sheet_name=0)
    df_fut = add_time_keys(df_fut)
    df_fut = df_fut.interpolate(method="linear", limit_direction="both")
    df_fut["Outdoor Temp (F)"] = df_fut["Outdoor Temp (F)"].fillna(75.0)
    df_fut["VAV Discharge Air Volume (ft^3 / min)"] = df_fut["VAV Discharge Air Volume (ft^3 / min)"].fillna(0.0)
    df_fut["VAV Discharge Air Temp (F)"] = df_fut["VAV Discharge Air Temp (F)"].fillna(65.0)
    df_fut["AHU Discharge Air Temp (F)"] = df_fut["AHU Discharge Air Temp (F)"].fillna(55.0)
    df_fut["GHI (W/m²)"] = df_fut["GHI (W/m²)"].fillna(0.0)
    df_fut["DNI (W/m²)"] = df_fut["DNI (W/m²)"].fillna(0.0)

    out_f = df_fut["Outdoor Temp (F)"].values
    ghi   = df_fut["GHI (W/m²)"].values
    dni   = df_fut["DNI (W/m²)"].values
    flow  = df_fut["VAV Discharge Air Volume (ft^3 / min)"].values
    how   = df_fut["how"].values
    q_mech = pd.Series(compute_q_mech_btu_per_hr(df_fut)).interpolate(limit_direction="both").fillna(0.0).values

    if "Room Temp (F)" in df_fut.columns:
        actual_room = pd.Series(df_fut["Room Temp (F)"]).interpolate(limit_direction="both").values
    else:
        actual_room = None

    C_fore        = float(params["C_Btu_per_F"])    if CUA_override is None else float(CUA_override["C_Btu_per_F"])
    UA_fore       = float(params["UA_Btu_per_hrF"]) if CUA_override is None else float(CUA_override["UA_Btu_per_hrF"])
    Q_base_BtuHr  = float(params.get("Q_base_BtuHr", 0.0))
    if CUA_override is not None and "Q_base_BtuHr" in CUA_override:
        Q_base_BtuHr = float(CUA_override["Q_base_BtuHr"])

    dates = pd.to_datetime(df_fut["Date"], errors="coerce").values
    dt_hr = infer_dt_hours(dates)

    n = len(df_fut)
    if n < 2:
        raise ValueError("Forecast input must contain at least two rows.")

    flow_mean = float(np.nanmean(flow)); flow_std = float(np.nanstd(flow) + 1e-6)
    is_day = ghi >= ghi_day_threshold
    ci = np.divide(dni, np.maximum(ghi, 1e-6))
    is_clear = (ci >= clear_cut) & is_day

    # Output trajectories (capitalized: full time series)
    T_phys_only = np.zeros(n); T_phys_only[0] = init_temp
    T_gated     = np.zeros(n); T_gated[0]     = init_temp
    T_roll_3h   = np.zeros(n); T_roll_3h[0]   = init_temp
    T_roll_6h   = np.zeros(n); T_roll_6h[0]   = init_temp
    T_roll_12h  = np.zeros(n); T_roll_12h[0]  = init_temp
    T_roll_24h  = np.zeros(n); T_roll_24h[0]  = init_temp

    gamma_arr      = np.ones(n)
    D_tow_arr      = np.zeros(n)
    D_solar_arr    = np.zeros(n)
    Z_internal_arr = np.zeros(n)
    D_total_arr    = np.zeros(n)

    # Per-horizon gating anchor (lowercase: scalars)
    gamma_anchor_3h  = 1.0
    gamma_anchor_6h  = 1.0
    gamma_anchor_12h = 1.0
    gamma_anchor_24h = 1.0

    for i in range(n - 1):
        # --- 1h gated: anchor to actual at every step when available ---
        t_curr_1h = float(actual_room[i]) if (actual_room is not None and np.isfinite(actual_room[i])) else T_gated[i]
        t_phys_next = t_curr_1h + (dt_hr / C_fore) * (
            UA_fore * (out_f[i] - t_curr_1h) + q_mech[i] + Q_base_BtuHr
        )
        T_phys_only[i + 1] = t_phys_next

        # Gaussian gating from physics-only residual at the just-finished step
        if i > 0 and actual_room is not None and sigma_phys is not None:
            resid_instant = abs(actual_room[i] - T_phys_only[i])
            gamma = compute_gating_factor(resid_instant, sigma_phys)
        else:
            gamma = 1.0
        gamma_arr[i + 1] = gamma

        # Statistical correction layer at i+1
        h_next = int(how[i + 1]) % 168
        d_tow_next      = B_tow[h_next]
        d_solar_next    = ((beta_s_clear if is_clear[i + 1] else beta_s_cloudy) * ghi[i + 1]) if is_day[i + 1] else 0.0
        z_internal_next = beta_i * ((flow[i + 1] - flow_mean) / flow_std)
        d_total_next    = d_tow_next + d_solar_next + z_internal_next
        D_tow_arr[i + 1]      = d_tow_next
        D_solar_arr[i + 1]    = d_solar_next
        Z_internal_arr[i + 1] = z_internal_next
        D_total_arr[i + 1]    = d_total_next

        T_gated[i + 1] = t_phys_next + (gamma * d_total_next)

        # === N-hour rollouts: design intent ===
        # N-hour rollout resets from actual temperature every N hours; between
        # resets, the anchor temperature (t_sN) is held fixed and only the
        # exogenous inputs (out_f, q_mech) advance. This is intentional - it is
        # NOT a propagating forecast. Errors are bounded to a single 1h step
        # rather than compounding over the horizon. The statistical correction
        # (d_tow + z_internal) is multiplied by lambda^(steps_since_anchor);
        # d_solar is applied at full strength.

        # 3h
        if i % 3 == 0:
            if actual_room is not None and np.isfinite(actual_room[i]):
                t_s3 = float(actual_room[i])
            else:
                t_s3 = T_roll_3h[i]
            gamma_anchor_3h = gamma
        elif propagating_rollout:
            t_s3 = T_roll_3h[i]
        T_roll_3h[i + 1] = (
            t_s3
            + (dt_hr / C_fore) * (UA_fore * (out_f[i] - t_s3) + q_mech[i] + Q_base_BtuHr)
            + d_solar_next
            + (gamma_anchor_3h * (decay_rate ** (i % 3))) * (d_tow_next + z_internal_next)
        )

        # 6h
        if i % 6 == 0:
            if actual_room is not None and np.isfinite(actual_room[i]):
                t_s6 = float(actual_room[i])
            else:
                t_s6 = T_roll_6h[i]
            gamma_anchor_6h = gamma
        elif propagating_rollout:
            t_s6 = T_roll_6h[i]
        T_roll_6h[i + 1] = (
            t_s6
            + (dt_hr / C_fore) * (UA_fore * (out_f[i] - t_s6) + q_mech[i] + Q_base_BtuHr)
            + d_solar_next
            + (gamma_anchor_6h * (decay_rate ** (i % 6))) * (d_tow_next + z_internal_next)
        )

        # 12h
        if i % 12 == 0:
            if actual_room is not None and np.isfinite(actual_room[i]):
                t_s12 = float(actual_room[i])
            else:
                t_s12 = T_roll_12h[i]
            gamma_anchor_12h = gamma
        elif propagating_rollout:
            t_s12 = T_roll_12h[i]
        T_roll_12h[i + 1] = (
            t_s12
            + (dt_hr / C_fore) * (UA_fore * (out_f[i] - t_s12) + q_mech[i] + Q_base_BtuHr)
            + d_solar_next
            + (gamma_anchor_12h * (decay_rate ** (i % 12))) * (d_tow_next + z_internal_next)
        )

        # 24h
        if i % 24 == 0:
            if actual_room is not None and np.isfinite(actual_room[i]):
                t_s24 = float(actual_room[i])
            else:
                t_s24 = T_roll_24h[i]
            gamma_anchor_24h = gamma
        elif propagating_rollout:
            t_s24 = T_roll_24h[i]
        T_roll_24h[i + 1] = (
            t_s24
            + (dt_hr / C_fore) * (UA_fore * (out_f[i] - t_s24) + q_mech[i] + Q_base_BtuHr)
            + d_solar_next
            + (gamma_anchor_24h * (decay_rate ** (i % 24))) * (d_tow_next + z_internal_next)
        )

    out_df_dict = {
        "Date": dates,
        "T_phys_only (F)":  T_phys_only,
        "T_full_gated (F)": T_gated,
        "T_rollout_3h (F)":  T_roll_3h,
        "T_rollout_6h (F)":  T_roll_6h,
        "T_rollout_12h (F)": T_roll_12h,
        "T_rollout_24h (F)": T_roll_24h,
        "Gating_Factor":  gamma_arr,
        "D_tow":          D_tow_arr,
        "D_solar":        D_solar_arr,
        "Z_internal":     Z_internal_arr,
        "D_total":        D_total_arr,
        "Outdoor Temp (F)": out_f,
        "VAV Discharge Air Volume (ft^3 / min)": flow,
    }
    if actual_room is not None:
        out_df_dict["Room Temp (F)"] = actual_room

    out_df = pd.DataFrame(out_df_dict)
    excel_path = os.path.join(outdir, "forecast_TOW_rollout_QUIET.xlsx")

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, sheet_name="forecast", index=False)
        if actual_room is not None:
            y = out_df["Room Temp (F)"].values
            m_rows = [
                {"Model": "1h Gated",        "MAE": float(np.nanmean(np.abs(y - T_gated)))},
                {"Model": "3h Rollout",      "MAE": float(np.nanmean(np.abs(y - T_roll_3h)))},
                {"Model": "6h Rollout",      "MAE": float(np.nanmean(np.abs(y - T_roll_6h)))},
                {"Model": "12h Rollout",     "MAE": float(np.nanmean(np.abs(y - T_roll_12h)))},
                {"Model": "24h Rollout",     "MAE": float(np.nanmean(np.abs(y - T_roll_24h)))},
                {"Model": "Pure Physics 1h", "MAE": float(np.nanmean(np.abs(y - T_phys_only)))},
            ]
            pd.DataFrame(m_rows).to_excel(writer, sheet_name="metrics", index=False)

    return excel_path
