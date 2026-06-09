#!/usr/bin/env python3
"""Learn room physical parameters (C, UA, Q_base) from low-occupancy hours, with recency weighting.

Quiet hours used for both UA/Q_base steady-state regression and C transient fit:
  - Weekdays (Mon-Fri): 00:00-06:00
  - Weekends (Sat-Sun): 00:00-08:00

Estimates internally in SI (C [J/K], UA [W/K], Q_base [W]); rc_forecast_pipeline converts to imperial.
"""
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd

# === Physical constants (SI) ===
RHO_AIR = 1.20              # kg / m^3 (air density at ~20 C)
CP_AIR  = 1006.0            # J / (kg * K) (specific heat of air)
CFM_TO_M3S = 0.00047194745  # m^3/s per CFM

# === Filter thresholds ===
DEFAULT_MAX_GAP_HOURS = 1.5      # drop rows whose previous timestamp is more than this far
MIN_DTDT_F_PER_HR = 0.05         # below this dT/dt the slope-fit signal is too weak (deg-F / hr)
MIN_TOUT_MINUS_TR_K_STEADY = 0.1 # steady-state slice still needs a non-trivial driver (K)


def F_to_K(f):
    """Convert Fahrenheit to Kelvin (scalar or array)."""
    return (np.asarray(f, dtype=float) - 32.0) * (5.0 / 9.0) + 273.15


@dataclass
class QuietConfig:
    """Tunable knobs for the quiet-hours learner."""
    dt_seconds: Optional[float] = None         # sampling interval [s]; inferred if None
    min_quiet_samples: int = 40                # min quiet-hour rows to fit C
    min_ss_samples: int = 30                   # min near-steady rows to fit UA / Q_base
    dTdt_abs_max_ss: float = 0.5 / 3600.0      # |dT/dt| upper bound for "near steady" [K/s]
    recency_weight_gain: float = 2.0           # newest-sample weight (1.0 = uniform)
    max_gap_hours: float = DEFAULT_MAX_GAP_HOURS
    min_abs_Tout_minus_Tr_F: float = 2.0       # required envelope driver [F]


def _resolve_cols(df: pd.DataFrame) -> Dict[str, str]:
    """Map internal column keys to whichever raw column name the data file uses."""
    lowers = {c.lower(): c for c in df.columns}
    def pick(names):
        for n in names:
            if n.lower() in lowers:
                return lowers[n.lower()]
        raise ValueError(f"Missing required column among: {names}")
    def pick_optional(names):
        for n in names:
            if n.lower() in lowers:
                return lowers[n.lower()]
        return None
    out = {
        "ts":          pick([
            "Date", "Timestamp", "Datetime", "DateTime", "Time", "Time Stamp", "TimeStamp",
        ]),
        "t_room_f":    pick([
            "Room Temp (F)", "Room Temp(F)", "Room Temp", "T_room", "T Room (F)",
            "Zone Temp (F)", "Zone Air Temp (F)", "Room Air Temp (F)", "RAT (F)",
            "/FS5/ZONE_A/CTL TEMP/presentValue",
            "/FS5/ZONE_B/CTL TEMP/presentValue",
        ]),
        "t_out_f":     pick([
            "Outdoor Temp (F)", "Outdoor Temp", "T_out",
            "OAT (F)", "Outside Air Temp (F)", "OA Temp (F)", "Outdoor Air Temp (F)",
            "temperature_2m",
        ]),
        "t_supply_f":  pick([
            "VAV Discharge Air Temp (F)", "Supply Temp (F)", "VAV DAT (F)",
            "VAV SAT (F)", "VAV Supply Temp (F)", "VAV Discharge Temp (F)",
            "/FS5/ZONE_A/AI 3/presentValue",
            "/FS5/ZONE_B/AI 3/presentValue",
        ]),
        "t_ahu_f":     pick([
            "AHU Discharge Air Temp (F)", "AHU DAT (F)", "AHU Supply Temp (F)",
            "AHU SAT (F)", "AHU Supply Air Temp (F)",
            "/FS5/BLDG/AHU_SAT/presentValue",
        ]),
        "cfm":         pick([
            "VAV Discharge Air Volume (ft^3 / min)", "VAV CFM", "CFM",
            "Supply Air Flow (CFM)", "Air Flow (CFM)", "Discharge Air Flow (CFM)",
            "Airflow (CFM)", "VAV Discharge Air Volume (cfm)",
            "/FS5/ZONE_A/AIR VOLUME/presentValue",
            "/FS5/ZONE_B/AIR VOLUME/presentValue",
        ]),
    }
    # Optional solar columns - omitted if absent so data files without solar still work
    ghi = pick_optional(["GHI", "shortwave_radiation"])
    dni = pick_optional(["DNI", "direct_normal_irradiance"])
    dhi = pick_optional(["DHI", "diffuse_radiation"])
    if ghi is not None: out["ghi"] = ghi
    if dni is not None: out["dni"] = dni
    if dhi is not None: out["dhi"] = dhi
    return out


def _finite_diff(y: np.ndarray, dt_s: float) -> np.ndarray:
    """Backward finite-difference derivative; [0] copied from [1] to avoid a head NaN."""
    y = np.asarray(y, dtype=float)
    d = np.full_like(y, np.nan)
    if y.size >= 2:
        d[1:] = (y[1:] - y[:-1]) / dt_s
        d[0]  = d[1]
    return d


def _iqr_mask(x: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Boolean inlier mask via the 1.5*IQR Tukey rule (k = IQR multiplier)."""
    x = np.asarray(x, dtype=float)
    q1 = np.nanpercentile(x, 25.0)
    q3 = np.nanpercentile(x, 75.0)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return np.isfinite(x)
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (x >= lo) & (x <= hi) & np.isfinite(x)


def _recency_weights(t_ns: np.ndarray, gain: float) -> np.ndarray:
    """Linearly ramp WLS weights from 1.0 (oldest) to `gain` (newest)."""
    t = np.asarray(t_ns, dtype=np.int64)
    n = t.size
    if n == 0 or gain <= 1.0:
        return np.ones(n, dtype=float)
    order = np.argsort(t)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, n)
    return 1.0 + (gain - 1.0) * ranks


class TOWQuietLearner:
    """Estimate (C, UA, Q_base) for one room from low-occupancy snapshots.

    Step 1 (steady slice): fit Q_mech = -UA*(T_out - T_room) - Q_base via WLS with intercept.
    Step 2 (transient quiet slice): fit dT/dt = (1/C)*(UA*(T_out-T_room) + Q_mech + Q_base) through origin.
    """

    def __init__(self, cfg: Optional[QuietConfig] = None):
        self.cfg = cfg or QuietConfig()
        self.learned_ = None
        self.dt_s_ = None

    def _prep(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Resolve columns, drop large-gap rows, infer dt, and add SI feature columns."""
        cols = _resolve_cols(df_raw)
        d = df_raw.rename(columns={v: k for k, v in cols.items()}).copy()
        d["ts"] = pd.to_datetime(d["ts"], errors="coerce")
        d = d.sort_values("ts").reset_index(drop=True)

        # convert open-meteo °C to °F if needed
        if cols.get("t_out_f") == "temperature_2m":
            d["t_out_f"] = d["t_out_f"] * 9 / 5 + 32

        # Drop rows whose gap to the previous timestamp exceeds max_gap_hours
        before = len(d)
        gap_hours = d["ts"].diff().dt.total_seconds() / 3600.0
        max_gap = getattr(self.cfg, "max_gap_hours", DEFAULT_MAX_GAP_HOURS)
        keep_mask = (gap_hours.isna()) | (gap_hours <= max_gap)
        d = d.loc[keep_mask].reset_index(drop=True)
        after = len(d)
        print(f"[TOWQuietLearner] Dropped {before - after} rows due to large time gaps.")

        # Sampling interval
        if self.cfg.dt_seconds is None:
            if len(d) >= 2:
                dt = np.median(
                    np.diff(d["ts"].values).astype("timedelta64[s]").astype(float)
                )
                dt = max(float(dt), 1.0)
            else:
                dt = 3600.0
        else:
            dt = float(self.cfg.dt_seconds)
        self.dt_s_ = dt

        # SI feature columns
        d["T_room_K"] = F_to_K(d["t_room_f"])
        d["T_out_K"]  = F_to_K(d["t_out_f"])
        d["T_sup_K"]  = F_to_K(d["t_supply_f"])
        d["vav_m3s"]  = d["cfm"].astype(float) * CFM_TO_M3S
        d["m_dot"]    = RHO_AIR * d["vav_m3s"]
        d["Q_mech_W"] = d["m_dot"] * CP_AIR * (d["T_sup_K"] - d["T_room_K"])
        d["dTdt_Ks"]  = _finite_diff(d["T_room_K"], self.dt_s_)
        d["Tout_minus_Tr"] = d["T_out_K"] - d["T_room_K"]

        d["dow"] = d["ts"].dt.dayofweek    # 0=Mon
        d["hour"] = d["ts"].dt.hour
        return d

    def _quiet_slice(self, d: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Restrict to nighttime hours (00:00-06:00 weekday, 00:00-08:00 weekend) for the C fit."""
        # Strict quiet hours - avoids west-facing solar thermal lag in evening
        wk = (d["dow"] <= 4) & (d["hour"] >= 0) & (d["hour"] < 6)   # Mon-Fri 00:00-06:00
        we = (d["dow"] >= 5) & (d["hour"] >= 0) & (d["hour"] < 8)   # Sat-Sun 00:00-08:00
        q = d.loc[wk | we].copy()
        if verbose:
            print(f"[quiet_slice] window ts range: {d['ts'].min()} -> {d['ts'].max()}")
            print(f"[quiet_slice] after strict time filter (00:00-06:00 wk / 00:00-08:00 wkn): {len(q)} rows")

        # Drop rows with NaN/Inf in any regression axis
        q = q[
            np.isfinite(q["dTdt_Ks"]) &
            np.isfinite(q["Tout_minus_Tr"]) &
            np.isfinite(q["Q_mech_W"])
        ]
        if verbose:
            print(f"[quiet_slice] after finite filter: {len(q)} rows")

        # Require sufficient envelope driver (|T_out - T_room| above threshold, in Kelvin)
        min_k = self.cfg.min_abs_Tout_minus_Tr_F * (5.0 / 9.0)
        q = q.loc[np.abs(q["Tout_minus_Tr"].values) >= min_k]
        if verbose:
            print(f"[quiet_slice] after |Tout-Troom| >= {self.cfg.min_abs_Tout_minus_Tr_F}F: {len(q)} rows")

        # Require enough dT/dt to identify slope
        min_dTdt_Ks = (MIN_DTDT_F_PER_HR * (5.0 / 9.0)) / 3600.0
        q = q.loc[np.abs(q["dTdt_Ks"].values) >= min_dTdt_Ks]
        if verbose:
            print(f"[quiet_slice] after |dT/dt| >= {MIN_DTDT_F_PER_HR}F/hr: {len(q)} rows")

        # IQR outlier filter
        if len(q) > 0:
            mask = _iqr_mask(q["dTdt_Ks"].values) & _iqr_mask(q["Tout_minus_Tr"].values)
            q = q.loc[mask]
            if verbose:
                print(f"[quiet_slice] after IQR outlier prune: {len(q)} rows")

        return q

    def _steady_hvac_slice(self, d: pd.DataFrame) -> pd.DataFrame:
        """Restrict to near-steady samples (22:00-06:00) for UA / Q_base regression.

        Deliberately a wider window than _quiet_slice; do not tighten without a physics decision.
        """
        night_mask = (d["hour"] >= 22) | (d["hour"] < 6)
        ss = d[night_mask].copy()
        ss = ss[np.isfinite(ss["Q_mech_W"]) & np.isfinite(ss["Tout_minus_Tr"])]
        ss = ss.loc[ss["dTdt_Ks"].abs() <= self.cfg.dTdt_abs_max_ss]
        ss = ss.loc[np.abs(ss["Tout_minus_Tr"].values) > MIN_TOUT_MINUS_TR_K_STEADY]
        return ss

    @staticmethod
    def _wls_slope_through_origin(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """Weighted least-squares slope of y = m*x (no intercept); NaN if degenerate."""
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); w = np.asarray(w, dtype=float)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        x = x[m]; y = y[m]; w = w[m]
        if x.size == 0:
            return np.nan
        num = np.sum(w * x * y)
        den = np.sum(w * x * x)
        if den == 0:
            return np.nan
        return float(num / den)

    @staticmethod
    def _wls_with_intercept(x: np.ndarray, y: np.ndarray, w: np.ndarray):
        """Weighted least-squares fit y = m*x + b; (NaN, NaN) if degenerate."""
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); w = np.asarray(w, dtype=float)
        m_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        x = x[m_mask]; y = y[m_mask]; w = w[m_mask]

        if x.size < 2 or np.sum(w) == 0:
            return np.nan, np.nan

        W = np.sum(w)
        x_bar = np.sum(w * x) / W
        y_bar = np.sum(w * y) / W

        num = np.sum(w * (x - x_bar) * (y - y_bar))
        den = np.sum(w * (x - x_bar) ** 2)
        if den == 0:
            return np.nan, np.nan

        m = float(num / den)
        b = float(y_bar - m * x_bar)
        return m, b

    def learn_C_UA(self, df_raw: pd.DataFrame, verbose_quiet: bool = False) -> Dict[str, float]:
        """Estimate (C, UA, Q_base, tau_hours) from a training dataframe."""
        d = self._prep(df_raw)

        # --- STEP 1: Estimate UA and Q_base from near-steady snapshots ---
        ss = self._steady_hvac_slice(d)
        if len(ss) < self.cfg.min_ss_samples:
            raise RuntimeError(f"Not enough steady-state snapshots ({len(ss)}) to estimate UA.")

        x_ss = ss["Tout_minus_Tr"].values
        y_ss = ss["Q_mech_W"].values
        w_ss = _recency_weights(ss["ts"].astype("int64").values, self.cfg.recency_weight_gain)

        # IQR prune
        good = _iqr_mask(x_ss) & _iqr_mask(y_ss)
        x_ss, y_ss, w_ss = x_ss[good], y_ss[good], w_ss[good]

        # In near-steady state: 0 = UA*(T_out - T_room) + Q_mech + Q_base
        # => Q_mech = -UA*(T_out - T_room) - Q_base, so slope = -UA, intercept = -Q_base
        m_ss, b_ss = self._wls_with_intercept(x_ss, y_ss, w_ss)
        if np.isnan(m_ss):
            raise RuntimeError("WLS regression for UA/Q_base failed.")

        UA = -m_ss
        Q_base_W = -b_ss
        if UA <= 0:
            raise RuntimeError(
                f"Estimated UA still invalid: {UA:.2f} (Base Load was {Q_base_W:.2f}W). "
                f"Severe physical mismatch."
            )

        # --- STEP 2: Estimate C from transient quiet slice ---
        q = self._quiet_slice(d, verbose=verbose_quiet)
        if len(q) < self.cfg.min_quiet_samples:
            raise RuntimeError(f"Not enough quiet samples ({len(q)}) to estimate C.")

        # C * dT/dt = UA*(T_out - T_room) + Q_mech + Q_base
        Q_net = UA * q["Tout_minus_Tr"].values + q["Q_mech_W"].values + Q_base_W
        dTdt = q["dTdt_Ks"].values
        w_q = _recency_weights(q["ts"].astype("int64").values, self.cfg.recency_weight_gain)

        # Fit dTdt = (1/C) * Q_net through origin
        inv_C = self._wls_slope_through_origin(Q_net, dTdt, w_q)
        if not np.isfinite(inv_C) or inv_C <= 0:
            raise RuntimeError(f"Estimated 1/C invalid: {inv_C}")
        C = 1.0 / inv_C

        self.learned_ = {
            "C_J_per_K":  float(C),
            "UA_W_per_K": float(UA),
            "Q_base_W":   float(Q_base_W),
            "tau_hours":  float(C / UA / 3600.0),
        }
        return self.learned_
