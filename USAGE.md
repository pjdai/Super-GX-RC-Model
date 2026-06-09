# Usage

## Overview

This code implement a rolling-day room temperature forecasting pipeline using Hybrid Grey-Box thermal model with schedule (TOW), solar, and internal gain. Thermal parameter (C, UA, Q_base) are hunt globally once from historical data to find absolute best baseline, and then fixed across all rolling forecast to avoid parameter drift.

The pipeline predict zone temperature over multiple horizon (1h, 3h, 6h, 12h, 24h) and employ Gaussian Soft Gating mechanism to dynamic manage sensor anomaly. It also use exponential decaying gating for long term rollout to smooth error. Only one script is execute directly. The other provide modeling logic, shared utility, and post-analysis.

---

## Programs and responsibilities

### `program_rolling.py` (entry point)

This is the only script to run for main pipeline.

Responsibilities:
- Execute Phase 1: Global Parameter Hunting across historical database
- Define rolling training window and forecast day
- Lock C, UA, and Q_base after find the champion training window (MAE < 1°F)
- Run daily multi-scale forecast
- Aggregate prediction, dynamic handle data gap, and generate plot

High-level workflow:
1. Load historical training data and future-period data
2. Phase 1 (Hunting): Slide a 30-day window across database to find best physical C/UA/Q_base baseline.
3. Phase 2 (Rolling Forecast): For each forecast day:
   - Build trailing 30-day window (if needed)
   - Load and enforce locked champion C/UA/Q_base
   - Calculate dynamic historical error ($\sigma$) for soft gating
   - Run 1h (Gated) and 3h/6h/12h/24h (Decaying Gated Rollout) forecast using observed initial temperature
   - Collect prediction and gating factor
4. Write combined output and dual-axis plot

---

### `airflow_control_TOW_pipeline_quiet.py` (core modeling logic)

Not execute directly.

#### `run_pipeline_quiet(...)`

Purpose:
- Estimate thermal parameter and regression coefficient from a given training window

What it do:
- Call quiet-hour extractor for pure C, UA, and Q_base
- Decouple signal: Extract Time-of-Week (TOW) schedule profile
- Fit solar coefficient ($\beta_{clear}$, $\beta_{cloudy}$) using pure residual ($e_{phys} - D_{tow}$)
- Evaluate multiple model variant for ablation

Outputs:
- `params_TOW_QUIET.json` (Store C, UA, Q_base, $\beta_{solar}$, $D_{tow}$)
- `predictions_and_metrics_TOW_QUIET.xlsx`

This function is call during Hunting phase or when force retraining.

---

#### `run_forecast_quiet(...)`

Purpose:
- Generate day-ahead multi-scale forecast using fixed parameter

What it do:
- Load frozen C/UA/Q_base and learned coefficient
- Calculate instant Solar Temp Gain ($\beta \cdot GHI$)
- Apply Gaussian Soft Gating ($\gamma$) for 1h prediction to block sensor anomaly
- Simulate temperature forward recursive for 3h, 6h, 12h, and 24h rollout using Physics, Solar, and decaying statistical correction
- Use last observed room temperature as initial condition for each reset cycle

Output:
- `forecast_TOW_rollout_QUIET.xlsx` (contain all model variant)

This function is call once per forecast day.

---

### `tow_pipeline_quiet.py` (shared utilities)

Not execute directly.

Contains:
- Strict quiet-hour filter (midnight to morning) to avoid solar thermal lag
- Pure physical parameter extraction (C and UA) using IQR outlier rejection
- Extract constant internal equipment heat load ($Q_{base}$) using Weighted Least Squares (WLS) with Intercept to stop negative UA crash
- Shared preprocess and alignment helper

---

### `error_analysis_on_hour.py` (evaluation script)

Execute independent after `program_rolling.py` finish.

Contains:
- Aggregate error parse for entire forecast period
- Calculate MAE, RMSE, Max Error, and <1°F Accuracy for all prediction scale
- Generate `_summary_report.csv`

---

## Data flow

program_rolling.py (Main Orchestrator)
|
|-- Phase 1: Parameter Hunting --> run_pipeline_quiet()
|                                |--> tow_pipeline_quiet.py
|                                |--> params_TOW_QUIET.json (Lock Champion C, UA, Q_base)
|
|-- Phase 2: Daily Rolling ------> run_forecast_quiet()
|                                |--> forecast_TOW_rollout_QUIET.xlsx
|
--> rolling_predictions_vs_actual.xlsx
--> final_long_term_comparison.png
|
|-- Post-Analysis ---------------> error_analysis_on_hour.py
                                 |--> _summary_report.csv

Daily output are aggregate by `program_rolling.py`.

---

## Outputs

### Always produced
- `rolling_predictions_vs_actual.xlsx`  
  Hourly prediction across all forecast day. Include variant: 1h Gated, 3h/6h/12h/24h Decaying Gated Rollout, `Gating_Factor` ($\gamma$), and `D_solar`.

- `final_long_term_comparison.png`  
  Dual-y-axis plot overlay Actual Room Temp with Gated 1h and multi-scale rollout, plus background area for Solar Temp Gain.

### Produced when available / upon evaluation
- `_summary_report.csv`  
  Generate by run `error_analysis_on_hour.py`. Summarize MAE and RMSE for all horizon.

- `params_TOW_QUIET.json`  
  Save in individual day folder, enforce locked C/UA/Q_base and store daily statistical coefficient.

---

## Fixed vs. rolling components

Fixed across all day (Post-Hunting):
- Thermal capacitance (C)
- Envelope conductance (UA)
- Base internal heat load ($Q_{base}$)

Rolling / Dynamic:
- Forecast day weather and input (GHI, VAV CFM)
- Initial room temperature (Reset every 3h/6h/12h/24h)
- Gating Factor ($\gamma$)
- Error metric

C, UA, and Q_base are intentional frozen as physical backbone to prevent drift from noisy short window.

---

## Differences from the previous version

Changed or added:
- **Global Parameter Hunting:** Search entire dataset for best baseline (MAE < 1°F) instead of just immediate previous window.
- **Base Heat Load Extraction:** Physical regression now include intercept to capture 24/7 internal equipment heat ($Q_{base}$). This stop negative UA crash in room with heavy equipment or severe thermal lag.
- **Multi-Scale Rollouts:** Added 3h, 6h, 12h, and 24h recursive blind prediction.
- **Solar Injection:** Solar heat ($\beta \cdot GHI$) is physical inject into rollout equation.
- **Gaussian Soft Gating (Scheme B):** Dynamic trust-weighting ($\gamma$) protect 1h forecast from sensor spike.
- **Decaying Soft Gating:** Long term rollout use exponential decay gating to apply TOW and internal heat correction. This pull prediction back to normal without break physics.
- **Robust Imputation:** Bi-directional linear interpolation to prevent simulation crash from missing data (NaN).

Not implemented (by design):
- Periodic re-estimation of C, UA, Q_base

---

## Forcing retraining

Delete a day `params_TOW_QUIET.json` or modify Global Parameter Hunting threshold in `program_rolling.py` to force pipeline to find new baseline.  
Otherwise, pipeline will reuse exist champion parameter and only run forecast.