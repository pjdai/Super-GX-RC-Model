# README.md

## Purpose

This repository implements a rolling-day room temperature forecasting pipeline based on a physics-driven thermal model with schedule (TOW), solar, and internal gain terms. The core goal is to produce stable, reproducible forecasts while keeping training behavior explicit and easy to audit.

The design avoids silent parameter drift and separates training from inference as much as possible.

---

## Environment setup

### Python
- Python 3.9+ recommended

### Virtual environment

Create and activate a virtual environment:

python -m venv venv

Windows:
venv\Scripts\activate

macOS / Linux:
source venv/bin/activate

### Dependencies

Install required packages:

pip install numpy pandas matplotlib xlsxwriter openpyxl

No machine-learning frameworks are used.

---

## Repository structure

.
├── program_rolling.py  
├── airflow_control_TOW_pipeline_quiet.py  
├── tow_pipeline_quiet.py  
├── Raw Data (Excel)/  
│   ├── *_Hourly.xlsx  
└── Room_Temp_Rolling/  
    ├── output/  
    └── figures/  

---

## Input data

Two Excel files are required and are defined at the top of `program_rolling.py`:

- Training history (hourly, historical)
- Forecast-period data (hourly, future)

Both must share the same schema and include:
- Room temperature
- Airflow
- Discharge air temperature
- Schedule (TOW)
- Weather fields

---

## How to run

From the project root:

python program_rolling.py

This is the only command needed.

---

## Outputs

Always generated:
- rolling_predictions_vs_actual.xlsx
- rolling_pred_vs_actual.png

Generated when available:
- rolling_model_metrics.xlsx
- rolling_fail_log.xlsx

All outputs are written to:

Room_Temp_Rolling/output  
Room_Temp_Rolling/figures  

---

## Re-running and resets

- Re-running the script is safe and deterministic
- To force retraining for a specific day, delete:
  train_until_YYYY-MM-DD/params_TOW_QUIET.json
- To fully reset the pipeline, delete:
  Room_Temp_Rolling/output

---

## Notes

- C and UA are learned once and locked
- Forecasting never re-learns parameters
- All model variants are preserved for diagnosis
- Errors are logged rather than silently ignored
