import pandas as pd
import numpy as np
import os

# Paths anchored to the repo root (this file lives in scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(REPO_ROOT, "Room_Temp_Rolling", "output", "rolling_predictions_vs_actual.xlsx")

def run_analysis():
    if not os.path.exists(FILE_PATH):
        print("Result file not found!"); return
    
    df = pd.read_excel(FILE_PATH)
    
    # 1. Dynamically get existing columns to prevent KeyError
    target_cols = [
        'Room Temp (F)', 'T_full_gated (F)', 'T_rollout_3h (F)', 
        'T_rollout_6h (F)', 'T_rollout_12h (F)', 'T_rollout_24h (F)', 'T_phys_only (F)'
    ]
    available_cols = [col for col in target_cols if col in df.columns]
    
    # 2. Drop NaNs to ensure fair comparison
    df = df.dropna(subset=available_cols)
    
    actual = df['Room Temp (F)']
    
    # 3. Update model naming: Emphasize the new Decaying Gating mechanism
    models_to_check = {
        "1h Gated Rollout (Decaying Disturbances)": 'T_full_gated (F)',
        "3h Gated Rollout (Decaying Disturbances)": 'T_rollout_3h (F)',
        "6h Gated Rollout (Decaying Disturbances)": 'T_rollout_6h (F)',
        "12h Gated Rollout (Decaying Disturbances)": 'T_rollout_12h (F)',
        "24h Gated Rollout (Decaying Disturbances)": 'T_rollout_24h (F)',
        "1h Phys Base": 'T_phys_only (F)'  # Pure Physics baseline
    }
    
    stats = []
    for name, col_name in models_to_check.items():
        if col_name not in df.columns:
            continue # Skip if column doesn't exist
            
        pred = df[col_name]
        error = pred - actual
        
        # Calculate metrics
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        max_err = np.max(np.abs(error))
        # Calculate percentage of errors within 1F
        within_1f = (np.abs(error) < 1.0).mean() * 100
        
        stats.append({
            "Model": name,
            "MAE (F)": round(mae, 3),
            "RMSE (F)": round(rmse, 3),
            "Max Error (F)": round(max_err, 3),
            "Accuracy < 1F (%)": round(within_1f, 1)
        })
    
    analysis_df = pd.DataFrame(stats)
    print("\n=== Rolling Forecast Error Report (Decaying Gating & Solar Integrated) ===")
    print(analysis_df.to_string(index=False))
    
    # Save report
    report_path = FILE_PATH.replace(".xlsx", "_summary_report.csv")
    analysis_df.to_csv(report_path, index=False)
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    run_analysis()
