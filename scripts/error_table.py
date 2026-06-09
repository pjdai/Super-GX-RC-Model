import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load Excel file (anchored to the repo root; this file lives in scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_excel(os.path.join(REPO_ROOT, "Room_Temp_Rolling", "output", "rolling_predictions_vs_actual.xlsx"))

# Rename for clarity
df.rename(columns={
    'T_room_pred (F)': 'predicted',
    'Room Temp (F)': 'actual'
}, inplace=True)

# Calculate metrics
mae = mean_absolute_error(df['actual'], df['predicted'])
rmse = np.sqrt(mean_squared_error(df['actual'], df['predicted']))
r2 = r2_score(df['actual'], df['predicted'])

# Print results as a single-row table
results_df = pd.DataFrame([["YourModelNameHere", mae, rmse, r2]], columns=["Model", "MAE", "RMSE", "R2"])
print(results_df)
