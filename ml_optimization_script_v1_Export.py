import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

# Load and merge data
processed_data = pd.read_csv("processed_data.csv")
v6_results = pd.read_csv("heuristic_optimization_results_v6.csv")
df = pd.merge(processed_data, v6_results, on="time", how="inner")

# Feature engineering
df["time"] = pd.to_datetime(df["time"])
df["hour"] = df["time"].dt.hour
df["dayofweek"] = df["time"].dt.dayofweek
df["total_renewable"] = df["dc_ground_1500vdc_power_output"] + df["windflow_33_[500kw]_power_output"]

# Features and target structure
features = ["hour", "dayofweek", "ac_primary_load", "total_consumption_rate", "grid_sellback_rate", "total_renewable"]
targets = ["P_import (kW)", "P_export (kW)", "P_bat_ch (kW)", "P_bat_dis (kW)", "SOC (%)"]

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Re-train model (or load it if you saved earlier)
Y = df[targets]
xgb_model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=10, random_state=42, verbosity=0))
xgb_model.fit(X_scaled, Y)
predictions = xgb_model.predict(X_scaled)

# Prepare DataFrame
predicted_df = pd.DataFrame(predictions, columns=targets)
predicted_df["time"] = df["time"]
predicted_df = predicted_df[["time"] + targets]

# Export to CSV
predicted_df.to_csv("ml_solution_predictions_v1.csv", index=False)
print("✅ ML predictions saved to ml_solution_predictions_v1.csv")
