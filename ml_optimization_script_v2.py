
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# Load and merge data
processed_data = pd.read_csv("processed_data.csv")
v6_results = pd.read_csv("heuristic_optimization_results_v6.csv")
df = pd.merge(processed_data, v6_results, on="time", how="inner")

# Feature engineering
df["time"] = pd.to_datetime(df["time"])
df["hour"] = df["time"].dt.hour
df["dayofweek"] = df["time"].dt.dayofweek
df["total_renewable"] = df["dc_ground_1500vdc_power_output"] + df["windflow_33_[500kw]_power_output"]

# Add lag feature: previous SOC
df["prev_SOC (%)"] = df["SOC (%)"].shift(1).fillna(5)  # initial SoC = 5%

# Features and targets
features = [
    "hour", "dayofweek", "ac_primary_load",
    "total_consumption_rate", "grid_sellback_rate",
    "total_renewable", "prev_SOC (%)"
]
targets = ["P_import (kW)", "P_export (kW)", "P_bat_ch (kW)", "P_bat_dis (kW)", "SOC (%)"]

X = df[features]
Y = df[targets]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train XGBoost with deeper trees
model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0))
model.fit(X_scaled, Y)
predictions = model.predict(X_scaled)

# Format predictions
pred_df = pd.DataFrame(predictions, columns=targets)
pred_df["time"] = df["time"]
pred_df = pred_df[["time"] + targets]

# Post-processing: clip and apply logic
pred_df["P_import (kW)"] = pred_df["P_import (kW)"].clip(lower=0)
pred_df["P_export (kW)"] = pred_df["P_export (kW)"].clip(lower=0)
pred_df["P_bat_ch (kW)"] = pred_df["P_bat_ch (kW)"].clip(lower=0)
pred_df["P_bat_dis (kW)"] = pred_df["P_bat_dis (kW)"].clip(lower=0)
pred_df["SOC (%)"] = pred_df["SOC (%)"].clip(lower=5, upper=100)

# Logical corrections: SoC-based export limits
for i in range(len(pred_df)):
    soc = pred_df.loc[i, "SOC (%)"]
    if soc >= 100:
        pred_df.loc[i, "P_bat_ch (kW)"] = 0
    if soc <= 5:
        pred_df.loc[i, "P_bat_dis (kW)"] = 0
    if soc < 20:
        pred_df.loc[i, "P_export (kW)"] = 0  # prevent export if SoC too low

# Export result
pred_df.to_csv("ml_solution_predictions_v2.csv", index=False)
print("✅ ML-guided optimization v2 complete. Output saved to 'ml_guided_solution_v2.csv'")
