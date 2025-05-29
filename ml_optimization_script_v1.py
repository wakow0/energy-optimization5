
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# Load your data
processed_data = pd.read_csv("processed_data.csv")
v6_results = pd.read_csv("heuristic_optimization_results_v6.csv")

# Merge data on time
merged_df = pd.merge(processed_data, v6_results, on="time", how="inner")
merged_df["time"] = pd.to_datetime(merged_df["time"])
merged_df["hour"] = merged_df["time"].dt.hour
merged_df["dayofweek"] = merged_df["time"].dt.dayofweek
merged_df["total_renewable"] = (
    merged_df["dc_ground_1500vdc_power_output"] + merged_df["windflow_33_[500kw]_power_output"]
)

# Define features and targets
features = [
    "hour", "dayofweek", "ac_primary_load",
    "total_consumption_rate", "grid_sellback_rate", "total_renewable"
]
targets = ["P_import (kW)", "P_export (kW)", "P_bat_ch (kW)", "P_bat_dis (kW)", "SOC (%)"]

X = merged_df[features]
Y = merged_df[targets]

# Split and scale
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression
lr_model = MultiOutputRegressor(LinearRegression())
lr_model.fit(X_train_scaled, Y_train)
lr_preds = lr_model.predict(X_test_scaled)

# Train XGBoost with fast settings
xgb_model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=10, random_state=42, verbosity=0))
xgb_model.fit(X_train_scaled, Y_train)
xgb_preds = xgb_model.predict(X_test_scaled)

# Evaluate models
def evaluate_predictions(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))
    print(f"\n{label} Model Performance:")
    for i, col in enumerate(Y.columns):
        print(f"{col}: MAE = {mae[i]:.2f}, RMSE = {rmse[i]:.2f}")

evaluate_predictions(Y_test, lr_preds, "Linear Regression")
evaluate_predictions(Y_test, xgb_preds, "XGBoost (n_estimators=10)")
