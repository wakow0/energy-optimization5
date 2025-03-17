import pandas as pd
import numpy as np

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Start timer
start_time = time.time()

# Load datasets
processed_data_path = "processed_data.csv"
solution_a_path = "WorkingCodeVersion1_DYNAMIC_v10_6.csv"

df_processed = pd.read_csv(processed_data_path)
df_solution_a = pd.read_csv(solution_a_path)

# Convert time to datetime for alignment
df_processed['time'] = pd.to_datetime(df_processed['time'], errors='coerce')
df_solution_a['time'] = pd.to_datetime(df_solution_a['time'], errors='coerce')

# Merge datasets on time (ensure no extra NaN rows)
df_merged = df_processed.merge(df_solution_a, on='time', how='inner')

# Fill missing timestamps (if any)
df_merged['time'].fillna(method='ffill', inplace=True)

# Define features (X) and targets (Y)
X = df_merged.drop(columns=['time', 'P_import (kW)', 'P_export (kW)', 'P_bat_ch (kW)', 'P_bat_dis (kW)', 'SOC (%)'])
Y = df_merged[['P_import (kW)', 'P_export (kW)', 'P_bat_ch (kW)', 'P_bat_dis (kW)', 'SOC (%)']]

# Ensure no negative values in target variables before training
Y = np.maximum(Y, 0)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Normalize targets for LSTM training
y_scaler = MinMaxScaler()
Y_train_scaled = y_scaler.fit_transform(Y_train)
Y_test_scaled = y_scaler.transform(Y_test)

# ---- Linear Regression ----
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, Y_train)
lr_preds = np.maximum(lr_model.predict(X_test_scaled), 0)  # Clip negative values

# ---- XGBoost ----
xgb_models = {}
xgb_preds = {}

for target in tqdm(Y_train.columns, desc='Training XGBoost Models'):
    xgb_models[target] = GradientBoostingRegressor()
    xgb_models[target].fit(X_train_scaled, Y_train[target])
    xgb_preds[target] = xgb_models[target].predict(X_test_scaled)

xgb_preds = np.maximum(pd.DataFrame(xgb_preds, columns=Y_train.columns), 0)  # Clip negative values

# ---- LSTM Model ----
X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lstm_model = keras.Sequential([
    keras.layers.Input(shape=(1, X_train_scaled.shape[1])),
    keras.layers.LSTM(100, return_sequences=True),
    keras.layers.LSTM(50),
    keras.layers.Dense(5, activation='relu')  # Ensures positive outputs
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, Y_train_scaled, epochs=50, batch_size=32, verbose=1)
lstm_preds = np.maximum(y_scaler.inverse_transform(lstm_model.predict(X_test_lstm)), 0)  # Clip negative values

# ---- Performance Evaluation ----
def evaluate_model(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

evaluate_model("Linear Regression", Y_test, lr_preds)
evaluate_model("XGBoost", Y_test, xgb_preds)
evaluate_model("LSTM", Y_test, lstm_preds)

# ✅ Save full dataset
df_merged.to_csv("full_year_data.csv", index=False)

# ✅ Ensure predictions match the FULL dataset
full_time_index = df_solution_a["time"].reset_index(drop=True)

# ✅ Re-run predictions on FULL dataset (train + test)
lr_preds_full = np.maximum(lr_model.predict(scaler.transform(X)), 0)  # Clip negatives
xgb_preds_full = pd.DataFrame({target: np.maximum(xgb_models[target].predict(scaler.transform(X)), 0) for target in Y.columns})
lstm_preds_full = np.maximum(y_scaler.inverse_transform(lstm_model.predict(np.reshape(scaler.transform(X), (X.shape[0], 1, X.shape[1])))), 0)

# ✅ Create DataFrames with FULL timestamps
df_LR = pd.DataFrame(lr_preds_full, columns=Y.columns)
df_XG = pd.DataFrame(xgb_preds_full, columns=Y.columns)
df_LSTM = pd.DataFrame(lstm_preds_full, columns=Y.columns)

# ✅ Insert correct time column
df_LR.insert(0, "time", full_time_index)
df_XG.insert(0, "time", full_time_index)
df_LSTM.insert(0, "time", full_time_index)

# ✅ Save results with full timestamps
df_LR.to_csv("WorkingCodeVersion_LR_v3.csv", index=False)
df_XG.to_csv("WorkingCodeVersion_XG_v3.csv", index=False)
df_LSTM.to_csv("WorkingCodeVersion_LSTM_v3.csv", index=False)

# ✅ Plot ML results
plt.figure(figsize=(12, 6))
plt.plot(Y_test['P_import (kW)'].values, label='Actual Import', color='blue')
plt.plot(lr_preds[:, 0], label='LR Predicted Import', linestyle='dashed', color='red')
plt.plot(xgb_preds['P_import (kW)'].values, label='XGBoost Predicted Import', linestyle='dotted', color='green')
plt.xlabel("Samples")
plt.ylabel("Import Power (kW)")
plt.title("Import Prediction Comparison")
plt.legend()
plt.grid(True)

# ✅ Save plot
plt.savefig("WorkingCodeVersion_LR_v3_plot.png")
plt.show()

# End timer
end_time = time.time()
print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
