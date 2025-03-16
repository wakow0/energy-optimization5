import pandas as pd
import numpy as np
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
df_processed['time'] = pd.to_datetime(df_processed['time'])
df_solution_a['time'] = pd.to_datetime(df_solution_a['time'])

# Merge datasets on time
df_merged = df_processed.merge(df_solution_a, on='time', how='inner')

# Define features (X) and targets (Y)
X = df_merged.drop(columns=['time', 'P_import (kW)', 'P_export (kW)', 'P_bat_ch (kW)', 'P_bat_dis (kW)', 'SOC (%)'])
Y = df_merged[['P_import (kW)', 'P_export (kW)', 'P_bat_ch (kW)', 'P_bat_dis (kW)', 'SOC (%)']]

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
lr_preds = lr_model.predict(X_test_scaled)

# ---- XGBoost ----
xgb_models = {}
xgb_preds = {}

for target in tqdm(Y_train.columns, desc='Training XGBoost Models'):
    xgb_models[target] = GradientBoostingRegressor()
    xgb_models[target].fit(X_train_scaled, Y_train[target])
    xgb_preds[target] = xgb_models[target].predict(X_test_scaled)

# Convert predictions to a DataFrame
xgb_preds = pd.DataFrame(xgb_preds, columns=Y_train.columns)

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
lstm_preds = lstm_model.predict(X_test_lstm)

# Convert LSTM predictions back to original scale
lstm_preds = y_scaler.inverse_transform(lstm_preds)

# ---- Performance Evaluation ----
def evaluate_model(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

evaluate_model("Linear Regression", Y_test, lr_preds)
evaluate_model("XGBoost", Y_test, xgb_preds)
evaluate_model("LSTM", Y_test, lstm_preds)

# Ensure no negative values in predictions
lr_preds = np.maximum(lr_preds, 0)
xgb_preds = np.maximum(xgb_preds, 0)
lstm_preds = np.maximum(lstm_preds, 0)

# ---- Plot Results ----
plt.figure(figsize=(10, 5))
plt.plot(Y_test['P_import (kW)'].values, label='Actual Import', color='blue')
plt.plot(lr_preds[:, 0], label='LR Predicted Import', linestyle='dashed', color='red')
plt.plot(xgb_preds['P_import (kW)'].values, label='XGBoost Predicted Import', linestyle='dotted', color='green')
plt.xlabel("Samples")
plt.ylabel("Import Power (kW)")
plt.title("Import Prediction Comparison")
plt.legend()
plt.show()


# Sample DataFrame (Replace with actual model results)
time_steps = list(range(24))  # Assuming 24 hours data

# Replace with actual model outputs
df_LR = pd.DataFrame({
    "time": time_steps,
    "import": [0.2] * 24, 
    "export": [0.1] * 24, 
    "charge": [0.3] * 24, 
    "discharge": [0.15] * 24, 
    "soc": [50 + i for i in range(24)]
})

df_XG = df_LR.copy()  # Replace with actual XGBoost model outputs
df_LSTM = df_LR.copy()  # Replace with actual LSTM model outputs


print(df_merged.shape)  # Should be ~17,500 rows if it's 30-min intervals for 1 year
print(df_merged['time'].min(), df_merged['time'].max())  # Check if all dates are present
df_merged = df_processed.merge(df_solution_a, on='time', how='outer')
df_merged.to_csv("full_year_data.csv", index=False)





# ✅ Save all three solutions
df_LR.to_csv("WorkingCodeVersion_LR_v1.csv", index=False)
df_XG.to_csv("WorkingCodeVersion_XG_v1.csv", index=False)
df_LSTM.to_csv("WorkingCodeVersion_LSTM_v1.csv", index=False)

print("✅ Results saved as:")
print("- WorkingCodeVersion_LR_v1.csv")
print("- WorkingCodeVersion_XG_v1.csv")
print("- WorkingCodeVersion_LSTM_v1.csv")

# ✅ Plot all 5 outputs
plt.figure(figsize=(12, 6))

plt.plot(df_LR['time'], df_LR['import'], label="Grid Import", linestyle="--", color="blue")
plt.plot(df_LR['time'], df_LR['export'], label="Grid Export", linestyle="--", color="orange")
plt.plot(df_LR['time'], df_LR['charge'], label="Battery Charge", linestyle="-", color="green")
plt.plot(df_LR['time'], df_LR['discharge'], label="Battery Discharge", linestyle="-", color="red")
plt.plot(df_LR['time'], df_LR['soc'], label="State of Charge (SOC)", linestyle="-", color="purple")

plt.xlabel("Time (Hours)")
plt.ylabel("Power (kW) / SOC (%)")
plt.title("Energy Optimization Results - Linear Regression")
plt.legend()
plt.grid(True)

# ✅ Save plot
plt.savefig("WorkingCodeVersion_LR_v1_plot.png")
plt.show()


# End timer and display execution time
end_time = time.time()
print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
