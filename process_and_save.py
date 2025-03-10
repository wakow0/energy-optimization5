import pandas as pd

# ✅ Load processed data
file_path = "processed_data.csv"
df = pd.read_csv(file_path, low_memory=False)

# 🔹 Debug: Print available columns
print("\n✅ Step 1: Available Columns in DataFrame")
print(df.columns.tolist())  # Print all column names

# ✅ Ensure the Time column is correctly formatted
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")  # Auto-convert if in string format

# ✅ Correct column mapping based on actual column names in the dataset
columns_mapping = {
    "P_import (kW)": "grid_purchases",
    "P_export (kW)": "grid_sales",
    "P_bat_ch (kW)": "wattstor_m5_0.5c_september_charge_power",
    "P_bat_dis (kW)": "wattstor_m5_0.5c_september_discharge_power",
    "SOC (%)": "wattstor_m5_0.5c_september_state_of_charge",
}

# ✅ Assign values only if columns exist
for new_col, original_col in columns_mapping.items():
    if original_col in df.columns:
        df[new_col] = df[original_col]
    else:
        print(f"❌ WARNING: Column '{original_col}' is missing from the dataset! {new_col} will not be included.")

# ✅ Remove missing columns from selection
selected_columns = ["time"] + [col for col in columns_mapping.keys() if col in df.columns]

# 🔹 Debug: Check computed columns
print("\n✅ Step 2: Computed Columns Added (First 5 rows)")
print(df[selected_columns].head())

# ✅ Save the modified data
output_file = "processed_data_fixed.csv"
df[selected_columns].to_csv(output_file, index=False, encoding="utf-8")
print(f"\n✅ Data successfully saved to {output_file}")
