import pandas as pd

# Load the solution file
file_path = "dynamic_pricing_v13_solution.csv"
df = pd.read_csv(file_path)

# Standardize column names
df.columns = df.columns.str.lower().str.strip()

# Extract relevant columns
charge_col = "p_bat_ch (kw)"
discharge_col = "p_bat_dis (kw)"

# Calculate total energy charged and discharged
total_charged = df[charge_col].sum()
total_discharged = df[discharge_col].sum()

# Assume battery capacity is 2000 kWh as per prior context
battery_capacity_kwh = 2000

# Calculate battery utilization
utilization_percentage = ((total_charged + total_discharged) / 2) / battery_capacity_kwh * 100


print(utilization_percentage)