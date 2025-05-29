import pandas as pd

# Load the latest solution file
file_path = "dynamic_pricing_v13_solution.csv"
file_path = "WorkingCodeVersion1_HOMER_v10.csv"
file_path = "solution_output_FIXED_v4.csv"

df = pd.read_csv(file_path)

# Normalize column names
df.columns = df.columns.str.lower().str.strip()

# Check cumulative charge/discharge to calculate utilization ratio
charge_col = "p_bat_ch (kw)"
discharge_col = "p_bat_dis (kw)"
soc_col = "soc (%)"

# Ensure required columns exist
required_cols = [charge_col, discharge_col, soc_col]
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"Missing one or more required columns in data: {required_cols}")

# Total charge and discharge energy
total_charge_energy = df[charge_col].sum()
total_discharge_energy = df[discharge_col].sum()
total_energy_cycled = total_charge_energy + total_discharge_energy

# Assume 2000 kWh battery (100% SoC) * 2 (charge+discharge) = 4000 kWh full cycle energy
battery_capacity_kwh = 2000
max_expected_cycle_energy = battery_capacity_kwh * 2 * len(df)

# Compute utilization percentage based on expected normal operation
battery_utilization_ratio = (total_energy_cycled / (battery_capacity_kwh * len(df))) * 100

print(battery_utilization_ratio)
