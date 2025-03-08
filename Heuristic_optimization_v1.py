import pandas as pd
import numpy as np
import os

# Load dataset (ensure it is already parsed)
DATA_FILE = "processed_data.csv"
OUTPUT_FILE = "heuristic_optimization_results_v1.csv"

# Battery and system constraints
BATTERY_CAPACITY = 2000  # kWh
BATTERY_MAX_CHARGE = 1000  # kW
BATTERY_MAX_DISCHARGE = 1000  # kW
SOC_MIN = 0.05 * BATTERY_CAPACITY  # Min SOC (5%)
SOC_MAX = 1.0 * BATTERY_CAPACITY  # Max SOC (100%)
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95

# Load data
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"❌ ERROR: {DATA_FILE} is missing.")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Ensure required columns exist
required_columns = [
    "time", "total_consumption_rate", "grid_sellback_rate", "ac_primary_load",
    "dc_ground_1500vdc_power_output", "windflow_33_[500kw]_power_output"
]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"❌ ERROR: Missing column: {col}")

# Initialize variables to match MILP output structure
df["battery_charge_power"] = 0  # kW
df["battery_discharge_power"] = 0  # kW
df["grid_import"] = 0  # kW
df["grid_export"] = 0  # kW
df["battery_soc"] = 0  # %
df["energy_cost"] = 0  # £
df["revenue_from_export"] = 0  # £

# Initial SOC
soc = SOC_MIN

# Apply rule-based decisions
for i in range(len(df)):
    demand = df.loc[i, "ac_primary_load"]
    renewable_gen = df.loc[i, "dc_ground_1500vdc_power_output"] + df.loc[i, "windflow_33_[500kw]_power_output"]
    market_price = df.loc[i, "total_consumption_rate"]
    sellback_price = df.loc[i, "grid_sellback_rate"]
    
    # If renewables exceed demand, charge the battery
    if renewable_gen > demand:
        excess_power = renewable_gen - demand
        charge_power = min(excess_power, BATTERY_MAX_CHARGE, SOC_MAX - soc)
        df.loc[i, "battery_charge_power"] = charge_power
        soc += charge_power * CHARGE_EFFICIENCY
    
    # If demand exceeds renewables, discharge battery first
    elif demand > renewable_gen and soc > SOC_MIN:
        required_power = demand - renewable_gen
        discharge_power = min(required_power, BATTERY_MAX_DISCHARGE, soc - SOC_MIN)
        df.loc[i, "battery_discharge_power"] = discharge_power
        soc -= discharge_power / DISCHARGE_EFFICIENCY
    
    # If battery cannot supply power, import from grid
    if demand > renewable_gen + df.loc[i, "battery_discharge_power"]:
        grid_needed = demand - renewable_gen - df.loc[i, "battery_discharge_power"]
        df.loc[i, "grid_import"] = grid_needed
    
    # If battery is full and market price is high, export excess
    if soc >= SOC_MAX and sellback_price > market_price:
        export_power = min(renewable_gen - demand, BATTERY_MAX_DISCHARGE)
        df.loc[i, "grid_export"] = export_power
    
    # Compute energy cost and revenue
    df.loc[i, "energy_cost"] = df.loc[i, "grid_import"] * market_price
    df.loc[i, "revenue_from_export"] = df.loc[i, "grid_export"] * sellback_price
    
    # Store SOC for the next iteration
    #df.loc[i, "battery_soc"] = soc
    df["battery_soc"] = np.clip(df["battery_soc"], 0, 100)  # Ensure SOC stays between 0 and 100%

# Save results
#df.to_csv(OUTPUT_FILE, index=False)


output_columns = [
    "time", "battery_charge_power", "battery_discharge_power", 
    "grid_import", "grid_export", "battery_soc", "energy_cost", 
    "revenue_from_export", "total_consumption_rate",
    "dc_ground_1500vdc_power_output", "windflow_33_[500kw]_power_output",
    "grid_sellback_rate"
]

if "rl_action" in df.columns:
    output_columns.append("rl_action")

df[output_columns].to_csv(OUTPUT_FILE, index=False)



print(f"✅ Heuristic optimization completed. Results saved to {OUTPUT_FILE}")
