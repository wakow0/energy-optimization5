import pandas as pd
import numpy as np
import os

# Load dataset (ensure it is already parsed)
DATA_FILE = "processed_data.csv"
OUTPUT_FILE = "heuristic_optimization_results_v2.csv"

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
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")

# Ensure required columns exist
required_columns = [
    "time", "total_consumption_rate", "grid_sellback_rate", "ac_primary_load",
    "dc_ground_1500vdc_power_output", "windflow_33_[500kw]_power_output"
]
for col in required_columns:
    if col not in df.columns:
        df[col] = 0  # Fill missing data with default 0

# Initialize variables to match MILP output structure
df["P_bat_ch (kW)"] = 0  # kW
df["P_bat_dis (kW)"] = 0  # kW
df["P_import (kW)"] = 0  # kW
df["P_export (kW)"] = 0  # kW
df["SOC (%)"] = SOC_MIN  # %

# Initial SOC
soc = SOC_MIN

# Apply rule-based decisions
for i in range(len(df)):
    demand = df.loc[i, "ac_primary_load"]
    renewable_gen = df.loc[i, "dc_ground_1500vdc_power_output"] + df.loc[i, "windflow_33_[500kw]_power_output"]
    market_price = df.loc[i, "total_consumption_rate"]
    sellback_price = df.loc[i, "grid_sellback_rate"]
    
    # If renewables exceed demand, charge the battery
    # If renewables exceed demand, charge the battery
    # If renewables exceed demand, charge the battery
    if renewable_gen > demand:
        excess_power = renewable_gen - demand
        charge_power = min(excess_power, BATTERY_MAX_CHARGE, max(0, (SOC_MAX - soc) / CHARGE_EFFICIENCY))  # Prevents overcharging
        df.loc[i, "P_bat_ch (kW)"] = charge_power if charge_power > 0 else 0
        soc += charge_power * CHARGE_EFFICIENCY
        soc = min(soc, SOC_MAX)  # Ensure SOC does not exceed 100%  

    # If demand exceeds renewables, discharge battery first
    elif demand > renewable_gen and soc > SOC_MIN:
        required_power = demand - renewable_gen
        discharge_power = min(required_power, BATTERY_MAX_DISCHARGE, max(0, (soc - SOC_MIN) * DISCHARGE_EFFICIENCY))  # Prevents over-discharge
        df.loc[i, "P_bat_dis (kW)"] = discharge_power if discharge_power > 0 else 0
        soc -= discharge_power / DISCHARGE_EFFICIENCY
        soc = max(soc, SOC_MIN)  # Ensure SOC does not go below 5%

# Ensure SOC stays within limits before storing
        df.loc[i, "SOC (%)"] = soc  # Save SOC in DataFrame

# Ensure SOC stays within limits before storing
#df.loc[i, "SOC (%)"] = soc  # Save SOC in DataFrame

    
    # If battery cannot supply power, import from grid
    if demand > renewable_gen + df.loc[i, "P_bat_dis (kW)"]:
        grid_needed = demand - renewable_gen - df.loc[i, "P_bat_dis (kW)"]
        df.loc[i, "P_import (kW)"] = grid_needed if grid_needed > 0 else 0
    
    # If battery is full and market price is high, export excess
    if soc >= SOC_MAX and sellback_price > market_price:
        export_power = min(renewable_gen - demand, BATTERY_MAX_DISCHARGE)
        df.loc[i, "P_export (kW)"] = export_power if export_power > 0 else 0
    
    # Ensure SOC stays within limits before storing
    soc = max(SOC_MIN, min(soc, SOC_MAX))
    df.loc[i, "SOC (%)"] = soc  # Save SOC in DataFrame

# Save results
output_columns = [
    "time", "P_import (kW)", "P_export (kW)", "P_bat_ch (kW)", "P_bat_dis (kW)", "SOC (%)"
]
df[output_columns].to_csv(OUTPUT_FILE, index=False)
print(f"✅ Heuristic optimization completed. Results saved to {OUTPUT_FILE}")
