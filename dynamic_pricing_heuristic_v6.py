
import pandas as pd
import numpy as np
import os
import time

# ========================
# Dynamic Pricing Heuristic v6
# ========================
start_time = time.time()

DATA_FILE = "processed_data.csv"
OUTPUT_FILE = "dynamic_pricing_v6_solution.csv"

BATTERY_CAPACITY = 2000  # kWh
BATTERY_MAX_CHARGE = 1000  # kW
BATTERY_MAX_DISCHARGE = 1000  # kW
SOC_MIN = 0.05 * BATTERY_CAPACITY
SOC_MAX = 1.0 * BATTERY_CAPACITY
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95

IMPORT_PRICE_THRESHOLD = 0.15
EXPORT_PRICE_THRESHOLD = 0.12

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")

required_columns = [
    "time", "total_consumption_rate", "grid_sellback_rate", "ac_primary_load",
    "dc_ground_1500vdc_power_output", "windflow_33_[500kw]_power_output"
]
for col in required_columns:
    if col not in df.columns:
        df[col] = 0

df["P_bat_ch (kW)"] = 0.0
df["P_bat_dis (kW)"] = 0.0
df["P_import (kW)"] = 0.0
df["P_export (kW)"] = 0.0
df["SOC (%)"] = 0.0

soc = SOC_MIN

for i in range(len(df)):
    demand = df.loc[i, "ac_primary_load"]
    renewable_gen = df.loc[i, "dc_ground_1500vdc_power_output"] + df.loc[i, "windflow_33_[500kw]_power_output"]
    import_price = df.loc[i, "total_consumption_rate"]
    export_price = df.loc[i, "grid_sellback_rate"]

    # Charge battery if prices are low and renewable is available
    if renewable_gen > demand and import_price <= IMPORT_PRICE_THRESHOLD:
        excess_power = renewable_gen - demand
        charge_power = min(excess_power, BATTERY_MAX_CHARGE, max(0, (SOC_MAX - soc) / CHARGE_EFFICIENCY))
        df.loc[i, "P_bat_ch (kW)"] = charge_power
        soc += charge_power * CHARGE_EFFICIENCY

    # Discharge battery if sellback price is high or to avoid costly import
    elif soc > SOC_MIN and (export_price >= EXPORT_PRICE_THRESHOLD or import_price > IMPORT_PRICE_THRESHOLD):
        discharge_power = min(BATTERY_MAX_DISCHARGE, (soc - SOC_MIN) * DISCHARGE_EFFICIENCY, demand)
        df.loc[i, "P_bat_dis (kW)"] = discharge_power
        soc -= discharge_power / DISCHARGE_EFFICIENCY

    # Export if battery full and generation exceeds demand
    if soc >= SOC_MAX and renewable_gen > demand:
        export_power = min(renewable_gen - demand, BATTERY_MAX_DISCHARGE)
        df.loc[i, "P_export (kW)"] = export_power

    # Grid import if battery and renewable insufficient
    remaining_demand = demand - df.loc[i, "P_bat_dis (kW)"] - renewable_gen
    if remaining_demand > 0:
        df.loc[i, "P_import (kW)"] = remaining_demand

    soc = max(SOC_MIN, min(soc, SOC_MAX))
    df.loc[i, "SOC (%)"] = (soc / BATTERY_CAPACITY) * 100

output_cols = ["time", "P_import (kW)", "P_export (kW)", "P_bat_ch (kW)", "P_bat_dis (kW)", "SOC (%)"]
df[output_cols].to_csv(OUTPUT_FILE, index=False)

end_time = time.time()
print(f"✅ Dynamic Pricing v6 completed. Results saved to {OUTPUT_FILE}")
print(f"Execution time: {end_time - start_time:.2f} seconds")
