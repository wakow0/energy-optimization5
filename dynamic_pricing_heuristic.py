
import pandas as pd
import numpy as np

# Parameters
BATTERY_CAPACITY = 2000  # kWh
BATTERY_MAX_CHARGE = 1000  # kW
BATTERY_MAX_DISCHARGE = 1000  # kW
SOC_MIN = 0.05 * BATTERY_CAPACITY
SOC_MAX = 1.0 * BATTERY_CAPACITY
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95
IMPORT_PRICE_THRESHOLD = 0.18
EXPORT_PRICE_THRESHOLD = 0.13

# Load data
df = pd.read_csv("processed_data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
df["time"] = pd.to_datetime(df["time"])
df["total_renewable"] = df["dc_ground_1500vdc_power_output"] + df["windflow_33_[500kw]_power_output"]

# Initialize result columns
df["P_import (kW)"] = 0.0
df["P_export (kW)"] = 0.0
df["P_bat_ch (kW)"] = 0.0
df["P_bat_dis (kW)"] = 0.0
df["SOC (%)"] = 0.0

soc = SOC_MIN

# Heuristic loop with dynamic pricing
for i in range(len(df)):
    demand = df.loc[i, "ac_primary_load"]
    renewable = df.loc[i, "total_renewable"]
    import_price = df.loc[i, "total_consumption_rate"]
    export_price = df.loc[i, "grid_sellback_rate"]

    net_demand = demand - renewable

    # Discharge battery to meet demand if import is expensive
    if net_demand > 0 and soc > SOC_MIN and import_price >= IMPORT_PRICE_THRESHOLD:
        discharge = min(net_demand, BATTERY_MAX_DISCHARGE, (soc - SOC_MIN) * DISCHARGE_EFFICIENCY)
        df.loc[i, "P_bat_dis (kW)"] = discharge
        soc -= discharge / DISCHARGE_EFFICIENCY
        net_demand -= discharge

    # Import remaining demand
    if net_demand > 0:
        df.loc[i, "P_import (kW)"] = net_demand

    # Charge battery from excess renewable
    if renewable > demand and soc < SOC_MAX:
        excess = renewable - demand
        charge = min(excess, BATTERY_MAX_CHARGE, (SOC_MAX - soc) / CHARGE_EFFICIENCY)
        df.loc[i, "P_bat_ch (kW)"] = charge
        soc += charge * CHARGE_EFFICIENCY

    # Export excess if battery is full and export price is good
    total_used = demand + df.loc[i, "P_bat_ch (kW)"]
    if renewable > total_used and export_price >= EXPORT_PRICE_THRESHOLD:
        export = renewable - total_used
        df.loc[i, "P_export (kW)"] = export

    # SOC clip and save
    soc = max(SOC_MIN, min(soc, SOC_MAX))
    df.loc[i, "SOC (%)"] = soc / BATTERY_CAPACITY * 100

# Save output
output_df = df[["time", "P_import (kW)", "P_export (kW)", "P_bat_ch (kW)", "P_bat_dis (kW)", "SOC (%)"]]
output_df.to_csv("dynamic_pricing_solution.csv", index=False)
print("✅ Dynamic pricing solution saved to 'dynamic_pricing_solution.csv'")
