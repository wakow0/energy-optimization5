
import pandas as pd
import numpy as np

# Parameters
BATTERY_CAPACITY = 2000
BATTERY_MAX_CHARGE = 1000
BATTERY_MAX_DISCHARGE = 1000
SOC_MIN = 0.05 * BATTERY_CAPACITY
SOC_MAX = 1.0 * BATTERY_CAPACITY
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95

EXPORT_PRICE_THRESHOLD = 0.05
IMPORT_PRICE_THRESHOLD = 0.15
SOC_EXPORT_THRESHOLD = 0.3 * BATTERY_CAPACITY
SOC_DISCHARGE_THRESHOLD = 0.08 * BATTERY_CAPACITY

# Load and prepare data
df = pd.read_csv("processed_data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
df["total_renewable"] = df["dc_ground_1500vdc_power_output"] + df["windflow_33_[500kw]_power_output"]

df["P_import (kW)"] = 0.0
df["P_export (kW)"] = 0.0
df["P_bat_ch (kW)"] = 0.0
df["P_bat_dis (kW)"] = 0.0
df["SOC (%)"] = 0.0

soc = SOC_MIN

# Logic
for i in range(len(df)):
    demand = df.loc[i, "ac_primary_load"]
    renewable = df.loc[i, "total_renewable"]
    import_price = df.loc[i, "total_consumption_rate"]
    export_price = df.loc[i, "grid_sellback_rate"]

    p_imp, p_exp, p_ch, p_dis = 0.0, 0.0, 0.0, 0.0

    net_demand = demand - renewable

    if net_demand > 0 and soc > SOC_DISCHARGE_THRESHOLD:
        discharge_power = min(net_demand, BATTERY_MAX_DISCHARGE, (soc - SOC_DISCHARGE_THRESHOLD) * DISCHARGE_EFFICIENCY)
        p_dis = discharge_power
        soc -= discharge_power / DISCHARGE_EFFICIENCY
        net_demand -= discharge_power

    if net_demand > 0:
        p_imp = net_demand

    surplus = renewable - demand
    if surplus > 0 and soc < SOC_MAX:
        charge_power = min(surplus, BATTERY_MAX_CHARGE, (SOC_MAX - soc) / CHARGE_EFFICIENCY)
        p_ch = charge_power
        soc += charge_power * CHARGE_EFFICIENCY
        surplus -= charge_power

    if surplus > 0 and soc > SOC_EXPORT_THRESHOLD and export_price >= EXPORT_PRICE_THRESHOLD:
        p_exp = min(surplus, BATTERY_MAX_DISCHARGE)

    soc = max(SOC_MIN, min(soc, SOC_MAX))
    df.loc[i, "P_import (kW)"] = p_imp
    df.loc[i, "P_export (kW)"] = p_exp
    df.loc[i, "P_bat_ch (kW)"] = p_ch
    df.loc[i, "P_bat_dis (kW)"] = p_dis
    df.loc[i, "SOC (%)"] = (soc / BATTERY_CAPACITY) * 100

# Save result
output_cols = ["time", "P_import (kW)", "P_export (kW)", "P_bat_ch (kW)", "P_bat_dis (kW)", "SOC (%)"]
df[output_cols].to_csv("dynamic_pricing_v8_solution.csv", index=False)
print("✅ Saved dynamic_pricing_v8_solution.csv")
