import pandas as pd
import os
import time

start_time = time.time()

DATA_FILE = "processed_data.csv"
OUTPUT_FILE = "heuristic_optimization_results_v7.csv"

# System constants
BATTERY_CAPACITY = 2000  # kWh
BATTERY_MAX_CHARGE = 1000  # kW
BATTERY_MAX_DISCHARGE = 1000  # kW
SOC_MIN = 0.05 * BATTERY_CAPACITY
SOC_MAX = 1.0 * BATTERY_CAPACITY
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95
SOC = SOC_MIN

# Pricing thresholds (relaxed)
GRID_CHARGE_PRICE_THRESHOLD = 0.20

# Load and preprocess data
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")

required_cols = [
    "time", "total_consumption_rate", "grid_sellback_rate", "ac_primary_load",
    "dc_ground_1500vdc_power_output", "windflow_33_[500kw]_power_output"
]
for col in required_cols:
    if col not in df.columns:
        df[col] = 0

# Initialize output columns (explicit float type)
df["P_bat_ch (kW)"] = 0.0
df["P_bat_dis (kW)"] = 0.0
df["P_import (kW)"] = 0.0
df["P_export (kW)"] = 0.0
df["SOC (%)"] = 0.0
df["Revenue (£)"] = 0.0

# Heuristic control loop
for i in range(len(df)):
    demand = df.loc[i, "ac_primary_load"]
    solar = df.loc[i, "dc_ground_1500vdc_power_output"]
    wind = df.loc[i, "windflow_33_[500kw]_power_output"]
    renewable = solar + wind
    import_price = df.loc[i, "total_consumption_rate"]
    sellback_price = df.loc[i, "grid_sellback_rate"]

    # 1. Meet demand from renewable
    net_demand = demand - renewable

    # 2. Discharge battery to cover net demand (priority)
    if net_demand > 0 and SOC > SOC_MIN:
        discharge = min(net_demand, BATTERY_MAX_DISCHARGE, (SOC - SOC_MIN) * DISCHARGE_EFFICIENCY)
        df.loc[i, "P_bat_dis (kW)"] += discharge
        SOC -= discharge / DISCHARGE_EFFICIENCY
        net_demand -= discharge

    # 3. Import remaining demand
    if net_demand > 0:
        df.loc[i, "P_import (kW)"] = net_demand

    # 4. Charge battery from excess renewable
    if renewable > demand and SOC < SOC_MAX:
        surplus = renewable - demand
        charge = min(surplus, BATTERY_MAX_CHARGE, (SOC_MAX - SOC) / CHARGE_EFFICIENCY)
        df.loc[i, "P_bat_ch (kW)"] = charge
        SOC += charge * CHARGE_EFFICIENCY
        renewable -= charge

    # 5. Grid charge if price is low and SOC < max
    if import_price < GRID_CHARGE_PRICE_THRESHOLD and SOC < SOC_MAX:
        grid_charge = min(BATTERY_MAX_CHARGE, (SOC_MAX - SOC) / CHARGE_EFFICIENCY)
        df.loc[i, "P_bat_ch (kW)"] += grid_charge
        df.loc[i, "P_import (kW)"] += grid_charge
        SOC += grid_charge * CHARGE_EFFICIENCY

    # 6. Discharge battery to export if profitable
    if sellback_price > import_price and SOC > SOC_MIN:
        discharge_to_export = min(BATTERY_MAX_DISCHARGE, (SOC - SOC_MIN) * DISCHARGE_EFFICIENCY)
        df.loc[i, "P_bat_dis (kW)"] += discharge_to_export
        df.loc[i, "P_export (kW)"] += discharge_to_export
        SOC -= discharge_to_export / DISCHARGE_EFFICIENCY

    # 7. Export any leftover renewable
    total_used = demand + df.loc[i, "P_bat_ch (kW)"]
    if renewable > total_used:
        export = renewable - total_used
        df.loc[i, "P_export (kW)"] += export

    # 8. Revenue from export
    df.loc[i, "Revenue (£)"] = df.loc[i, "P_export (kW)"] * sellback_price

    # 9. Final SOC clipping and record
    SOC = max(SOC_MIN, min(SOC, SOC_MAX))
    df.loc[i, "SOC (%)"] = SOC / BATTERY_CAPACITY * 100

# Save results
df[["time", "P_import (kW)", "P_export (kW)", "P_bat_ch (kW)", "P_bat_dis (kW)", "SOC (%)", "Revenue (£)"]].to_csv(OUTPUT_FILE, index=False)

print(f"✅ Battery-optimized heuristic completed. Results saved to {OUTPUT_FILE}")
print(f"⏱️ Execution time: {time.time() - start_time:.2f} seconds")
