import pandas as pd
import numpy as np

# Load input data
df = pd.read_csv("processed_data.csv")

# Parameters
MAX_SOC = 100
MIN_SOC = 5
BATTERY_CAPACITY = 2000  # kWh
INVERTER_CAPACITY = 1000  # kW
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95
MAX_CHANGE = 1000  # kW or %

# Initialize outputs
results = {
    "time": [],
    "p_import (kw)": [],
    "p_export (kw)": [],
    "p_bat_ch (kw)": [],
    "p_bat_dis (kw)": [],
    "soc (%)": []
}

soc = 50  # Initial SoC %

# Define thresholds
t_low = df["grid_purchases"].mean() * 0.85
t_high = df["grid_sellback_rate"].mean() * 1.10

# Optimization loop
for i, row in df.iterrows():
    load = row["total_consumption_rate"]
    solar = row["dc_ground_1500vdc_power_output"]
    wind = row["windflow_33_[500kw]_power_output"]
    gen = solar + wind
    import_price = row["grid_purchases"]
    export_price = row["grid_sellback_rate"]

    p_bat_ch = 0
    p_bat_dis = 0
    p_import = 0
    p_export = 0

    if import_price < t_low and soc < 95:
        p_bat_ch = min(INVERTER_CAPACITY, (BATTERY_CAPACITY * (MAX_SOC - soc) / 100) / CHARGE_EFFICIENCY)
    elif export_price > t_high and soc > 20:
        p_bat_dis = min(INVERTER_CAPACITY, (BATTERY_CAPACITY * (soc - MIN_SOC) / 100) * DISCHARGE_EFFICIENCY)

    available = gen + p_bat_dis
    shortfall = load - available

    if shortfall > 0:
        p_import = shortfall
    else:
        surplus = -shortfall
        if p_bat_ch > 0:
            used_for_charge = min(p_bat_ch, surplus)
            p_bat_ch = used_for_charge
            surplus -= used_for_charge
        p_export = surplus

    # Update SoC
    soc += (p_bat_ch * CHARGE_EFFICIENCY - p_bat_dis / DISCHARGE_EFFICIENCY) * 100 / BATTERY_CAPACITY
    soc = min(max(soc, MIN_SOC), MAX_SOC)

    # Apply spike limits
    for key, val in zip(
        ["p_import (kw)", "p_export (kw)", "p_bat_ch (kw)", "p_bat_dis (kw)", "soc (%)"],
        [p_import, p_export, p_bat_ch, p_bat_dis, soc]
    ):
        prev_val = results[key][-1] if results[key] else val
        if abs(val - prev_val) > MAX_CHANGE:
            val = prev_val + MAX_CHANGE if val > prev_val else prev_val - MAX_CHANGE
        if "soc" not in key and val > INVERTER_CAPACITY:
            val = INVERTER_CAPACITY
        if "soc" not in key and val < 0:
            val = 0
        results[key].append(val)

    results["time"].append(row["time"])

# Save results
output_df = pd.DataFrame(results)
output_df.to_csv("dynamic_pricing_v13_solution.csv", index=False)
print("✅ Saved: dynamic_pricing_heuristic_v13_solution.csv")
