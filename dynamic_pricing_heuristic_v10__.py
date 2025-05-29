# dynamic_pricing_heuristic_v10.py

import pandas as pd

# Load processed data
df = pd.read_csv("processed_data.csv")

# Initialize output lists
results = []

# Constants
SOC_MIN = 5
SOC_MAX = 95
SOC_START = 50
BATTERY_CAPACITY = 2000  # in kWh
CHARGE_EFF = 0.95
DISCHARGE_EFF = 0.95
MAX_CHARGE = 1000  # in kW
MAX_DISCHARGE = 1000  # in kW
EXPORT_LIMIT = 1000  # in kW
IMPORT_LIMIT = 1000  # in kW

# Thresholds

t_low = df['grid_purchases'].mean() * 0.85
t_high = df['grid_sellback_rate'].mean() * 1.10


soc = SOC_START

for i in range(len(df)):
    row = df.iloc[i]
    load = row["Total Consumption Rate"]
    pv = row["DC Ground 1500VDC Power Output"]
    wind = row["Windflow 33 [500kW] Power Output"]
    import_price = row["Grid Purchases"]
    export_price = row["Grid Sellback Rate"]

    next_export_price = df.iloc[i+1]["Grid Sellback Rate"] if i < len(df)-1 else export_price

    gen = pv + wind
    net_gen = gen - load

    p_import = 0
    p_export = 0
    p_bat_ch = 0
    p_bat_dis = 0

    # Decide on charging/discharging/exporting based on logic
    if net_gen > 0:
        # Surplus generation: prioritize export or charging
        if export_price >= t_high or soc >= 90:
            p_export = min(EXPORT_LIMIT, net_gen)
        elif soc < SOC_MAX:
            energy_to_store = min(net_gen, MAX_CHARGE, ((SOC_MAX - soc) / 100) * BATTERY_CAPACITY / CHARGE_EFF)
            p_bat_ch = energy_to_store
            soc += (p_bat_ch * CHARGE_EFF) / BATTERY_CAPACITY * 100
        else:
            p_export = min(EXPORT_LIMIT, net_gen)
    else:
        # Deficit: load > generation
        shortfall = abs(net_gen)

        if soc > SOC_MIN:
            energy_available = min(shortfall, MAX_DISCHARGE, ((soc - SOC_MIN) / 100) * BATTERY_CAPACITY * DISCHARGE_EFF)
            p_bat_dis = energy_available
            soc -= (p_bat_dis / DISCHARGE_EFF) / BATTERY_CAPACITY * 100
            shortfall -= p_bat_dis

        if shortfall > 0:
            p_import = min(shortfall, IMPORT_LIMIT)

    # Clip SoC within limits
    soc = min(max(soc, SOC_MIN), SOC_MAX)

    results.append({
        "time": row["time"],
        "P_import (kW)": round(p_import, 2),
        "P_export (kW)": round(p_export, 2),
        "P_bat_ch (kW)": round(p_bat_ch, 2),
        "P_bat_dis (kW)": round(p_bat_dis, 2),
        "SOC (%)": round(soc, 2)
    })

# Save solution
output_df = pd.DataFrame(results)
output_df.to_csv("dynamic_pricing_v10_solution.csv", index=False)
print("✅ dynamic_pricing_v10_solution.csv saved.")
