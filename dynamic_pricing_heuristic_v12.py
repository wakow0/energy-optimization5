
import pandas as pd

# Load data
df = pd.read_csv("processed_data.csv")

# Output columns
results = []

# Constants
MAX_SOC = 100
MIN_SOC = 5
SOC = 50
BATTERY_CAPACITY = 2000  # in kWh
MAX_CHARGE = 1000  # kW
MAX_DISCHARGE = 1000  # kW
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95

t_low = 278.7361
t_high = 0.0846

for index, row in df.iterrows():
    load = row["total_consumption_rate"]
    pv = row["dc_ground_1500vdc_power_output"]
    wind = row["windflow_33_[500kw]_power_output"]
    price_import = row["grid_purchases"]
    price_export = row["grid_sellback_rate"]

    gen = pv + wind
    net = gen - load

    p_import = 0
    p_export = 0
    p_ch = 0
    p_dis = 0

    # Improved logic for dynamic pricing + battery use
    if net > 0:
        if SOC < 95 and price_export < t_high:
            p_ch = min(MAX_CHARGE, net)
            SOC += (p_ch * CHARGE_EFFICIENCY) / BATTERY_CAPACITY * 100
        else:
            p_export = max(0, net)
    else:
        if SOC > 20 and price_import > t_low:
            p_dis = min(MAX_DISCHARGE, -net)
            SOC -= (p_dis / DISCHARGE_EFFICIENCY) / BATTERY_CAPACITY * 100
        else:
            p_import = max(0, -net)

    SOC = max(MIN_SOC, min(SOC, MAX_SOC))

    results.append([
        row["time"],
        round(p_import, 2),
        round(p_export, 2),
        round(p_ch, 2),
        round(p_dis, 2),
        round(SOC, 2)
    ])

# Save results
output = pd.DataFrame(results, columns=[
    "time", "P_import (kW)", "P_export (kW)", "P_bat_ch (kW)", "P_bat_dis (kW)", "SOC (%)"
])
output.to_csv("dynamic_pricing_v12_solution.csv", index=False)
