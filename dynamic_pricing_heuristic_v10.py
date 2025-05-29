
import pandas as pd

# Load dataset
df = pd.read_csv("processed_data.csv")

# Thresholds for pricing strategy
t_low = df['grid_purchases'].mean() * 0.85
t_high = df['grid_sellback_rate'].mean() * 1.10

# Initialize decision variables
results = []
soc = 50  # Start SoC at 50%
max_soc = 100
min_soc = 5
battery_capacity = 2000  # kWh
charge_eff = 0.95
discharge_eff = 0.95
max_charge_power = 1000
max_discharge_power = 1000

for index, row in df.iterrows():
    load = row["total_consumption_rate"]
    solar = row["dc_ground_1500vdc_power_output"]
    wind = row["windflow_33_[500kw]_power_output"]
    import_price = row["grid_purchases"]
    export_price = row["grid_sellback_rate"]

    renewable = solar + wind
    soc_kwh = soc / 100 * battery_capacity

    P_import = 0
    P_export = 0
    P_bat_ch = 0
    P_bat_dis = 0

    # Heuristic strategy based on dynamic pricing
    if import_price < t_low and soc < max_soc:
        # Charge battery when import is cheap
        P_bat_ch = min(max_charge_power, (max_soc - soc) / 100 * battery_capacity / charge_eff)
        P_bat_ch = min(P_bat_ch, load - renewable)
        P_bat_ch = max(P_bat_ch, 0)
    elif export_price > t_high and soc > min_soc:
        # Discharge battery when export is lucrative
        P_bat_dis = min(max_discharge_power, (soc - min_soc) / 100 * battery_capacity * discharge_eff)
        P_bat_dis = min(P_bat_dis, load - renewable)
        P_bat_dis = max(P_bat_dis, 0)

    net_load = load - renewable - P_bat_dis + P_bat_ch

    if net_load >= 0:
        P_import = net_load
    else:
        P_export = -net_load

    # Update SoC
    soc += (P_bat_ch * charge_eff - P_bat_dis / discharge_eff) / battery_capacity * 100
    soc = max(min(soc, max_soc), min_soc)

    results.append({
        "time": row["time"],
        "P_import (kW)": round(P_import, 2),
        "P_export (kW)": round(P_export, 2),
        "P_bat_ch (kW)": round(P_bat_ch, 2),
        "P_bat_dis (kW)": round(P_bat_dis, 2),
        "SOC (%)": round(soc, 2)
    })

# Save result
output_df = pd.DataFrame(results)
output_df.to_csv("dynamic_pricing_v10_solution.csv", index=False)
