
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("processed_data.csv")

# Parameters
battery_capacity = 2000  # kWh
max_charge = 1000        # kW
max_discharge = 1000     # kW
soc = 50.0               # initial SoC (%)
soc_series = []

# Thresholds based on real data
t_low = df['grid_purchases'].mean() * 0.90
t_high = df['grid_sellback_rate'].mean() * 1.05

# Decision outputs
P_import, P_export = [], []
P_bat_ch, P_bat_dis = [], []

for i, row in df.iterrows():
    load = row['total_consumption_rate']
    solar = row['dc_ground_1500vdc_power_output']
    wind = row['windflow_33_[500kw]_power_output']
    gen = solar + wind
    price_buy = row['grid_purchases']
    price_sell = row['grid_sellback_rate']

    # SOC boundaries
    soc_min = 5.0
    soc_max = 100.0
    eta_ch = 0.95
    eta_dis = 0.95

    charge_power = 0.0
    discharge_power = 0.0
    import_power = 0.0
    export_power = 0.0

    # Charging logic (renewables or low price)
    if soc < soc_max and (gen > load or price_buy < t_low):
        charge_power = min(max_charge, (battery_capacity * (soc_max - soc) / 100) / eta_ch)
        charge_power = min(charge_power, gen - load) if gen > load else charge_power
    else:
        charge_power = 0.0

    # Discharging logic (even at medium price to improve utilization)
    if soc > soc_min and (price_sell > t_high * 0.85 or load > gen):
        discharge_power = min(max_discharge, (battery_capacity * (soc - soc_min) / 100) * eta_dis)
        discharge_power = min(discharge_power, load - gen) if load > gen else discharge_power
    else:
        discharge_power = 0.0

    # Net balance after battery
    net_demand = load - gen + charge_power - discharge_power

    if net_demand > 0:
        import_power = net_demand
    elif net_demand < 0:
        export_power = abs(net_demand)

    # Update SoC
    delta_soc = (charge_power * eta_ch - discharge_power / eta_dis) * 100 / battery_capacity
    soc = min(max(soc + delta_soc, soc_min), soc_max)
    soc_series.append(soc)

    P_import.append(import_power)
    P_export.append(export_power)
    P_bat_ch.append(charge_power)
    P_bat_dis.append(discharge_power)

# Save results
output_df = pd.DataFrame({
    "time": df["time"],
    "P_import (kW)": P_import,
    "P_export (kW)": P_export,
    "P_bat_ch (kW)": P_bat_ch,
    "P_bat_dis (kW)": P_bat_dis,
    "SOC (%)": soc_series
})

output_df.to_csv("dynamic_pricing_v11_solution.csv", index=False)
