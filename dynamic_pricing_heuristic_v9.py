import pandas as pd

# Load processed data
df = pd.read_csv("processed_data.csv")
df["SOC (%)"] = 50  # Initial SoC at 50%

# Parameters
battery_capacity_kwh = 2000
max_charge_kw = 1000
max_discharge_kw = 1000
soc_min = 5
soc_max = 100
charge_eff = 0.95
discharge_eff = 0.95

P_import, P_export, P_bat_ch, P_bat_dis, SOC = [], [], [], [], []

for i in range(len(df)):
    row = df.iloc[i]
    soc = SOC[-1] if i > 0 else 50

    load = row["Total Consumption Rate"]
    gen = row["DC Ground 1500VDC Power Output"] + row["Windflow 33 [500kW] Power Output"]
    net = gen - load
    import_price = row["Grid Purchases"]
    export_price = row["Grid Sellback Rate"]

    charge = discharge = imp = exp = 0

    # Export if excess gen and SoC > 20%
    if net > 0:
        if soc > 20 and export_price > 0.05:
            discharge = min(max_discharge_kw, net)
            soc -= discharge * (1 / discharge_eff) / battery_capacity_kwh * 100
            exp = min(discharge, net)
        elif soc < soc_max and net > 0:
            charge = min(max_charge_kw, net)
            soc += charge * charge_eff / battery_capacity_kwh * 100
        else:
            exp = net  # curtail excess

    # Import if deficit and SoC < 80%
    elif net < 0:
        demand = abs(net)
        if soc > soc_min and import_price > 0.05:
            discharge = min(max_discharge_kw, demand)
            soc -= discharge * (1 / discharge_eff) / battery_capacity_kwh * 100
            imp = max(0, demand - discharge)
        elif soc < 80:
            imp = min(demand, max_charge_kw)
            soc += imp * charge_eff / battery_capacity_kwh * 100
        else:
            imp = demand

    # Clip SoC
    soc = max(min(soc, soc_max), soc_min)

    # Save
    P_import.append(imp)
    P_export.append(exp)
    P_bat_ch.append(charge)
    P_bat_dis.append(discharge)
    SOC.append(soc)

# Save result
output = pd.DataFrame({
    "time": df["time"],
    "P_import (kW)": P_import,
    "P_export (kW)": P_export,
    "P_bat_ch (kW)": P_bat_ch,
    "P_bat_dis (kW)": P_bat_dis,
    "SOC (%)": SOC
})

output.to_csv("dynamic_pricing_v9_solution.csv", index=False)
print("Saved dynamic_pricing_v9_solution.csv")