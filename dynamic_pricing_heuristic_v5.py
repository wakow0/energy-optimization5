
import pandas as pd

# Load and clean data
df = pd.read_csv("processed_data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")

df = df.rename(columns={
    "ac_primary_load": "load",
    "dc_ground_1500vdc_power_output": "solar",
    "windflow_33_[500kw]_power_output": "wind",
    "total_consumption_rate": "import_price",
    "grid_sellback_rate": "export_price"
})

df["renewable_gen"] = df["solar"] + df["wind"]
time_col = df["time"]

BATTERY_CAPACITY = 2000
BATTERY_MAX_CHARGE = 1000
BATTERY_MAX_DISCHARGE = 1000
SOC_MIN = 0.05 * BATTERY_CAPACITY
SOC_MAX = 1.0 * BATTERY_CAPACITY
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95

EXPORT_PRICE_THRESHOLD = 0.06
SOC_DISCHARGE_THRESHOLD = 0.10

# Initialize output DataFrame
output = pd.DataFrame()
output["time"] = time_col
output["P_import (kW)"] = 0.0
output["P_export (kW)"] = 0.0
output["P_bat_ch (kW)"] = 0.0
output["P_bat_dis (kW)"] = 0.0
output["SOC (%)"] = 0.0

soc = SOC_MIN

# Run simulation
for i in range(len(df)):
    demand = df.loc[i, "load"]
    renewable = df.loc[i, "renewable_gen"]
    import_price = df.loc[i, "import_price"]
    export_price = df.loc[i, "export_price"]

    p_ch, p_dis, p_imp, p_exp = 0.0, 0.0, 0.0, 0.0

    if renewable > demand:
        excess = renewable - demand
        capacity_left = max(0, (SOC_MAX - soc) / CHARGE_EFFICIENCY)
        charge_power = min(excess, BATTERY_MAX_CHARGE, capacity_left)
        p_ch = charge_power
        soc += p_ch * CHARGE_EFFICIENCY

        exportable = excess - charge_power
        if export_price > EXPORT_PRICE_THRESHOLD and soc > 0.5 * BATTERY_CAPACITY:
            p_exp = max(0, min(exportable, BATTERY_MAX_DISCHARGE))

    else:
        shortfall = demand - renewable
        if soc > SOC_DISCHARGE_THRESHOLD * BATTERY_CAPACITY:
            dis_power = min(shortfall, BATTERY_MAX_DISCHARGE, (soc - SOC_DISCHARGE_THRESHOLD * BATTERY_CAPACITY) * DISCHARGE_EFFICIENCY)
            p_dis = dis_power
            soc -= dis_power / DISCHARGE_EFFICIENCY
            shortfall -= dis_power
        if shortfall > 0:
            p_imp = shortfall

    soc = max(SOC_MIN, min(soc, SOC_MAX))

    output.loc[i, "P_bat_ch (kW)"] = p_ch
    output.loc[i, "P_bat_dis (kW)"] = p_dis
    output.loc[i, "P_import (kW)"] = p_imp
    output.loc[i, "P_export (kW)"] = p_exp
    output.loc[i, "SOC (%)"] = (soc / BATTERY_CAPACITY) * 100

output.to_csv("dynamic_pricing_v5_solution.csv", index=False)
print("✅ Saved as dynamic_pricing_heuristic_v5.csv")
