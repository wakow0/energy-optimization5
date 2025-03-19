import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# Load your parsed dataset
df = pd.read_csv("processed_data.csv")

# Define parameters
battery_capacity = 2000  # kWh
battery_max_charge = 1000  # kW
battery_max_discharge = 1000  # kW
soc_min = 5  # %
soc_max = 100  # %
eta_ch = 0.95
eta_dis = 0.95
FIXED_IMPORT_PRICE = 0.1939
FIXED_EXPORT_PRICE = 0.0769
time_step = 0.5  # 30-minute intervals

# Pricing thresholds
Q1_import = df["total_consumption_rate"].quantile(0.25)
Q3_import = df["total_consumption_rate"].quantile(0.75)
Q3_export = df["grid_sellback_rate"].quantile(0.75)

# Initialize result DataFrames
df_fixed = df.copy()
df_dynamic = df.copy()

for df_version in [df_fixed, df_dynamic]:
    df_version["P_import (kW)"] = 0.0
    df_version["P_export (kW)"] = 0.0
    df_version["P_bat_ch (kW)"] = 0.0
    df_version["P_bat_dis (kW)"] = 0.0
    df_version["SOC (%)"] = 50.0

start_time = time.time()

# Optimization Loop
for strategy, df_version in [("FIXED", df_fixed), ("DYNAMIC", df_dynamic)]:
    for t in tqdm(range(1, len(df_version)), desc=f"{strategy} Pricing Optimization"):
        available_renewable = df_version.loc[t, "dc_ground_1500vdc_power_output"] + df_version.loc[t, "windflow_33_[500kw]_power_output"]
        demand = df_version.loc[t, "ac_primary_load"]

        import_price = FIXED_IMPORT_PRICE if strategy == "FIXED" else df_version.loc[t, "total_consumption_rate"]
        export_price = FIXED_EXPORT_PRICE if strategy == "FIXED" else df_version.loc[t, "grid_sellback_rate"]

        future_import_price = df_version["total_consumption_rate"].iloc[t:min(t+10, len(df_version)-1)].mean()

        # Charging decision
        if import_price <= Q1_import and future_import_price >= import_price and df_version.loc[t-1, "SOC (%)"] < soc_max:
            charge_power = min(battery_max_charge, available_renewable, (soc_max - df_version.loc[t-1, "SOC (%)"]) / 100 * battery_capacity)
        else:
            charge_power = 0

        # Discharging decision
        if (import_price >= Q3_import or export_price >= Q3_export) and df_version.loc[t-1, "SOC (%)"] > soc_min:
            discharge_power = min(battery_max_discharge, demand, (df_version.loc[t-1, "SOC (%)"] - soc_min) / 100 * battery_capacity)
        else:
            discharge_power = 0

        # Mutual exclusivity battery
        if charge_power > 0:
            discharge_power = 0

        # Calculate energy balance explicitly
        net_energy = available_renewable + discharge_power - demand - charge_power

        if net_energy >= 0:
            grid_import = 0
            grid_export = net_energy if export_price >= Q3_export else 0
        else:
            grid_import = abs(net_energy)
            grid_export = 0

        # Mutual exclusivity grid
        if grid_import > 0:
            grid_export = 0

        # Update SOC
        soc_change = ((charge_power * eta_ch - discharge_power / eta_dis) * time_step / battery_capacity) * 100
        new_soc = np.clip(df_version.loc[t-1, "SOC (%)"] + soc_change, soc_min, soc_max)

        # Store results explicitly with correct numeric assignment
        df_version.loc[t, ["P_import (kW)", "P_export (kW)", "P_bat_ch (kW)", "P_bat_dis (kW)", "SOC (%)"]] = [
            float(grid_import), float(grid_export), float(charge_power), float(discharge_power), float(new_soc)
        ]





# Save results explicitly
df_fixed.to_csv("solution_output_FIXED_v20.csv", index=False)
df_dynamic.to_csv("solution_output_DYNAMIC_v20.csv", index=False)

# Execution summary
execution_time = time.time() - start_time
print(f"⏳ Total Execution Time: {execution_time:.2f} seconds")

# Validation function
def validate_solution(df_path):
    df_check = pd.read_csv(df_path)
    violations = []

    for t in range(len(df_check)):
        renewable = df.loc[t, "dc_ground_1500vdc_power_output"] + df.loc[t, "windflow_33_[500kw]_power_output"]
        demand = df.loc[t, "ac_primary_load"]

        balance = available_renewable + df_check.loc[t, "P_import (kW)"] + df_check.loc[t, "P_bat_dis (kW)"] - demand - df_check.loc[t, "P_bat_ch (kW)"] - df_check.loc[t, "P_export (kW)"]

        if not np.isclose(balance, 0, atol=0.01):
            violations.append(f"Violation at timestep {t}: Imbalance = {balance:.2f} kW")

    if violations:
        print(f"❌ Violations in {df_path}:")
        for violation in violations[:10]:
            print(violation)
    else:
        print(f"✅ No violations found in {df_path}")

# Validate immediately
print("Validation Results:")
validate_solution("solution_output_FIXED_v20.csv")
validate_solution("solution_output_DYNAMIC_v20.csv")
