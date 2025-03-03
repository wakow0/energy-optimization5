import os
import time
#from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
from tqdm import tqdm
import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD
import subprocess

# =======================
# CONFIGURABLE PARAMETERS
# =======================
BATCH_SIZE = 48
LOOK_AHEAD_WINDOW = 0
NUM_CORES = multiprocessing.cpu_count()
SMOOTHING_WINDOW = 10

# Fixed pricing strategy
FIXED_IMPORT_PRICE = 0.1939
FIXED_EXPORT_PRICE = 0.0769

# =======================
# LOAD & PREPROCESS DATA
# =======================
from data_parser import load_csv_data

df = pd.read_csv("processed_data.csv", dtype=str, low_memory=False)

# ✅ Ensure dataframe is valid before proceeding
if df.empty or len(df.columns) < 2:
    raise ValueError("❌ ERROR: Dataframe is empty! Check 'processed_data.csv'.")

# ✅ Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(":", "")

# ✅ Convert "time" column to datetime
if "time" not in df.columns:
    raise ValueError("❌ ERROR: 'time' column is missing! Check 'processed_data.csv'.")

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df.dropna(subset=["time"], inplace=True)





expected_columns = [
    "time", "total_consumption_rate", "grid_sellback_rate", "ac_primary_load",
    "dc_ground_1500vdc_power_output", "windflow_33_[500kw]_power_output",
    "wattstor_m5_0.5c_september_charge_power", "grid_purchases", "grid_sales"
]

for col in expected_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df["time"] = pd.to_datetime(df["time"], errors="coerce")

time_intervals = len(df)

# =======================
# SYSTEM SPECIFICATIONS
# =======================
battery_capacity = 2000
battery_max_charge = 1000
battery_max_discharge = 1000
inverter_capacity = 1000
soc_min = 0.05 * battery_capacity
soc_max = 1.0 * battery_capacity
eta_ch = 0.95
eta_dis = 0.95

max_change_limit = 500
max_battery_rate_change = 300

# =======================
# OPTIMIZATION FUNCTION




def optimize_energy(start_t, STRATEGY, prev_final_soc):
    model = LpProblem("Energy_Optimization", LpMinimize)
    end_t = min(start_t + BATCH_SIZE, time_intervals)

    inverse_eta_dis = 1 / eta_dis

    P_import, P_export, P_bat_ch, P_bat_dis, SOC = {}, {}, {}, {}, {}
    X_import, X_export, X_bat_ch, X_bat_dis = {}, {}, {}, {}
    Delta_P_import, Delta_P_export, Delta_P_bat_ch, Delta_P_bat_dis = {}, {}, {}, {}

    for t in range(start_t, end_t):
        P_import[t] = LpVariable(f"P_import_{t}", 0, inverter_capacity)
        P_export[t] = LpVariable(f"P_export_{t}", 0, inverter_capacity)
        P_bat_ch[t] = LpVariable(f"P_bat_ch_{t}", 0, battery_max_charge)
        P_bat_dis[t] = LpVariable(f"P_bat_dis_{t}", 0, battery_max_discharge)
        SOC[t] = LpVariable(f"SOC_{t}", soc_min, soc_max)
        X_import[t] = LpVariable(f"X_import_{t}", cat="Binary")
        X_export[t] = LpVariable(f"X_export_{t}", cat="Binary")
        X_bat_ch[t] = LpVariable(f"X_bat_ch_{t}", cat="Binary")
        X_bat_dis[t] = LpVariable(f"X_bat_dis_{t}", cat="Binary")

        if t > start_t:
            Delta_P_import[t] = LpVariable(f"Delta_P_import_{t}", 0)
            Delta_P_export[t] = LpVariable(f"Delta_P_export_{t}", 0)
            Delta_P_bat_ch[t] = LpVariable(f"Delta_P_bat_ch_{t}", 0)
            Delta_P_bat_dis[t] = LpVariable(f"Delta_P_bat_dis_{t}", 0)

    for t in range(start_t, end_t):
        if t > start_t:
            model += Delta_P_import[t] <= max_change_limit
            model += Delta_P_export[t] <= max_change_limit
            model += Delta_P_bat_ch[t] <= max_battery_rate_change
            model += Delta_P_bat_dis[t] <= max_battery_rate_change

    model += lpSum(
        P_import[t] * (FIXED_IMPORT_PRICE if STRATEGY == "FIXED" else df["total_consumption_rate"].iloc[t]) -
        P_export[t] * (FIXED_EXPORT_PRICE if STRATEGY == "FIXED" else df["grid_sellback_rate"].iloc[t])
        for t in range(start_t, end_t))

    for t in range(start_t, end_t):
        available_renewable = df["dc_ground_1500vdc_power_output"].iloc[t] + df["windflow_33_[500kw]_power_output"].iloc[t]
        demand = df["ac_primary_load"].iloc[t]

        model += available_renewable + P_bat_dis[t] + P_import[t] == demand + P_bat_ch[t] + P_export[t]

        model += P_import[t] <= X_import[t] * inverter_capacity
        model += P_export[t] <= X_export[t] * inverter_capacity
        model += X_import[t] + X_export[t] <= 1

        model += P_bat_ch[t] <= X_bat_ch[t] * battery_max_charge
        model += P_bat_dis[t] <= X_bat_dis[t] * battery_max_discharge
        model += X_bat_ch[t] + X_bat_dis[t] <= 1

        if t == start_t:
            model += SOC[t] == prev_final_soc
        else:
            model += SOC[t] == SOC[t - 1] + (P_bat_ch[t] * eta_ch) - (P_bat_dis[t] * inverse_eta_dis)

            model += Delta_P_import[t] >= P_import[t] - P_import[t - 1]
            model += Delta_P_import[t] >= P_import[t - 1] - P_import[t]
            model += Delta_P_export[t] >= P_export[t] - P_export[t - 1]
            model += Delta_P_export[t] >= P_export[t - 1] - P_export[t]
            model += Delta_P_bat_ch[t] >= P_bat_ch[t] - P_bat_ch[t - 1]
            model += Delta_P_bat_ch[t] >= P_bat_ch[t - 1] - P_bat_ch[t]
            model += Delta_P_bat_dis[t] >= P_bat_dis[t] - P_bat_dis[t - 1]
            model += Delta_P_bat_dis[t] >= P_bat_dis[t - 1] - P_bat_dis[t]

    status = model.solve(PULP_CBC_CMD(msg=0))
    if status != 1:
        print(f"❌ Warning: Solver did not find an optimal solution in batch starting at interval {start_t}. Status code: {status}")

    prev_final_soc = max(SOC[end_t - 1].varValue, soc_min)

    results = [{
        "time": df["time"].iloc[t],
        "P_import (kW)": max(P_import[t].varValue, 0),
        "P_export (kW)": max(P_export[t].varValue, 0),
        "P_bat_ch (kW)": max(P_bat_ch[t].varValue, 0),
        "P_bat_dis (kW)": max(P_bat_dis[t].varValue, 0),
        "SOC (%)": max((SOC[t].varValue / battery_capacity) * 100, 0)
    } for t in range(start_t, end_t)]

    return results, prev_final_soc

all_results_fixed = []
prev_final_soc_fixed = soc_min
prev_batch_soc_fixed = soc_min
max_soc_change_per_batch = 200  # Example limit in kWh
for start_t in tqdm(range(0, time_intervals, BATCH_SIZE), desc="FIXED Optimization Progress", unit="batch"):
    batch_results, prev_final_soc_fixed = optimize_energy(start_t, "FIXED", prev_final_soc_fixed)
    if abs(prev_final_soc_fixed - prev_batch_soc_fixed) > max_soc_change_per_batch:
        if prev_final_soc_fixed > prev_batch_soc_fixed:
            prev_final_soc_fixed = prev_batch_soc_fixed + max_soc_change_per_batch
        else:
            prev_final_soc_fixed = prev_batch_soc_fixed - max_soc_change_per_batch
    prev_batch_soc_fixed = prev_final_soc_fixed
    all_results_fixed.extend(batch_results)

all_results_dynamic = []
prev_final_soc_dynamic = soc_min
prev_batch_soc_dynamic = soc_min
for start_t in tqdm(range(0, time_intervals, BATCH_SIZE), desc="DYNAMIC Optimization Progress", unit="batch"):
    batch_results, prev_final_soc_dynamic = optimize_energy(start_t, "DYNAMIC", prev_final_soc_dynamic)
    if abs(prev_final_soc_dynamic - prev_batch_soc_dynamic) > max_soc_change_per_batch:
        if prev_final_soc_dynamic > prev_batch_soc_dynamic:
            prev_final_soc_dynamic = prev_batch_soc_dynamic + max_soc_change_per_batch
        else:
            prev_final_soc_dynamic = prev_batch_soc_dynamic - max_soc_change_per_batch
    prev_batch_soc_dynamic = prev_final_soc_dynamic
    all_results_dynamic.extend(batch_results)










# =======================
# RUN OPTIMIZATION
# =======================
if __name__ == "__main__":
    start_time = time.time()
    print("🚀 Running optimization for Fixed and Dynamic strategies...")

    all_results_fixed = []
    all_results_dynamic = []

    # ✅ Start SOC at 5% for both strategies
    prev_final_soc_fixed = 0.05 * battery_capacity
    prev_final_soc_dynamic = 0.05 * battery_capacity

    # ✅ Sequential loop for Fixed strategy
    for i in range(0, len(df) - LOOK_AHEAD_WINDOW, BATCH_SIZE):
        batch_results, prev_final_soc_fixed = optimize_energy(i, "FIXED", prev_final_soc_fixed)
        all_results_fixed.extend(batch_results)

    # ✅ Sequential loop for Dynamic strategy
    for i in range(0, len(df) - LOOK_AHEAD_WINDOW, BATCH_SIZE):
        batch_results, prev_final_soc_dynamic = optimize_energy(i, "DYNAMIC", prev_final_soc_dynamic)
        all_results_dynamic.extend(batch_results)

    # ✅ Convert results to DataFrames
    results_fixed_df = pd.DataFrame(all_results_fixed)
    results_dynamic_df = pd.DataFrame(all_results_dynamic)


    # =======================
    # LOAD HOMER DATA
    # =======================
    homer_df = df.copy()
    homer_df["P_import (kW)"] = df["grid_purchases"]
    homer_df["P_export (kW)"] = df["grid_sales"]
    homer_df["P_bat_ch (kW)"] = df["wattstor_m5_0.5c_september_charge_power"]
    homer_df["P_bat_dis (kW)"] = df["wattstor_m5_0.5c_september_discharge_power"]
    homer_df["SOC (%)"] = df["wattstor_m5_0.5c_september_state_of_charge"]

    # Keep only the relevant columns
    result_columns = ["time", "P_import (kW)", "P_export (kW)", "P_bat_ch (kW)", "P_bat_dis (kW)", "SOC (%)"]
    results_fixed_df = results_fixed_df[result_columns]
    results_dynamic_df = results_dynamic_df[result_columns]
    homer_df = homer_df[result_columns]

    results_fixed_df["time"] = results_fixed_df["time"].astype(str)
    results_dynamic_df["time"] = results_dynamic_df["time"].astype(str)
    homer_df["time"] = homer_df["time"].astype(str)
    
    print("✅ Start to Save Results...!")
    
    # =======================
    # SAVE RESULTS (Versioning)
    # =======================
    version = "v7"  # Update version as needed
    results_fixed_df.to_csv(f"WorkingCodeVersion1_FIXED_{version}.csv", index=False)
    results_dynamic_df.to_csv(f"WorkingCodeVersion1_DYNAMIC_{version}.csv", index=False)
    homer_df.to_csv(f"WorkingCodeVersion1_HOMER_{version}.csv", index=False)

    print("✅ Results saved successfully!")

    # ✅ Run Plotting
    #print("📊 Generating plots...")
    #subprocess.run(["python", "plot_results.py"])

    total_time = time.time() - start_time
    print(f"⏳ Total Execution Time: {total_time:.2f} seconds")
