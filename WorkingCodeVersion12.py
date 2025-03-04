#implement WorkingCodeVersion11 with the overlapping strategy and improved spike control?


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
BATCH_SIZE = 60
LOOK_AHEAD_WINDOW = 0
NUM_CORES = multiprocessing.cpu_count()
SMOOTHING_WINDOW = 10

OVERLAP_BATCH_SIZE = 40
OVERLAP_SAVE_SIZE = 20
# Fixed pricing strategy
FIXED_IMPORT_PRICE = 0.1939
FIXED_EXPORT_PRICE = 0.0769

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

max_power_change_per_batch = 800
max_soc_change_per_batch = 600

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
# OPTIMIZATION FUNCTION

def optimize_energy(start_t, STRATEGY, prev_final_soc, prev_batch_powers):
    model = LpProblem("Energy_Optimization", LpMinimize)
    end_t = min(start_t + OVERLAP_BATCH_SIZE, time_intervals)

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

    status = model.solve(PULP_CBC_CMD(msg=0))
    if status != 1:
        print(f"❌ Warning: Solver did not find an optimal solution in batch starting at interval {start_t}. Status code: {status}")
        return None, prev_final_soc, prev_batch_powers

    soc_value = SOC[end_t - 1].varValue
    if soc_value is None:
        print(f"❌ Warning: No valid SoC value for interval {end_t - 1}. Using soc_min.")
        prev_final_soc = soc_min
    else:
        prev_final_soc = max(soc_value, soc_min)


    results = [{
    "time": df["time"].iloc[t],
    "P_import (kW)": max((P_import[t].varValue or 0), 0),
    "P_export (kW)": max((P_export[t].varValue or 0), 0),
    "P_bat_ch (kW)": max((P_bat_ch[t].varValue or 0), 0),
    "P_bat_dis (kW)": max((P_bat_dis[t].varValue or 0), 0),
    "SOC (%)": max((((SOC[t].varValue or soc_min) / battery_capacity) * 100), 0)
    } for t in range(start_t, end_t)]


    final_powers = {
    "P_import": P_import[end_t - 1].varValue or 0,
    "P_export": P_export[end_t - 1].varValue or 0,
    "P_bat_ch": P_bat_ch[end_t - 1].varValue or 0,
    "P_bat_dis": P_bat_dis[end_t - 1].varValue or 0
    }



    return results, prev_final_soc, final_powers




# =======================
# RUN OPTIMIZATION
# =======================
if __name__ == "__main__":
    start_time = time.time()
    print("🚀 Running optimization for Fixed and Dynamic strategies...")

    all_results_fixed = []
    all_results_dynamic = []

    # ✅ Start SOC at 5% for both strategies
    prev_final_soc_fixed = soc_min
    prev_final_soc_dynamic = soc_min

    prev_batch_powers_fixed = {"P_import": 0, "P_export": 0, "P_bat_ch": 0, "P_bat_dis": 0}
    prev_batch_powers_dynamic = {"P_import": 0, "P_export": 0, "P_bat_ch": 0, "P_bat_dis": 0}

    #OVERLAP_SAVE_SIZE = 24  # Number of intervals to save from each batch

    #OVERLAP_BATCH_SIZE = 4
    #OVERLAP_SAVE_SIZE = 2

    failed_batches_fixed = []
    failed_batches_dynamic = []

    # FIXED Strategy
    for start_t in tqdm(range(0, time_intervals - OVERLAP_BATCH_SIZE, OVERLAP_SAVE_SIZE), desc="FIXED Optimization Progress", unit="batch"):
        batch_results, prev_final_soc_fixed, prev_batch_powers_fixed = optimize_energy(
            start_t, "FIXED", prev_final_soc_fixed, prev_batch_powers_fixed)
        if batch_results:
            all_results_fixed.extend(batch_results[:OVERLAP_SAVE_SIZE])
        else:
            failed_batches_fixed.append(start_t)

    # DYNAMIC Strategy
    for start_t in tqdm(range(0, time_intervals - OVERLAP_BATCH_SIZE, OVERLAP_SAVE_SIZE), desc="DYNAMIC Optimization Progress", unit="batch"):
        batch_results, prev_final_soc_dynamic, prev_batch_powers_dynamic = optimize_energy(
            start_t, "DYNAMIC", prev_final_soc_dynamic, prev_batch_powers_dynamic)
        if batch_results:
            all_results_dynamic.extend(batch_results[:OVERLAP_SAVE_SIZE])
        else:
            failed_batches_dynamic.append(start_t)

    # Print failure summary
    print(f"❌ FIXED Strategy Failures: {len(failed_batches_fixed)} batches failed.")
    print(f"✅ FIXED Strategy Success: {100 * (1 - len(failed_batches_fixed) / ((time_intervals - OVERLAP_BATCH_SIZE) / OVERLAP_SAVE_SIZE)):.2f}%")

    print(f"❌ DYNAMIC Strategy Failures: {len(failed_batches_dynamic)} batches failed.")
    print(f"✅ DYNAMIC Strategy Success: {100 * (1 - len(failed_batches_dynamic) / ((time_intervals - OVERLAP_BATCH_SIZE) / OVERLAP_SAVE_SIZE)):.2f}%")


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
    version = "v12"  # Update version as needed
    results_fixed_df.to_csv(f"WorkingCodeVersion1_FIXED_{version}.csv", index=False)
    results_dynamic_df.to_csv(f"WorkingCodeVersion1_DYNAMIC_{version}.csv", index=False)
    homer_df.to_csv(f"WorkingCodeVersion1_HOMER_{version}.csv", index=False)

    print("✅ Results saved successfully!")

    # ✅ Run Plotting
    #print("📊 Generating plots...")
    #subprocess.run(["python", "plot_results.py"])

    total_time = time.time() - start_time
    print(f"⏳ Total Execution Time: {total_time:.2f} seconds")
