import os
import time
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
LOOK_AHEAD_WINDOW = 10
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



def optimize_energy1(start_t, STRATEGY):
    """Runs optimization for a batch of intervals."""

    # Initialize prev_final_soc before the first batch
    if start_t == 0:
        prev_final_soc = 0.05 * battery_capacity  # First batch starts at 5% SOC
    else:
        prev_final_soc = None  # Will be updated after the first batch

    model = LpProblem("Energy_Cost_Minimization", LpMinimize)
    end_t = min(start_t + BATCH_SIZE, len(df) - LOOK_AHEAD_WINDOW)

    inverse_eta_dis = 1 / eta_dis

    # Pricing thresholds
    CHEAP_IMPORT_PRICE = df["total_consumption_rate"].quantile(0.25)
    EXPENSIVE_IMPORT_PRICE = df["total_consumption_rate"].quantile(0.75)
    CHEAP_EXPORT_PRICE = df["grid_sellback_rate"].quantile(0.25)
    EXPENSIVE_EXPORT_PRICE = df["grid_sellback_rate"].quantile(0.75)

    # Tuning parameters
    grid_import_penalty = 5.0
    battery_discharge_bonus = 1.0

    # Define variables
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

        Delta_P_import[t] = LpVariable(f"Delta_P_import_{t}", lowBound=0)
        Delta_P_export[t] = LpVariable(f"Delta_P_export_{t}", lowBound=0)
        Delta_P_bat_ch[t] = LpVariable(f"Delta_P_bat_ch_{t}", lowBound=0)
        Delta_P_bat_dis[t] = LpVariable(f"Delta_P_bat_dis_{t}", lowBound=0)

    # Objective function
    model += lpSum(
        P_import[t] * (FIXED_IMPORT_PRICE + grid_import_penalty if STRATEGY == "FIXED" else df["total_consumption_rate"].iloc[t] + grid_import_penalty)
        - P_export[t] * (FIXED_EXPORT_PRICE if STRATEGY == "FIXED" else df["grid_sellback_rate"].iloc[t])
        - battery_discharge_bonus * P_bat_dis[t]
        for t in range(start_t, end_t)
    )

    for t in range(start_t, end_t):
        available_renewable = df["dc_ground_1500vdc_power_output"].iloc[t] + df["windflow_33_[500kw]_power_output"].iloc[t]

        model += P_import[t] <= df["ac_primary_load"].iloc[t] - available_renewable

        if df["total_consumption_rate"].iloc[t] <= CHEAP_IMPORT_PRICE:
            model += P_bat_ch[t] <= battery_max_charge
        else:
            model += P_bat_ch[t] <= battery_max_charge * 0.1

        model += P_export[t] <= inverter_capacity
        model += P_export[t] <= P_bat_dis[t]
        model += available_renewable + P_bat_dis[t] + P_import[t] >= df["ac_primary_load"].iloc[t] + P_bat_ch[t] + P_export[t]
        model += P_bat_dis[t] <= df["ac_primary_load"].iloc[t]
        model += P_bat_ch[t] + P_bat_dis[t] <= inverter_capacity

        model += X_bat_ch[t] + X_bat_dis[t] <= 1
        model += P_bat_ch[t] <= X_bat_ch[t] * battery_max_charge
        model += P_bat_dis[t] <= X_bat_dis[t] * battery_max_discharge

        model += X_import[t] + X_export[t] <= 1
        model += P_import[t] <= X_import[t] * inverter_capacity
        model += P_export[t] <= X_export[t] * inverter_capacity

        model += P_import[t] >= 0
        model += P_export[t] >= 0
        model += P_bat_ch[t] >= 0
        model += P_bat_dis[t] >= 0

        if t == start_t:
            model += SOC[t] == prev_final_soc
        else:
            model += SOC[t] == SOC[t - 1] + (P_bat_ch[t] * eta_ch) - (P_bat_dis[t] * inverse_eta_dis)

            model += Delta_P_import[t] >= P_import[t] - P_import[t - 1]
            model += Delta_P_import[t] >= P_import[t - 1] - P_import[t]
            model += Delta_P_import[t] <= max_change_limit

            model += Delta_P_export[t] >= P_export[t] - P_export[t - 1]
            model += Delta_P_export[t] >= P_export[t - 1] - P_export[t]
            model += Delta_P_export[t] <= max_change_limit

            model += Delta_P_bat_ch[t] >= P_bat_ch[t] - P_bat_ch[t - 1]
            model += Delta_P_bat_ch[t] >= P_bat_ch[t - 1] - P_bat_ch[t]
            model += Delta_P_bat_ch[t] <= max_battery_rate_change

            model += Delta_P_bat_dis[t] >= P_bat_dis[t] - P_bat_dis[t - 1]
            model += Delta_P_bat_dis[t] >= P_bat_dis[t - 1] - P_bat_dis[t]
            model += Delta_P_bat_dis[t] <= max_battery_rate_change

    model.solve(PULP_CBC_CMD(msg=0))

    if (end_t - 1) in SOC:
        prev_final_soc = SOC[end_t - 1].varValue
    else:
        prev_final_soc = soc_min

    results = [{
        "time": df["time"].iloc[t],
        "P_import (kW)": P_import[t].varValue,
        "P_export (kW)": P_export[t].varValue,
        "P_bat_ch (kW)": P_bat_ch[t].varValue,
        "P_bat_dis (kW)": P_bat_dis[t].varValue,
        "SOC (%)": (SOC[t].varValue / battery_capacity) * 100
    } for t in range(start_t, end_t)]

    return results

def optimize_energy2(start_t, STRATEGY):
    """Runs optimization for a batch of intervals."""

    global prev_final_soc  # Ensure we track SOC across batches

    if start_t != 0 and prev_final_soc is None:
        raise ValueError(f"❌ prev_final_soc is undefined for batch starting at interval {start_t}")

    if start_t == 0:
        prev_final_soc = 0.05 * battery_capacity  # First batch starts at 5% SOC

    model = LpProblem("Energy_Cost_Minimization", LpMinimize)
    end_t = min(start_t + BATCH_SIZE, len(df) - LOOK_AHEAD_WINDOW)

    inverse_eta_dis = 1 / eta_dis

    # Pricing thresholds
    CHEAP_IMPORT_PRICE = df["total_consumption_rate"].quantile(0.25)
    EXPENSIVE_IMPORT_PRICE = df["total_consumption_rate"].quantile(0.75)
    CHEAP_EXPORT_PRICE = df["grid_sellback_rate"].quantile(0.25)
    EXPENSIVE_EXPORT_PRICE = df["grid_sellback_rate"].quantile(0.75)

    # Tuning parameters
    grid_import_penalty = 5.0
    battery_discharge_bonus = 1.0

    # Define variables
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

        Delta_P_import[t] = LpVariable(f"Delta_P_import_{t}", lowBound=0)
        Delta_P_export[t] = LpVariable(f"Delta_P_export_{t}", lowBound=0)
        Delta_P_bat_ch[t] = LpVariable(f"Delta_P_bat_ch_{t}", lowBound=0)
        Delta_P_bat_dis[t] = LpVariable(f"Delta_P_bat_dis_{t}", lowBound=0)

    # Objective function
    model += lpSum(
        P_import[t] * (FIXED_IMPORT_PRICE + grid_import_penalty if STRATEGY == "FIXED" else df["total_consumption_rate"].iloc[t] + grid_import_penalty)
        - P_export[t] * (FIXED_EXPORT_PRICE if STRATEGY == "FIXED" else df["grid_sellback_rate"].iloc[t])
        - battery_discharge_bonus * P_bat_dis[t]
        for t in range(start_t, end_t)
    )

    for t in range(start_t, end_t):
        available_renewable = df["dc_ground_1500vdc_power_output"].iloc[t] + df["windflow_33_[500kw]_power_output"].iloc[t]

        model += P_import[t] <= df["ac_primary_load"].iloc[t] - available_renewable

        if df["total_consumption_rate"].iloc[t] <= CHEAP_IMPORT_PRICE:
            model += P_bat_ch[t] <= battery_max_charge
        else:
            model += P_bat_ch[t] <= battery_max_charge * 0.1

        model += P_export[t] <= inverter_capacity
        model += P_export[t] <= P_bat_dis[t]
        model += available_renewable + P_bat_dis[t] + P_import[t] >= df["ac_primary_load"].iloc[t] + P_bat_ch[t] + P_export[t]
        model += P_bat_dis[t] <= df["ac_primary_load"].iloc[t]
        model += P_bat_ch[t] + P_bat_dis[t] <= inverter_capacity

        model += X_bat_ch[t] + X_bat_dis[t] <= 1
        model += P_bat_ch[t] <= X_bat_ch[t] * battery_max_charge
        model += P_bat_dis[t] <= X_bat_dis[t] * battery_max_discharge

        model += X_import[t] + X_export[t] <= 1
        model += P_import[t] <= X_import[t] * inverter_capacity
        model += P_export[t] <= X_export[t] * inverter_capacity

        model += P_import[t] >= 0
        model += P_export[t] >= 0
        model += P_bat_ch[t] >= 0
        model += P_bat_dis[t] >= 0

        if t == start_t:
            if prev_final_soc is None:
                prev_final_soc = 0.05 * battery_capacity
            model += SOC[t] == prev_final_soc
        else:
            model += SOC[t] == SOC[t - 1] + (P_bat_ch[t] * eta_ch) - (P_bat_dis[t] * inverse_eta_dis)

            # Smoothing constraints
            model += Delta_P_import[t] >= P_import[t] - P_import[t - 1]
            model += Delta_P_import[t] >= P_import[t - 1] - P_import[t]
            model += Delta_P_import[t] <= max_change_limit

            model += Delta_P_export[t] >= P_export[t] - P_export[t - 1]
            model += Delta_P_export[t] >= P_export[t - 1] - P_export[t]
            model += Delta_P_export[t] <= max_change_limit

            model += Delta_P_bat_ch[t] >= P_bat_ch[t] - P_bat_ch[t - 1]
            model += Delta_P_bat_ch[t] >= P_bat_ch[t - 1] - P_bat_ch[t]
            model += Delta_P_bat_ch[t] <= max_battery_rate_change

            model += Delta_P_bat_dis[t] >= P_bat_dis[t] - P_bat_dis[t - 1]
            model += Delta_P_bat_dis[t] >= P_bat_dis[t - 1] - P_bat_dis[t]
            model += Delta_P_bat_dis[t] <= max_battery_rate_change

    model.solve(PULP_CBC_CMD(msg=0))

    if (end_t - 1) in SOC and SOC[end_t - 1].varValue is not None:
        prev_final_soc = SOC[end_t - 1].varValue
    else:
        prev_final_soc = soc_min

    results = [{
        "time": df["time"].iloc[t],
        "P_import (kW)": max(P_import[t].varValue, 0),
        "P_export (kW)": max(P_export[t].varValue, 0),
        "P_bat_ch (kW)": max(P_bat_ch[t].varValue, 0),
        "P_bat_dis (kW)": max(P_bat_dis[t].varValue, 0),
        "SOC (%)": (SOC[t].varValue / battery_capacity) * 100
    } for t in range(start_t, end_t)]

    return results

def optimize_energy(start_t, STRATEGY, prev_final_soc):
    """Runs optimization for a batch of intervals."""

    model = LpProblem("Energy_Cost_Minimization", LpMinimize)
    end_t = min(start_t + BATCH_SIZE, len(df) - LOOK_AHEAD_WINDOW)

    inverse_eta_dis = 1 / eta_dis

    # Pricing thresholds
    CHEAP_IMPORT_PRICE = df["total_consumption_rate"].quantile(0.25)
    EXPENSIVE_IMPORT_PRICE = df["total_consumption_rate"].quantile(0.75)
    CHEAP_EXPORT_PRICE = df["grid_sellback_rate"].quantile(0.25)
    EXPENSIVE_EXPORT_PRICE = df["grid_sellback_rate"].quantile(0.75)

    # Tuning parameters
    grid_import_penalty = 5.0
    battery_discharge_bonus = 1.0

    # Define variables
    P_import, P_export, P_bat_ch, P_bat_dis, SOC = {}, {}, {}, {}, {}
    Delta_P_import, Delta_P_export, Delta_P_bat_ch, Delta_P_bat_dis = {}, {}, {}, {}

    for t in range(start_t, end_t):
        P_import[t] = LpVariable(f"P_import_{t}", 0, inverter_capacity)
        P_export[t] = LpVariable(f"P_export_{t}", 0, inverter_capacity)
        P_bat_ch[t] = LpVariable(f"P_bat_ch_{t}", 0, battery_max_charge)
        P_bat_dis[t] = LpVariable(f"P_bat_dis_{t}", 0, battery_max_discharge)
        SOC[t] = LpVariable(f"SOC_{t}", soc_min, soc_max)

        if t > start_t:
            Delta_P_import[t] = LpVariable(f"Delta_P_import_{t}", lowBound=0)
            Delta_P_export[t] = LpVariable(f"Delta_P_export_{t}", lowBound=0)
            Delta_P_bat_ch[t] = LpVariable(f"Delta_P_bat_ch_{t}", lowBound=0)
            Delta_P_bat_dis[t] = LpVariable(f"Delta_P_bat_dis_{t}", lowBound=0)

    # Objective function
    model += lpSum(
        P_import[t] * (FIXED_IMPORT_PRICE + grid_import_penalty if STRATEGY == "FIXED" else df["total_consumption_rate"].iloc[t] + grid_import_penalty)
        - P_export[t] * (FIXED_EXPORT_PRICE if STRATEGY == "FIXED" else df["grid_sellback_rate"].iloc[t])
        - battery_discharge_bonus * P_bat_dis[t]
        for t in range(start_t, end_t)
    )

    for t in range(start_t, end_t):
        available_renewable = df["dc_ground_1500vdc_power_output"].iloc[t] + df["windflow_33_[500kw]_power_output"].iloc[t]

        model += P_import[t] <= df["ac_primary_load"].iloc[t] - available_renewable

        if df["total_consumption_rate"].iloc[t] <= CHEAP_IMPORT_PRICE:
            model += P_bat_ch[t] <= battery_max_charge
        else:
            model += P_bat_ch[t] <= battery_max_charge * 0.1

        model += P_export[t] <= inverter_capacity
        model += P_export[t] <= P_bat_dis[t]
        model += available_renewable + P_bat_dis[t] + P_import[t] >= df["ac_primary_load"].iloc[t] + P_bat_ch[t] + P_export[t]
        model += P_bat_dis[t] <= df["ac_primary_load"].iloc[t]
        model += P_bat_ch[t] + P_bat_dis[t] <= inverter_capacity

        model += P_import[t] >= 0
        model += P_export[t] >= 0
        model += P_bat_ch[t] >= 0
        model += P_bat_dis[t] >= 0

        if t == start_t:
            model += SOC[t] == prev_final_soc
        else:
            model += SOC[t] == SOC[t - 1] + (P_bat_ch[t] * eta_ch) - (P_bat_dis[t] * inverse_eta_dis)

            # Smoothing constraints
            model += Delta_P_import[t] >= P_import[t] - P_import[t - 1]
            model += Delta_P_import[t] >= P_import[t - 1] - P_import[t]
            model += Delta_P_import[t] <= max_change_limit

            model += Delta_P_export[t] >= P_export[t] - P_export[t - 1]
            model += Delta_P_export[t] >= P_export[t - 1] - P_export[t]
            model += Delta_P_export[t] <= max_change_limit

            model += Delta_P_bat_ch[t] >= P_bat_ch[t] - P_bat_ch[t - 1]
            model += Delta_P_bat_ch[t] >= P_bat_ch[t - 1] - P_bat_ch[t]
            model += Delta_P_bat_ch[t] <= max_battery_rate_change

            model += Delta_P_bat_dis[t] >= P_bat_dis[t] - P_bat_dis[t - 1]
            model += Delta_P_bat_dis[t] >= P_bat_dis[t - 1] - P_bat_dis[t]
            model += Delta_P_bat_dis[t] <= max_battery_rate_change

    model.solve(PULP_CBC_CMD(msg=0))

    if (end_t - 1) in SOC and SOC[end_t - 1].varValue is not None:
        prev_final_soc = SOC[end_t - 1].varValue
    else:
        prev_final_soc = soc_min

    results = [{
        "time": df["time"].iloc[t],
        "P_import (kW)": max(P_import[t].varValue, 0),
        "P_export (kW)": max(P_export[t].varValue, 0),
        "P_bat_ch (kW)": max(P_bat_ch[t].varValue, 0),
        "P_bat_dis (kW)": max(P_bat_dis[t].varValue, 0),
        "SOC (%)": (SOC[t].varValue / battery_capacity) * 100
    } for t in range(start_t, end_t)]

    return results, prev_final_soc


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

    
    # =======================
    # SAVE RESULTS (Versioning)
    # =======================
    version = "v4"  # Update version as needed
    results_fixed_df.to_csv(f"WorkingCodeVersion1_FIXED_{version}.csv", index=False)
    results_dynamic_df.to_csv(f"WorkingCodeVersion1_DYNAMIC_{version}.csv", index=False)
    homer_df.to_csv(f"WorkingCodeVersion1_HOMER_{version}.csv", index=False)

    print("✅ Results saved successfully!")

    # ✅ Run Plotting
    #print("📊 Generating plots...")
    #subprocess.run(["python", "plot_results.py"])

    total_time = time.time() - start_time
    print(f"⏳ Total Execution Time: {total_time:.2f} seconds")
