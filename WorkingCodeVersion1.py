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
# =======================
def optimize_energy(start_t, STRATEGY):
    """ Runs optimization for a batch of intervals. """
    
    # Initialize prev_final_soc before the first batch
    if start_t == 0:
        prev_final_soc = 0.05 * battery_capacity  # First batch starts at 5% SOC
    else:
        prev_final_soc = None  # This will be updated after the first batch
    
    model = LpProblem("Energy_Cost_Minimization", LpMinimize)
    end_t = min(start_t + BATCH_SIZE, len(df) - LOOK_AHEAD_WINDOW)

    P_import = {t: LpVariable(f"P_import_{t}", lowBound=0, upBound=inverter_capacity) for t in range(start_t, end_t)}
    P_export = {t: LpVariable(f"P_export_{t}", lowBound=0, upBound=inverter_capacity) for t in range(start_t, end_t)}
    P_bat_ch = {t: LpVariable(f"P_bat_ch_{t}", lowBound=0, upBound=battery_max_charge) for t in range(start_t, end_t)}
    P_bat_dis = {t: LpVariable(f"P_bat_dis_{t}", lowBound=0, upBound=battery_max_discharge) for t in range(start_t, end_t)}
    SOC = {t: LpVariable(f"SOC_{t}", lowBound=soc_min, upBound=soc_max) for t in range(start_t, end_t)}


    X_import = {t: LpVariable(f"X_import_{t}", cat="Binary") for t in range(start_t, end_t)}
    X_export = {t: LpVariable(f"X_export_{t}", cat="Binary") for t in range(start_t, end_t)}
    X_bat_ch = {t: LpVariable(f"X_bat_ch_{t}", cat="Binary") for t in range(start_t, end_t)}
    X_bat_dis = {t: LpVariable(f"X_bat_dis_{t}", cat="Binary") for t in range(start_t, end_t)}

    # Tuning parameters: set high penalty and bonus
    grid_import_penalty = 5.0       # Strong penalty on grid imports
    battery_discharge_bonus = 1.0   # Strong bonus for battery discharge

   # Objective Function: Minimize total cost
    if STRATEGY == "FIXED":
        model += lpSum(
            P_import[t] * (FIXED_IMPORT_PRICE + grid_import_penalty) -
            P_export[t] * FIXED_EXPORT_PRICE -
            battery_discharge_bonus * P_bat_dis[t]
            for t in range(start_t, end_t)
        )
    elif STRATEGY == "DYNAMIC":
        model += lpSum(
            P_import[t] * (df["total_consumption_rate"].iloc[t] + grid_import_penalty) -
            P_export[t] * df["grid_sellback_rate"].iloc[t] -
            battery_discharge_bonus * P_bat_dis[t]
            for t in range(start_t, end_t)
        )

    
    # Smoothing auxiliary variables (to encourage gradual changes)
    Delta_P_import = {t: LpVariable(f"Delta_P_import_{t}", lowBound=0) for t in range(start_t, end_t)}
    Delta_P_export = {t: LpVariable(f"Delta_P_export_{t}", lowBound=0) for t in range(start_t, end_t)}
    Delta_P_bat_ch = {t: LpVariable(f"Delta_P_bat_ch_{t}", lowBound=0) for t in range(start_t, end_t)}
    Delta_P_bat_dis = {t: LpVariable(f"Delta_P_bat_dis_{t}", lowBound=0) for t in range(start_t, end_t)}
    
    for t in range(start_t, end_t):
        # Calculate available renewable energy at time t
        available_renewable = df["dc_ground_1500vdc_power_output"].iloc[t] + df["windflow_33_[500kw]_power_output"].iloc[t]
        
        # Grid Import Constraint:
        # Allow grid import only to cover the shortfall between load and available renewable.
        #model += P_import[t] <= max(0, df["ac_primary_load"].iloc[t] - available_renewable)
        model += P_import[t] <= df["ac_primary_load"].iloc[t] - available_renewable

        
        # Battery Charging: allow up to max capacity (no condition based on price)
        model += P_bat_ch[t] <= battery_max_charge
        
        # Battery Discharging: allow up to max capacity (no forced minimum)
        model += P_bat_dis[t] <= battery_max_discharge
        
        # Grid Export Constraint: allow export up to inverter capacity (no sellback condition)
        
        model += P_export[t] <= inverter_capacity
        model += P_export[t] <= P_bat_dis[t]  # Ensure we only export if we're actually discharging


        # Energy Balance Constraint: ensure supply (renewable + battery discharge + grid import)
        # meets load plus battery charging and export.
        #model += available_renewable + P_bat_dis[t] + P_import[t] >= df["ac_primary_load"].iloc[t] + P_bat_ch[t] + P_export[t]
        
        model += available_renewable + P_bat_dis[t] + P_import[t] >= df["ac_primary_load"].iloc[t] + P_bat_ch[t] + P_export[t]
        model += P_bat_dis[t] <= df["ac_primary_load"].iloc[t]  # Ensure discharge doesn't exceed load
        model += P_import[t] <= df["ac_primary_load"].iloc[t]  # Ensure grid import is realistic

        model += P_import[t] >=0
        model += P_export[t] >=0
        model += P_bat_dis[t] >= 0
        model += P_bat_ch[t] >= 0
        model += SOC[t] >= soc_min  # Ensure SOC never drops below minimum limit
        model += SOC[t] <= soc_max  # Ensure SOC does not exceed maximum capacity


        # ✅ Ensure battery is either charging or discharging, but not both
        model += X_bat_ch[t] + X_bat_dis[t] <= 1
        model += P_bat_ch[t] <= X_bat_ch[t] * battery_max_charge
        model += P_bat_dis[t] <= X_bat_dis[t] * battery_max_discharge

        # ✅ Ensure import and export do not happen simultaneously
        model += X_import[t] + X_export[t] <= 1
        model += P_import[t] <= X_import[t] * inverter_capacity
        model += P_export[t] <= X_export[t] * inverter_capacity


        # Battery SOC Dynamics
        """ if t == start_t:
            model += SOC[t] == 0.05 * battery_capacity  # Initial SOC
        else:
            model += SOC[t] == SOC[t - 1] + (P_bat_ch[t] * eta_ch) - (P_bat_dis[t] * (1 / eta_dis)) """
        
        
        if t == start_t:
            if start_t == 0:

                prev_final_soc = 0.05 * battery_capacity  # First batch starts at 5% SOC

                model += SOC[t] == 0.05 * battery_capacity  # First batch starts at 5% SOC
            else:
                model += SOC[t] == prev_final_soc  # Next batches use SOC from previous batch

        # Smoothing Constraints (for t > start_t)
            if t > start_t:
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
        
        # Ensure battery charge and discharge together do not exceed inverter capacity
        model += P_bat_ch[t] + P_bat_dis[t] <= inverter_capacity

        
        model += SOC[t] >= soc_min  # Prevent SOC from going too low

    
    # Solve the model using CBC
    model.solve(PULP_CBC_CMD(msg=0))
    if (end_t - 1) in SOC:
        prev_final_soc = SOC[end_t - 1].varValue
    else:
        prev_final_soc = soc_min  # Default to minimum SOC if key is invalid


    results = [{
        "time": df["time"].iloc[t],
        "P_import (kW)": P_import[t].varValue,
        "P_export (kW)": P_export[t].varValue,
        "P_bat_ch (kW)": P_bat_ch[t].varValue,
        "P_bat_dis (kW)": P_bat_dis[t].varValue,
        "SOC (%)": (SOC[t].varValue / battery_capacity) * 100
    } for t in range(start_t, end_t)]

    return results

# =======================
# RUN OPTIMIZATION
# =======================
if __name__ == "__main__":
    start_time = time.time()
    print("🚀 Running optimization for Fixed and Dynamic strategies...")

    #with multiprocessing.Pool(NUM_CORES) as pool:
    with multiprocessing.Pool(NUM_CORES, maxtasksperchild=1) as pool:

        all_results_fixed = pool.starmap(optimize_energy, [(i, "FIXED") for i in range(0, len(df) - LOOK_AHEAD_WINDOW, BATCH_SIZE)])
        all_results_dynamic = pool.starmap(optimize_energy, [(i, "DYNAMIC") for i in range(0, len(df) - LOOK_AHEAD_WINDOW, BATCH_SIZE)])

    # Convert results to DataFrame (Fixed & Dynamic)
    results_fixed_df = pd.DataFrame([item for sublist in all_results_fixed for item in sublist])
    results_dynamic_df = pd.DataFrame([item for sublist in all_results_dynamic for item in sublist])

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
    version = "v1"  # Update version as needed

    results_fixed_df.to_csv(f"WorkingCodeVersion1_FIXED_{version}.csv", index=False)
    results_dynamic_df.to_csv(f"WorkingCodeVersion1_DYNAMIC_{version}.csv", index=False)
    homer_df.to_csv(f"WorkingCodeVersion1_HOMER_{version}.csv", index=False)

    print("✅ Results saved successfully!")

    # ✅ Run Plotting
    print("📊 Generating plots...")
    subprocess.run(["python", "plot_results.py"])

    total_time = time.time() - start_time
    print(f"⏳ Total Execution Time: {total_time:.2f} seconds")
