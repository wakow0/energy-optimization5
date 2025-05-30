import os
import time
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
from gurobipy import Model, GRB, quicksum
import gurobipy as gp

BATCH_SIZE = 60
LOOK_AHEAD_WINDOW = 20
NUM_CORES = multiprocessing.cpu_count()
SMOOTHING_WINDOW = 20

FIXED_IMPORT_PRICE = 0.1939
FIXED_EXPORT_PRICE = 0.0769

battery_capacity = 2000
battery_max_charge = 1000
battery_max_discharge = 1000
inverter_capacity = 1000
soc_min = 0.05 * battery_capacity
soc_max = 1.0 * battery_capacity
eta_ch = 0.95
eta_dis = 0.95

max_change_limit = 300
max_battery_rate_change = 500
PV_capacity = 1500
Wind_capacity = 500
max_renewable_capacity = PV_capacity + Wind_capacity

min_discharge_value = 5
penalty_cost = 50
SOC_BUFFER = 0.98 * soc_max
OVERLAP = int(BATCH_SIZE * 0.25)

ENABLE_RAMP_CONTROL = True

# Define ramp rate limits (adjust these values based on your system's capabilities)

ramp_rate_import = 300  # kW per interval
ramp_rate_export = 300  # kW per interval

ramp_rate_bat_ch = 200  # kW per interval
ramp_rate_bat_dis = 200  # kW per interval

# Define penalty coefficients (adjust these values based on your system's requirements)

penalty_import = 0.5
penalty_export = 0.5
penalty_bat_ch = 0.5
penalty_bat_dis = 0.5

# Define maximum change limits (adjust these values based on your system's capabilities)
max_change_import = 500
max_change_export = 500
max_change_bat_ch = 300
max_change_bat_dis = 300



from data_parser import load_csv_data

df = pd.read_csv("processed_data.csv", dtype=str, low_memory=False)

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(":", "")
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df.dropna(subset=["time"], inplace=True)

expected_columns = [
    "time", "total_consumption_rate", "grid_sellback_rate", "ac_primary_load",
    "dc_ground_1500vdc_power_output", "windflow_33_[500kw]_power_output",
    "wattstor_m5_0.5c_september_charge_power", "grid_purchases", "grid_sales"
]
for col in expected_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

time_intervals = len(df)

fixed_batch_starts = list(range(0, time_intervals - BATCH_SIZE + 1, OVERLAP))
dynamic_batch_starts = list(range(0, time_intervals - BATCH_SIZE + 1, OVERLAP))


def optimize_energy(start_t, STRATEGY, prev_final_soc, prev_batch_powers):
    model = Model("Energy_Optimization")
    model.Params.OutputFlag = 0
    end_t = min(start_t + BATCH_SIZE, time_intervals)

    # Define battery efficiency and time interval
    charging_efficiency = 0.95  # 95% charging efficiency
    discharging_efficiency = 0.95  # 95% discharging efficiency
    time_interval = 0.5  # 1 hour time interval

    P_import, P_export, P_bat_ch, P_bat_dis, SOC = {}, {}, {}, {}, {}
    X_import, X_export, X_bat_ch, X_bat_dis, X_high_soc, X_curtail = {}, {}, {}, {}, {}, {}
    P_curtail = {}

    for t in range(start_t, end_t):
        P_import[t] = model.addVar(lb=0, ub=inverter_capacity)
        P_export[t] = model.addVar(lb=0, ub=inverter_capacity)
        P_bat_ch[t] = model.addVar(lb=0, ub=battery_max_charge)
        P_bat_dis[t] = model.addVar(lb=0, ub=battery_max_discharge)
        SOC[t] = model.addVar(lb=soc_min, ub=soc_max)
        P_curtail[t] = model.addVar(lb=0, ub=max_renewable_capacity)
        X_import[t] = model.addVar(vtype=GRB.BINARY)
        X_export[t] = model.addVar(vtype=GRB.BINARY)
        X_bat_ch[t] = model.addVar(vtype=GRB.BINARY)
        X_bat_dis[t] = model.addVar(vtype=GRB.BINARY)
        X_high_soc[t] = model.addVar(vtype=GRB.BINARY)
        X_curtail[t] = model.addVar(vtype=GRB.BINARY)

    # Set initial SOC
    if start_t == 0:
        model.addConstr(SOC[start_t] == soc_min)
    else:
        model.addConstr(SOC[start_t] == prev_final_soc)

    # if start_t == 0:
    #     SOC[start_t] = prev_final_soc
    # else:
    #     SOC[start_t] = prev_final_soc

    # Add SOC continuity constraints
    for t in range(start_t + 1, end_t):
        model.addConstr(SOC[t] == SOC[t - 1] +
                       (P_bat_ch[t] * charging_efficiency - P_bat_dis[t] / discharging_efficiency) * time_interval)

    # Add battery usage incentives to the objective function
    target_soc = 90  # Target SOC in percentage
    soc_penalty = 0.9  # Penalty for deviations from the target SOC

    model.setObjective(
        quicksum(
            P_import[t] * (FIXED_IMPORT_PRICE if STRATEGY == "FIXED" else df["total_consumption_rate"].iloc[t]) -
            P_export[t] * (FIXED_EXPORT_PRICE if STRATEGY == "FIXED" else df["grid_sellback_rate"].iloc[t]) +
            penalty_cost * P_curtail[t] +
            soc_penalty * ((SOC[t] / battery_capacity * 100) - target_soc)**2  # Penalize deviations from target SOC
            for t in range(start_t, end_t)
        ), GRB.MINIMIZE
    )


    

    # Add other constraints
    for t in range(start_t, end_t):
        available_renewable = df["dc_ground_1500vdc_power_output"].iloc[t] + df["windflow_33_[500kw]_power_output"].iloc[t]
        demand = df["ac_primary_load"].iloc[t]

        model.addConstr(available_renewable + P_import[t] + P_bat_dis[t] == demand + P_bat_ch[t] + P_export[t] + P_curtail[t])

        model.addConstr(P_import[t] <= X_import[t] * inverter_capacity)
        model.addConstr(P_export[t] <= X_export[t] * inverter_capacity)
        model.addConstr(X_import[t] + X_export[t] <= 1)
        model.addConstr(P_bat_ch[t] <= X_bat_ch[t] * battery_max_charge)
        model.addConstr(P_bat_dis[t] <= X_bat_dis[t] * battery_max_discharge)
        model.addConstr(X_bat_ch[t] + X_bat_dis[t] <= 1)
        model.addConstr(P_import[t] <= inverter_capacity * (1 - X_curtail[t]))
        model.addConstr(P_curtail[t] <= max_renewable_capacity * X_curtail[t])
        model.addConstr(SOC[t] - SOC_BUFFER >= -soc_max * (1 - X_high_soc[t]))
        model.addConstr(SOC[t] - SOC_BUFFER <= soc_max * X_high_soc[t])
        model.addConstr(P_bat_dis[t] >= min_discharge_value * X_high_soc[t])

            





    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"❌ Batch failed at start interval {start_t}")
        #prev_final_soc = min(max(SOC[end_t - 1].X, soc_min), SOC_BUFFER)
        return None, prev_final_soc, prev_batch_powers, False

    prev_final_soc = min(max(SOC[end_t - 1].X, soc_min), SOC_BUFFER)
        
    batch_results = [{
        "time": str(df["time"].iloc[t]),
        "P_import (kW)": max(P_import[t].X, 0),
        "P_export (kW)": max(P_export[t].X, 0),
        "P_bat_ch (kW)": max(P_bat_ch[t].X, 0),
        "P_bat_dis (kW)": max(P_bat_dis[t].X, 0),
        "P_curtail (kW)": max(P_curtail[t].X, 0),
        "SOC (%)": max((SOC[t].X if isinstance(SOC[t], gp.Var) else SOC[t]) / battery_capacity * 100, 0)

    } for t in range(start_t, end_t)]

    final_powers = {
        "P_import": P_import[end_t - 1].X,
        "P_export": P_export[end_t - 1].X,
        "P_bat_ch": P_bat_ch[end_t - 1].X,
        "P_bat_dis": P_bat_dis[end_t - 1].X
    }

    print(f"✅ Batch succeeded at start interval {start_t}")
    return batch_results, prev_final_soc, final_powers, True






if __name__ == "__main__":
    start_time = time.time()
    print("🚀 Running optimization for Fixed and Dynamic strategies...")

    all_results_fixed = []
    all_results_dynamic = []

    prev_final_soc_fixed = soc_min
    prev_final_soc_dynamic = soc_min

    prev_batch_powers_fixed = {"P_import": 0, "P_export": 0, "P_bat_ch": 0, "P_bat_dis": 0}
    prev_batch_powers_dynamic = {"P_import": 0, "P_export": 0, "P_bat_ch": 0, "P_bat_dis": 0}

    failed_batches_fixed = []
    failed_batches_dynamic = []

    # Fixed strategy optimization
    for start_t in tqdm(fixed_batch_starts, desc="FIXED Optimization Progress", unit="batch"):
        batch_results, prev_final_soc_fixed, prev_batch_powers_fixed, success = optimize_energy(
            start_t, "FIXED", prev_final_soc_fixed, prev_batch_powers_fixed)
        if success:
            all_results_fixed.extend(batch_results[:OVERLAP])
        else:
            failed_batches_fixed.append(start_t)

    # Dynamic strategy optimization
    for start_t in tqdm(dynamic_batch_starts, desc="DYNAMIC Optimization Progress", unit="batch"):
        batch_results, prev_final_soc_dynamic, prev_batch_powers_dynamic, success = optimize_energy(
            start_t, "DYNAMIC", prev_final_soc_dynamic, prev_batch_powers_dynamic)
        if success:
            all_results_dynamic.extend(batch_results[:OVERLAP])
        else:
            failed_batches_dynamic.append(start_t)

    # Print success/failure statistics
    print(f"❌ FIXED Strategy Failures: {len(failed_batches_fixed)} batches failed.")
    print(f"✅ FIXED Strategy Success: {100 * (1 - len(failed_batches_fixed) / (len(fixed_batch_starts))):.2f}%")

    print(f"❌ DYNAMIC Strategy Failures: {len(failed_batches_dynamic)} batches failed.")
    print(f"✅ DYNAMIC Strategy Success: {100 * (1 - len(failed_batches_dynamic) / (len(dynamic_batch_starts))):.2f}%")

    # Convert results to DataFrames
    results_fixed_df = pd.DataFrame(all_results_fixed)
    results_dynamic_df = pd.DataFrame(all_results_dynamic)

    print("✅ Start to Save Results...!")
    
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

         

    # Ensure all DataFrames have the same length
    min_length = min(len(results_fixed_df), len(results_dynamic_df), len(df))
    results_fixed_df = results_fixed_df.iloc[:min_length].reset_index(drop=True)
    results_dynamic_df = results_dynamic_df.iloc[:min_length].reset_index(drop=True)
    homer_df = homer_df.iloc[:min_length].reset_index(drop=True)

    # Convert the time column to a human-readable format
    if 'time' in results_fixed_df.columns:
        # Convert the time column to numeric (float or int) before division
        results_fixed_df['time'] = pd.to_numeric(results_fixed_df['time'], errors='coerce')
        # Convert nanoseconds to seconds by dividing by 1e9
        results_fixed_df['time'] = pd.to_datetime(results_fixed_df['time'] / 1e9, unit='s')

    if 'time' in results_dynamic_df.columns:
        # Convert the time column to numeric (float or int) before division
        results_dynamic_df['time'] = pd.to_numeric(results_dynamic_df['time'], errors='coerce')
        # Convert nanoseconds to seconds by dividing by 1e9
        results_dynamic_df['time'] = pd.to_datetime(results_dynamic_df['time'] / 1e9, unit='s')

    if 'time' in homer_df.columns:
        # Convert the time column to numeric (float or int) before division
        homer_df['time'] = pd.to_numeric(homer_df['time'], errors='coerce')
        # Convert nanoseconds to seconds by dividing by 1e9
        homer_df['time'] = pd.to_datetime(homer_df['time'] / 1e9, unit='s')

    # Save results to CSV files
    version = "v10_7"
    results_fixed_df.to_csv(f"WorkingCodeVersion1_FIXED_{version}.csv", index=False)
    results_dynamic_df.to_csv(f"WorkingCodeVersion1_DYNAMIC_{version}.csv", index=False)
    homer_df.to_csv(f"WorkingCodeVersion1_HOMER_{version}.csv", index=False)

    print("✅ Results saved successfully!")
    total_time = time.time() - start_time
    print(f"⏳ Total Execution Time: {total_time:.2f} seconds")