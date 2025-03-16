import os
import time
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
from gurobipy import Model, GRB, quicksum

BATCH_SIZE = 48
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
#SOC_BUFFER = 0.98 * soc_max
SOC_BUFFER = 0.96 * soc_max  # Allow SOC to drop below 98%

OVERLAP = int(BATCH_SIZE * 0.5)

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

    model.setObjective(quicksum(
        P_import[t] * (FIXED_IMPORT_PRICE if STRATEGY == "FIXED" else df["total_consumption_rate"].iloc[t]) -
        P_export[t] * (FIXED_EXPORT_PRICE if STRATEGY == "FIXED" else df["grid_sellback_rate"].iloc[t]) +
        penalty_cost * P_curtail[t] for t in range(start_t, end_t)
    ), GRB.MINIMIZE)


    time_step = 0.5  # 30 minutes (half an hour)
    for t in range(start_t, end_t):
        available_renewable = df["dc_ground_1500vdc_power_output"].iloc[t] + df["windflow_33_[500kw]_power_output"].iloc[t]
        demand = df["ac_primary_load"].iloc[t]
        model.addConstr(available_renewable + P_import[t] + P_bat_dis[t] == demand + P_bat_ch[t] + P_export[t] + P_curtail[t])
        model.addConstr(P_import[t] <= X_import[t] * inverter_capacity)
        model.addConstr(P_export[t] <= X_export[t] * inverter_capacity)
        model.addConstr(X_import[t] + X_export[t] <= 1.1)
        model.addConstr(P_bat_ch[t] <= X_bat_ch[t] * battery_max_charge)
        model.addConstr(P_bat_dis[t] <= X_bat_dis[t] * battery_max_discharge)
        model.addConstr(X_bat_ch[t] + X_bat_dis[t] <= 1)
        model.addConstr(P_import[t] <= inverter_capacity * (1 - X_curtail[t]))
        model.addConstr(P_curtail[t] <= max_renewable_capacity * X_curtail[t])
        
        #time_step = 0.5  # 30-minute intervals
        #for t in range(start_t, end_t):
        if t == start_t:
            model.addConstr(SOC[t] == prev_final_soc)  # Maintain batch continuity
        else:
            model.addConstr(
                SOC[t] == SOC[t - 1] + (P_bat_ch[t] * eta_ch * time_step) - (P_bat_dis[t] * time_step / eta_dis)
            )

        
    model.Params.NumericFocus = 3  # Stronger numerical stability
    model.Params.IntFeasTol = 1e-6  # Improve integer feasibility
    model.Params.DualReductions = 0  # Prevent solver from assuming infeasibility
    model.Params.Presolve = 2  # Use aggressive presolve to simplify constraints
    model.Params.ScaleFlag = 2  # Improves numerical scaling

    model.Params.InfUnbdInfo = 1  # Detects which constraints are causing issues
    model.Params.FeasRelaxBigM = 1e3  # Adds flexibility to infeasible constraints
    model.Params.FeasibilityTol = 1e-6  # Allow minor violations
    model.Params.IntFeasTol = 1e-6  # Tolerate small integer feasibility errors

    model.Params.IterationLimit = 1e7  # Allow 10 million iterations per batch
    model.Params.BarIterLimit = 10000  # Increase barrier iterations
    model.Params.TimeLimit = 120  # Allow up to 2 minutes per batch

    model.Params.NumericFocus = 3  # Strongest numerical stability
    model.Params.ScaleFlag = 2  # Improve model scaling
    model.Params.FeasibilityTol = 1e-6  # Allow minor constraint relaxations
    model.Params.IntFeasTol = 1e-6  # Reduce strict integer feasibility requirements
    model.Params.DualReductions = 0  # Prevent solver from assuming infeasibility
    

    model.optimize()

    if model.status == GRB.OPTIMAL:
        batch_results = [{
            "time": str(df["time"].iloc[t]),
            "P_import (kW)": P_import[t].X,
            "P_export (kW)": P_export[t].X,
            "P_bat_ch (kW)": P_bat_ch[t].X,
            "P_bat_dis (kW)": P_bat_dis[t].X,
            "P_curtail (kW)": P_curtail[t].X,
            "SOC (%)": (SOC[t].X / battery_capacity) * 100
        } for t in range(start_t, end_t)]
    else:
        print(f"⚠️ Skipping batch {start_t} due to solver failure (Status: {model.status})")
        return None, prev_final_soc, prev_batch_powers, False  # Prevents accessing `.X`


        return batch_results, prev_final_soc, prev_batch_powers, True

    
    
    if model.status != GRB.OPTIMAL:
        print(f"⚠️ Batch {start_t} failed. Solver Status: {model.status}")
        print(f"📊 SOC: {prev_final_soc}, Import: {prev_batch_powers['P_import']}, Export: {prev_batch_powers['P_export']}, Battery Charge: {prev_batch_powers['P_bat_ch']}, Battery Discharge: {prev_batch_powers['P_bat_dis']}")


        if model.status == GRB.INFEASIBLE:
            print("🚨 The model is INFEASIBLE. Checking constraint conflicts...")
            model.computeIIS()
            model.write(f"infeasible_batch_{start_t}.ilp")

        elif model.status == GRB.UNBOUNDED:
            print("❌ The model is UNBOUNDED. Writing LP model...")
            model.write(f"unbounded_batch_{start_t}.lp")

        elif model.status == GRB.INF_OR_UNBD:
            print("⚠️ The model is INFEASIBLE or UNBOUNDED. Investigate constraints.")
            model.write(f"inf_or_unbd_batch_{start_t}.lp")

        elif model.status == GRB.ITERATION_LIMIT:
            print("⚠️ The solver stopped due to reaching the iteration limit!")

        elif model.status == GRB.NUMERIC:
            print("⚠️ The solver encountered numerical issues!")

        elif model.status == GRB.TIME_LIMIT:
            print("⚠️ The solver stopped due to time limit!")

        return None, prev_final_soc, prev_batch_powers, False


    prev_final_soc = min(max(SOC[end_t - 1].X, soc_min), SOC_BUFFER)

    batch_results = [{
            "time": str(df["time"].iloc[t]),
            "P_import (kW)": max(P_import[t].X, 0),
            "P_export (kW)": max(P_export[t].X, 0),
            "P_bat_ch (kW)": max(P_bat_ch[t].X, 0),
            "P_bat_dis (kW)": max(P_bat_dis[t].X, 0),
            "P_curtail (kW)": max(P_curtail[t].X, 0),
            "SOC (%)": max((SOC[t].X / battery_capacity) * 100, 0)
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

    result_columns = ["time", "P_import (kW)", "P_export (kW)", "P_bat_ch (kW)", "P_bat_dis (kW)", "SOC (%)"]
    results_fixed_df = results_fixed_df[result_columns]
    results_dynamic_df = results_dynamic_df[result_columns]
    homer_df = homer_df[result_columns]

    results_fixed_df["time"] = results_fixed_df["time"].astype(str)
    results_dynamic_df["time"] = results_dynamic_df["time"].astype(str)
    homer_df["time"] = homer_df["time"].astype(str)

    
    
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
    version = "v10_6"
    results_fixed_df.to_csv(f"WorkingCodeVersion1_FIXED_{version}.csv", index=False)
    results_dynamic_df.to_csv(f"WorkingCodeVersion1_DYNAMIC_{version}.csv", index=False)
    homer_df.to_csv(f"WorkingCodeVersion1_HOMER_{version}.csv", index=False)

    print("✅ Results saved successfully!")
    total_time = time.time() - start_time
    print(f"⏳ Total Execution Time: {total_time:.2f} seconds")