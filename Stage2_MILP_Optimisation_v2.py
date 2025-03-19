import os
import time
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
from gurobipy import Model, GRB, quicksum

# =======================
# CONFIGURABLE PARAMETERS
# =======================
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
max_power_change_per_batch = 1000  # kW
max_battery_rate_change = 200
PV_capacity = 1500  # kWp
Wind_capacity = 500  # kW
max_renewable_capacity = PV_capacity + Wind_capacity  # 2000 kW

min_discharge_value = 5  # kW
penalty_cost = 50  # Curtailment penalty
max_power_change_per_batch = 1000  # kW
SOC_BUFFER = 0.98 * soc_max
OVERLAP = int(BATCH_SIZE * 0.5)

# Load initial rule-based solution
#initial_solution = pd.read_csv("initial_solution.csv")
initial_solution = pd.read_csv("solution_output_FIXED_v5.csv")

initial_solution["time"] = pd.to_datetime(initial_solution["time"])
initial_solution.set_index("time", inplace=True)

# Debug: Check initial solution
print("Initial Solution Head:")
print(initial_solution.head())

# =======================
# LOAD DATA
# =======================
df = pd.read_csv("processed_data.csv", dtype=str, low_memory=False)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(":", "")
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df.dropna(subset=["time"], inplace=True)

# Debug: Check processed data
print("Processed Data Head:")
print(df.head())

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
    inverse_eta_dis = 1 / eta_dis

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

        # Warm start using initial solution
        time_t = df["time"].iloc[t]
        if time_t in initial_solution.index:
            P_import[t].Start = initial_solution.loc[time_t, "P_import (kW)"]
            P_export[t].Start = initial_solution.loc[time_t, "P_export (kW)"]
            P_bat_ch[t].Start = initial_solution.loc[time_t, "P_bat_ch (kW)"]
            P_bat_dis[t].Start = initial_solution.loc[time_t, "P_bat_dis (kW)"]
            SOC[t].Start = initial_solution.loc[time_t, "SOC (%)"] * battery_capacity / 100
            print(f"Warm start applied at time {time_t}")
        else:
            print(f"No warm start data for time {time_t}")

    model.setObjective(quicksum(
        P_import[t] * (FIXED_IMPORT_PRICE if STRATEGY == "FIXED" else df["total_consumption_rate"].iloc[t]) -
        P_export[t] * (FIXED_EXPORT_PRICE if STRATEGY == "FIXED" else df["grid_sellback_rate"].iloc[t]) +
        penalty_cost * P_curtail[t] for t in range(start_t, end_t)
    ), GRB.MINIMIZE)

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

    if model.status == GRB.OPTIMAL:
        print("✅ Optimization successful!")
    elif model.status == GRB.INFEASIBLE:
        print("❌ Model is infeasible!")
        model.computeIIS()
        model.write("model.ilp")
        return None, prev_final_soc, prev_batch_powers, False
    else:
        print(f"❌ Solver ended with status {model.status}")
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

    # ✅ FIXED Strategy
    for start_t in fixed_batch_starts:
        print(f"Processing FIXED batch starting at {start_t}")
        batch_results, prev_final_soc_fixed, prev_batch_powers_fixed, success = optimize_energy(
            start_t, "FIXED", prev_final_soc_fixed, prev_batch_powers_fixed)
        if success:
            all_results_fixed.extend(batch_results[:OVERLAP])
        else:
            failed_batches_fixed.append(start_t)

    # ✅ DYNAMIC Strategy
    for start_t in dynamic_batch_starts:
        print(f"Processing DYNAMIC batch starting at {start_t}")
        batch_results, prev_final_soc_dynamic, prev_batch_powers_dynamic, success = optimize_energy(
            start_t, "DYNAMIC", prev_final_soc_dynamic, prev_batch_powers_dynamic)
        if success:
            all_results_dynamic.extend(batch_results[:OVERLAP])
        else:
            failed_batches_dynamic.append(start_t)

    print(f"❌ FIXED Strategy Failures: {len(failed_batches_fixed)} batches failed.")
    print(f"✅ FIXED Strategy Success: {100 * (1 - len(failed_batches_fixed) / (len(fixed_batch_starts))):.2f}%")

    print(f"❌ DYNAMIC Strategy Failures: {len(failed_batches_dynamic)} batches failed.")
    print(f"✅ DYNAMIC Strategy Success: {100 * (1 - len(failed_batches_dynamic) / (len(dynamic_batch_starts))):.2f}%")

    results_fixed_df = pd.DataFrame(all_results_fixed)
    results_dynamic_df = pd.DataFrame(all_results_dynamic)

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

    # Ensure 'time' is an integer before converting
    results_fixed_df["time"] = pd.to_numeric(results_fixed_df["time"], errors="coerce")
    results_fixed_df["time"] = pd.to_datetime(results_fixed_df["time"], unit="ns", errors="coerce")

    results_dynamic_df["time"] = pd.to_numeric(results_dynamic_df["time"], errors="coerce")
    results_dynamic_df["time"] = pd.to_datetime(results_dynamic_df["time"], unit="ns", errors="coerce")

    print("✅ Start to Save Results...!")

    version = "v10_5"
    results_fixed_df.to_csv(f"WorkingCodeVersion1_FIXED_{version}.csv", index=False)
    results_dynamic_df.to_csv(f"WorkingCodeVersion1_DYNAMIC_{version}.csv", index=False)
    homer_df.to_csv(f"WorkingCodeVersion1_HOMER_{version}.csv", index=False)

    print("✅ Results saved successfully!")

    total_time = time.time() - start_time
    print(f"⏳ Total Execution Time: {total_time:.2f} seconds")