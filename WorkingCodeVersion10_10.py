import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from gurobipy import Model, GRB, quicksum
import gurobipy as gp


start_time = time.time()

BATCH_SIZE = 48
OVERLAP = int(BATCH_SIZE * 0.75)

battery_capacity = 2000
battery_max_charge = 1200
battery_max_discharge = 1200
inverter_capacity = 1000
soc_min = 0.05 * battery_capacity
soc_max = battery_capacity
SOC_BUFFER = 0.85 * battery_capacity  # Further reduced buffer
FIXED_IMPORT_PRICE = 0.1939
FIXED_EXPORT_PRICE = 0.0769
penalty_cost = 50

max_change_import = 300
max_change_export = 300
max_change_bat_ch = 200
max_change_bat_dis = 200
spike_penalty_weight = 0.01

df = pd.read_csv("processed_data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(":", "")
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df.dropna(subset=["time"], inplace=True)

print("\n✅ Dataset check:")
print(df[["ac_primary_load", "dc_ground_1500vdc_power_output", "windflow_33_[500kw]_power_output"]].describe())

df["total_consumption_rate"] = df["total_consumption_rate"].rolling(window=5, center=True).mean().bfill().ffill()
df["grid_sellback_rate"] = df["grid_sellback_rate"].rolling(window=5, center=True).mean().bfill().ffill()

def optimize_batch(batch_df, initial_soc, strategy, prev_P_import, prev_P_export, prev_P_bat_ch, prev_P_bat_dis):
    model = Model("Batch_Optimization")
    model.Params.OutputFlag = 0
    batch_size = len(batch_df)

    P_import, P_export, P_bat_ch, P_bat_dis, SOC = {}, {}, {}, {}, {}
    delta_P_import, delta_P_export, delta_P_bat_ch, delta_P_bat_dis = {}, {}, {}, {}

    for t in range(batch_size):
        P_import[t] = model.addVar(lb=0, ub=inverter_capacity)
        P_export[t] = model.addVar(lb=0, ub=1000)  # Restored export cap to 1000 kW
        P_bat_ch[t] = model.addVar(lb=0, ub=battery_max_charge)
        P_bat_dis[t] = model.addVar(lb=0, ub=battery_max_discharge)
        SOC[t] = model.addVar(lb=soc_min, ub=soc_max)
        if t > 0:
            delta_P_import[t] = model.addVar(lb=0)
            delta_P_export[t] = model.addVar(lb=0)
            delta_P_bat_ch[t] = model.addVar(lb=0)
            delta_P_bat_dis[t] = model.addVar(lb=0)

    for t in range(batch_size):
        model.addConstr(P_bat_dis[t] >= 1)
        model.addConstr(P_export[t] >= 1)
        model.addConstr(SOC[t] <= SOC_BUFFER - t * (battery_capacity * 0.0005))  # Gradual SOC ceiling reduction

        if t == 0:
            model.addConstr(SOC[t] == initial_soc)
        else:
            model.addConstr(SOC[t] == SOC[t-1] + P_bat_ch[t] * 0.95 - P_bat_dis[t] / 0.95)
            model.addConstr(delta_P_import[t] >= P_import[t] - P_import[t-1])
            model.addConstr(delta_P_import[t] >= P_import[t-1] - P_import[t])
            model.addConstr(delta_P_export[t] >= P_export[t] - P_export[t-1])
            model.addConstr(delta_P_export[t] >= P_export[t-1] - P_export[t])
            model.addConstr(delta_P_bat_ch[t] >= P_bat_ch[t] - P_bat_ch[t-1])
            model.addConstr(delta_P_bat_ch[t] >= P_bat_ch[t-1] - P_bat_ch[t])
            model.addConstr(delta_P_bat_dis[t] >= P_bat_dis[t] - P_bat_dis[t-1])
            model.addConstr(delta_P_bat_dis[t] >= P_bat_dis[t-1] - P_bat_dis[t])
            excess_soc = model.addVar(lb=0)
            model.addConstr(excess_soc >= SOC[t] - 0.95 * battery_capacity)
            model.addConstr(P_bat_dis[t] >= 0.2 * excess_soc)  # Increased proportional discharge

    model.setObjective(
        sum(
            P_import[t] * (FIXED_IMPORT_PRICE if strategy == "FIXED" else batch_df["total_consumption_rate"].iloc[t]) -
            P_export[t] * (FIXED_EXPORT_PRICE if strategy == "FIXED" else batch_df["grid_sellback_rate"].iloc[t])
            for t in range(batch_size)
        ) + spike_penalty_weight * sum(
            delta_P_import[t] + delta_P_export[t] + delta_P_bat_ch[t] + delta_P_bat_dis[t]
            for t in range(1, batch_size)
        ), GRB.MINIMIZE
    )

    model.optimize()

    if model.status != GRB.OPTIMAL:
        return [], initial_soc, "infeasible", prev_P_import, prev_P_export, prev_P_bat_ch, prev_P_bat_dis

    results = []
    for t in range(batch_size):
        results.append({
            "time": str(batch_df["time"].iloc[t]),
            "P_import (kW)": P_import[t].X,
            "P_export (kW)": P_export[t].X,
            "P_bat_ch (kW)": P_bat_ch[t].X,
            "P_bat_dis (kW)": P_bat_dis[t].X,
            "SOC (%)": SOC[t].X
        })

    last_idx = batch_size - 1
    return results, SOC[last_idx].X, "feasible", P_import[last_idx].X, P_export[last_idx].X, P_bat_ch[last_idx].X, P_bat_dis[last_idx].X




strategies = ["FIXED", "DYNAMIC"]

for strategy in strategies:
    initial_soc = soc_min
    total_intervals = len(df)
    all_results = []
    num_batches = int(np.ceil((total_intervals - OVERLAP) / OVERLAP))
    feasible_count, infeasible_count = 0, 0

    prev_P_import, prev_P_export, prev_P_bat_ch, prev_P_bat_dis = 0, 0, 0, 0  # For the first batch
    for batch_idx in tqdm(range(num_batches), desc=f"{strategy} Optimization"):
        start_idx = batch_idx * OVERLAP
        end_idx = min(start_idx + BATCH_SIZE, total_intervals)
        batch_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        #batch_results, final_soc, status = optimize_batch(batch_df, initial_soc, strategy)
        batch_results, final_soc, status, last_P_import, last_P_export, last_P_bat_ch, last_P_bat_dis = optimize_batch(
        batch_df, initial_soc, strategy, prev_P_import, prev_P_export, prev_P_bat_ch, prev_P_bat_dis)

        prev_P_import, prev_P_export, prev_P_bat_ch, prev_P_bat_dis = last_P_import, last_P_export, last_P_bat_ch, last_P_bat_dis


        if status == "feasible":
            feasible_count += 1
        else:
            infeasible_count += 1

        overlap_end = OVERLAP if end_idx < total_intervals else (end_idx - start_idx)
        
        all_results.extend(batch_results[:overlap_end])

    

        initial_soc = final_soc

    results_df = pd.DataFrame(all_results)
    version = "v10_10"
    results_df.to_csv(f"WorkingCodeVersion1_{strategy}_{version}.csv", index=False)
     # Save results to CSV files
    

    print("✅ Results saved successfully!")
    

    print(f"✅ {strategy} Strategy - Feasible Batches: {feasible_count}")
    print(f"❌ {strategy} Strategy - Infeasible Batches: {infeasible_count}")
    print(f"✅ {strategy} Results saved to 'WorkingCodeVersion10_8_Results_{strategy}.csv'")

    print("✅ Optimization completed successfully!")    

    total_time = time.time() - start_time
    print(f"⏳ Total Execution Time: {total_time:.2f} seconds")