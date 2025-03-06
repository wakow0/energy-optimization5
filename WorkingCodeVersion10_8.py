import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from gurobipy import Model, GRB, quicksum
import gurobipy as gp


# Configuration
BATCH_SIZE = 48
OVERLAP = int(BATCH_SIZE / 2)
battery_capacity = 2000
battery_max_charge = 1200  # softened
battery_max_discharge = 1200  # softened
inverter_capacity = 1000
soc_min = 0.05 * battery_capacity
soc_max = battery_capacity
SOC_BUFFER = 0.98 * battery_capacity
FIXED_IMPORT_PRICE = 0.1939
FIXED_EXPORT_PRICE = 0.0769
penalty_cost = 50

# Load data
df = pd.read_csv("processed_data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(":", "")
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df.dropna(subset=["time"], inplace=True)

for col in ["total_consumption_rate", "grid_sellback_rate", "ac_primary_load", "dc_ground_1500vdc_power_output", "windflow_33_[500kw]_power_output"]:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

def optimize_batch(batch_df, initial_soc, strategy):
    model = Model("Batch_Optimization")
    model.Params.OutputFlag = 0
    batch_size = len(batch_df)

    P_import, P_export, P_bat_ch, P_bat_dis, SOC = {}, {}, {}, {}, {}
    X_import, X_export, X_bat_ch, X_bat_dis = {}, {}, {}, {}

    for t in range(batch_size):
        X_high_soc = model.addVar(vtype=GRB.BINARY)
        P_import[t] = model.addVar(lb=0, ub=inverter_capacity)
        P_export[t] = model.addVar(lb=0, ub=inverter_capacity)
        P_bat_ch[t] = model.addVar(lb=0, ub=battery_max_charge)
        P_bat_dis[t] = model.addVar(lb=0, ub=battery_max_discharge)
        SOC[t] = model.addVar(lb=soc_min, ub=soc_max)
        X_import[t] = model.addVar(vtype=GRB.BINARY)
        X_export[t] = model.addVar(vtype=GRB.BINARY)
        X_bat_ch[t] = model.addVar(vtype=GRB.BINARY)
        X_bat_dis[t] = model.addVar(vtype=GRB.BINARY)

    model.setObjective(quicksum(
        P_import[t] * (FIXED_IMPORT_PRICE if strategy == "FIXED" else batch_df["total_consumption_rate"].iloc[t]) -
        P_export[t] * (FIXED_EXPORT_PRICE if strategy == "FIXED" else batch_df["grid_sellback_rate"].iloc[t])
        for t in range(batch_size)
    ), GRB.MINIMIZE)

    for t in range(batch_size):
        available_renewable = batch_df["dc_ground_1500vdc_power_output"].iloc[t] + batch_df["windflow_33_[500kw]_power_output"].iloc[t]
        demand = batch_df["ac_primary_load"].iloc[t]

        model.addConstr(available_renewable + P_import[t] + P_bat_dis[t] >= demand + P_bat_ch[t] + P_export[t])
        model.addConstr(P_import[t] <= inverter_capacity * X_import[t])
        model.addConstr(P_export[t] <= inverter_capacity * X_export[t])
        model.addConstr(X_import[t] + X_export[t] <= 1)
        model.addConstr(P_bat_ch[t] <= battery_max_charge * X_bat_ch[t])
        model.addConstr(P_bat_dis[t] <= battery_max_discharge * X_bat_dis[t])
        model.addConstr(X_bat_ch[t] + X_bat_dis[t] <= 1)

        if t == 0:
            model.addConstr(SOC[t] == initial_soc)
        else:
            model.addConstr(SOC[t] == SOC[t-1] + P_bat_ch[t] * 0.95 - P_bat_dis[t] / 0.95)
            model.addConstr(SOC[t] - SOC[t-1] <= battery_max_charge)
            model.addConstr(SOC[t-1] - SOC[t] <= battery_max_discharge)
            model.addConstr(SOC[t] - 0.95 * battery_capacity >= -battery_capacity * (1 - X_high_soc))
            model.addConstr(P_bat_ch[t] <= battery_max_charge * (1 - X_high_soc))
            model.addConstr(P_bat_dis[t] >= 5 * X_high_soc)
            model.addConstr(SOC[t] <= SOC_BUFFER)
            model.addConstr(SOC[t] >= soc_min)
        
        

    model.optimize()

    if model.status != GRB.OPTIMAL:
        return [], initial_soc, "infeasible"

    results = []
    for t in range(batch_size):
        results.append({
            "time": str(batch_df["time"].iloc[t]),
            "P_import (kW)": P_import[t].X,
            "P_export (kW)": P_export[t].X,
            "P_bat_ch (kW)": P_bat_ch[t].X,
            "P_bat_dis (kW)": P_bat_dis[t].X,
            "SOC": SOC[t].X
        })
    return results, SOC[batch_size - 1].X, "feasible"

strategies = ["FIXED", "DYNAMIC"]

for strategy in strategies:
    initial_soc = soc_min
    total_intervals = len(df)
    all_results = []
    num_batches = int(np.ceil((total_intervals - OVERLAP) / OVERLAP))
    feasible_count, infeasible_count = 0, 0

    for batch_idx in tqdm(range(num_batches), desc=f"{strategy} Optimization"):
        start_idx = batch_idx * OVERLAP
        end_idx = min(start_idx + BATCH_SIZE, total_intervals)
        batch_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        batch_results, final_soc, status = optimize_batch(batch_df, initial_soc, strategy)

        if status == "feasible":
            feasible_count += 1
        else:
            infeasible_count += 1

        overlap_end = OVERLAP if end_idx < total_intervals else (end_idx - start_idx)
        all_results.extend(batch_results[:overlap_end])

        initial_soc = final_soc

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"WorkingCodeVersion10_8_Results_{strategy}.csv", index=False)

    print(f"✅ {strategy} Strategy - Feasible Batches: {feasible_count}")
    print(f"❌ {strategy} Strategy - Infeasible Batches: {infeasible_count}")
    print(f"✅ {strategy} Results saved to 'WorkingCodeVersion10_8_Results_{strategy}.csv'")
