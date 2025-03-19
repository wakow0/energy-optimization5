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

deviation_penalty = 0.001  
max_adjustment = 100  

battery_capacity = 2000
battery_max_charge = 1000
battery_max_discharge = 1000
inverter_capacity = 1000
soc_min = 0.05 * battery_capacity
soc_max = 1.0 * battery_capacity
eta_ch = 0.95
eta_dis = 0.95

PV_capacity = 1500
Wind_capacity = 500
max_renewable_capacity = PV_capacity + Wind_capacity

penalty_cost = 50
SOC_BUFFER = 0.98 * soc_max
OVERLAP = int(BATCH_SIZE * 0.5)

# =======================
# LOAD INITIAL SOLUTION
# =======================
try:
    print("🚀 Loading initial solution...")
    initial_solution = pd.read_csv("solution_output_FIXED_v5.csv")
    initial_solution.columns = initial_solution.columns.str.strip().str.lower().str.replace(" ", "_")
    initial_solution["time"] = pd.to_datetime(initial_solution["time"], errors="coerce")
    initial_solution.fillna(0, inplace=True)
    print("✅ Initial solution loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load initial solution: {e}")
    exit(1)

# =======================
# LOAD PROCESSED DATA
# =======================
try:
    print("🚀 Loading processed data for pricing and constraints...")
    df = pd.read_csv("processed_data.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df.fillna(0, inplace=True)

    required_columns = ["total_consumption_rate", "grid_sellback_rate"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"❌ Missing required columns in processed_data.csv: {missing_columns}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    print("✅ Processed data loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load processed data: {e}")
    exit(1)

time_intervals = len(df)

# =======================
# MILP OPTIMIZATION
# =======================
def optimize_energy_with_warm_start(pricing_strategy):
    try:
        print(f"✅ Running optimization for {pricing_strategy} Pricing...")

        model = Model("Energy_Optimization")
        model.Params.OutputFlag = 1
        model.Params.TimeLimit = 60

        P_import, P_export, P_bat_ch, P_bat_dis, SOC = {}, {}, {}, {}, {}
        deviation_import, deviation_export, deviation_charge, deviation_discharge = {}, {}, {}, {}

        for t in range(time_intervals):
            P_import[t] = model.addVar(lb=0, ub=inverter_capacity, name=f"P_import_{t}")
            P_export[t] = model.addVar(lb=0, ub=inverter_capacity, name=f"P_export_{t}")
            P_bat_ch[t] = model.addVar(lb=0, ub=battery_max_charge, name=f"P_bat_ch_{t}")
            P_bat_dis[t] = model.addVar(lb=0, ub=battery_max_discharge, name=f"P_bat_dis_{t}")
            SOC[t] = model.addVar(lb=soc_min, ub=soc_max, name=f"SOC_{t}")

            deviation_import[t] = model.addVar(lb=0, name=f"dev_import_{t}")
            deviation_export[t] = model.addVar(lb=0, name=f"dev_export_{t}")
            deviation_charge[t] = model.addVar(lb=0, name=f"dev_charge_{t}")
            deviation_discharge[t] = model.addVar(lb=0, name=f"dev_discharge_{t}")

        model.update()

        for t in range(time_intervals):
            model.addConstr(P_import[t] == initial_solution.loc[t, "p_import_(kw)"], name=f"init_import_{t}")
            model.addConstr(P_export[t] == initial_solution.loc[t, "p_export_(kw)"], name=f"init_export_{t}")
            model.addConstr(P_bat_ch[t] == initial_solution.loc[t, "p_bat_ch_(kw)"], name=f"init_charge_{t}")
            model.addConstr(P_bat_dis[t] == initial_solution.loc[t, "p_bat_dis_(kw)"], name=f"init_discharge_{t}")
            model.addConstr(SOC[t] == (initial_solution.loc[t, "soc_(%)"] / 100) * battery_capacity, name=f"init_soc_{t}")

            model.addConstr(deviation_import[t] >= P_import[t] - initial_solution.loc[t, "p_import_(kw)"])
            model.addConstr(deviation_import[t] >= initial_solution.loc[t, "p_import_(kw)"] - P_import[t])

            model.addConstr(deviation_export[t] >= P_export[t] - initial_solution.loc[t, "p_export_(kw)"])
            model.addConstr(deviation_export[t] >= initial_solution.loc[t, "p_export_(kw)"] - P_export[t])

            model.addConstr(deviation_charge[t] >= P_bat_ch[t] - initial_solution.loc[t, "p_bat_ch_(kw)"])
            model.addConstr(deviation_charge[t] >= initial_solution.loc[t, "p_bat_ch_(kw)"] - P_bat_ch[t])

            model.addConstr(deviation_discharge[t] >= P_bat_dis[t] - initial_solution.loc[t, "p_bat_dis_(kw)"])
            model.addConstr(deviation_discharge[t] >= initial_solution.loc[t, "p_bat_dis_(kw)"] - P_bat_dis[t])

        model.setObjective(
            quicksum(
                P_import[t] * (FIXED_IMPORT_PRICE if pricing_strategy == "FIXED" else df["total_consumption_rate"].iloc[t])
                - P_export[t] * (FIXED_EXPORT_PRICE if pricing_strategy == "FIXED" else df["grid_sellback_rate"].iloc[t])
                + deviation_penalty * (deviation_import[t] + deviation_export[t] + deviation_charge[t] + deviation_discharge[t])
                for t in range(time_intervals)
            ), GRB.MINIMIZE
        )

        model.optimize()

        if model.Status == GRB.OPTIMAL:
            print(f"✅ Optimal solution found for {pricing_strategy} Pricing!")
        else:
            print(f"⚠️ Unexpected optimization status: {model.Status}")
            return

        results = pd.DataFrame({
            "time": initial_solution["time"],
            "P_import (kW)": [P_import[t].X for t in range(time_intervals)],
            "P_export (kW)": [P_export[t].X for t in range(time_intervals)],
            "P_bat_ch (kW)": [P_bat_ch[t].X for t in range(time_intervals)],
            "P_bat_dis (kW)": [P_bat_dis[t].X for t in range(time_intervals)],
            "SOC (%)": [(SOC[t].X / battery_capacity) * 100 for t in range(time_intervals)]
        })

        results.to_csv(f"optimized_{pricing_strategy.lower()}.csv", index=False)
        print(f"✅ Optimized {pricing_strategy} results saved successfully!")

    except Exception as e:
        print(f"❌ An error occurred during optimization: {e}")

print("🚀 Starting MILP Optimization with Warm Start...")
optimize_energy_with_warm_start("FIXED")
optimize_energy_with_warm_start("DYNAMIC")
print("✅ Optimization completed successfully!")
