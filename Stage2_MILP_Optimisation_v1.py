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

deviation_penalty = 0.001  # Penalize unnecessary deviation from initial values
max_adjustment = 100  # Allow max 100 kW deviation from initial solution
max_deviation = 50  # Allow 50 kW flexibility

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
    initial_solution["p_export_(kw)"] = initial_solution["p_export_(kw)"].clip(upper=1000)  # Ensure no conflicts
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
        
        for t in range(time_intervals):
            P_import[t] = model.addVar(lb=0, ub=inverter_capacity, name=f"P_import_{t}")
            P_export[t] = model.addVar(lb=0, ub=1000, name=f"P_export_{t}")
            P_bat_ch[t] = model.addVar(lb=0, ub=battery_max_charge, name=f"P_bat_ch_{t}")
            P_bat_dis[t] = model.addVar(lb=0, ub=battery_max_discharge, name=f"P_bat_dis_{t}")
            SOC[t] = model.addVar(lb=soc_min, ub=soc_max, name=f"SOC_{t}")

            # ✅ Set warm start values from the initial solution
            P_import[t].start = initial_solution.loc[t, "p_import_(kw)"]
            P_export[t].start = initial_solution.loc[t, "p_export_(kw)"]
            P_bat_ch[t].start = initial_solution.loc[t, "p_bat_ch_(kw)"]
            P_bat_dis[t].start = initial_solution.loc[t, "p_bat_dis_(kw)"]
            SOC[t].start = (initial_solution.loc[t, "soc_(%)"] / 100) * battery_capacity
        
        model.update()

        model.setObjective(
            quicksum(
                P_import[t] * FIXED_IMPORT_PRICE
                - P_export[t] * FIXED_EXPORT_PRICE
                + deviation_penalty * (abs(P_import[t] - initial_solution.loc[t, "p_import_(kw)"])
                                     + abs(P_export[t] - initial_solution.loc[t, "p_export_(kw)"])
                                     + abs(P_bat_ch[t] - initial_solution.loc[t, "p_bat_ch_(kw)"])
                                     + abs(P_bat_dis[t] - initial_solution.loc[t, "p_bat_dis_(kw)"]))
                for t in range(time_intervals)
            ), GRB.MINIMIZE
        )
        
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            print("❌ Model is infeasible. Running IIS computation to diagnose issues...")
            model.computeIIS()
            model.write("infeasible_model.ilp")
            print("❌ IIS report saved as infeasible_model.ilp")
            return

        print(f"✅ Optimal solution found for {pricing_strategy} Pricing!")

    except Exception as e:
        print(f"❌ An error occurred during optimization: {e}")

print("🚀 Starting MILP Optimization with Warm Start...")
optimize_energy_with_warm_start("FIXED")
optimize_energy_with_warm_start("DYNAMIC")
print("✅ Optimization completed successfully!")
