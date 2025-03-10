import pandas as pd
import numpy as np
import os
import gym
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

# Load datasets
ORIGINAL_DATA_FILE = "processed_data.csv"
HEURISTIC_RESULTS_FILE = "heuristic_optimization_results_v3.csv"
OUTPUT_FILE = "rl_optimization_results_v3.csv"

# Check if files exist
if not os.path.exists(ORIGINAL_DATA_FILE):
    raise FileNotFoundError(f"❌ ERROR: {ORIGINAL_DATA_FILE} is missing.")
if not os.path.exists(HEURISTIC_RESULTS_FILE):
    raise FileNotFoundError(f"❌ ERROR: {HEURISTIC_RESULTS_FILE} is missing.")

# Load original data and heuristic results
original_df = pd.read_csv(ORIGINAL_DATA_FILE)
heuristic_df = pd.read_csv(HEURISTIC_RESULTS_FILE)

# Standardize column names
original_df.columns = original_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
heuristic_df.columns = heuristic_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")

# Merge datasets on the "time" column
df = pd.merge(original_df, heuristic_df, on="time", how="left")



###############################################################

# Define percentage of data to use (e.g., 25% of 1-year data)
DATA_PERCENTAGE = 0.005  # Change to 0.1 for 10%, 0.5 for 50%, etc.

# Calculate number of rows to use
total_rows = len(df)
sampled_rows = int(total_rows * DATA_PERCENTAGE)

# Option 1: Select first X% of the dataset (sequential)
df_sampled = df.iloc[:sampled_rows]  # Uses first X% of the year

# Option 2: Randomly sample X% of the dataset (faster but less structured)
# df_sampled = df.sample(n=sampled_rows, random_state=42)  # Uncomment to use random selection

#print(f"✅ Using {DATA_PERCENTAGE * 100}% of the dataset ({sampled_rows} rows).")

df = df_sampled  # NEW (Use only sampled data)

################################################################




# Ensure required columns exist
required_columns = [
    "time", "ac_primary_load", "dc_ground_1500vdc_power_output", "windflow_33_[500kw]_power_output",
    "p_import_(kw)", "p_export_(kw)", "p_bat_ch_(kw)", "p_bat_dis_(kw)", "soc_(%)"
]
for col in required_columns:
    if col not in df.columns:
        df[col] = 0  # Default to zero if missing



print(df.columns.tolist())