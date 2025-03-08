import os
import time
import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD

# Configurable Parameters
BATCH_SIZE = 48  # Number of intervals to process at once (e.g., 1 day = 48)
LOOK_AHEAD_WINDOW = 10  # Increased look-ahead to 6 intervals (3 hours)
NUM_CORES = multiprocessing.cpu_count()  # Use all available CPU cores
SMOOTHING_WINDOW = 10  # Larger window size for smoothing

#FIXED_IMPORT_PRICE = 20 / 100  # Fixed import price (£/kWh) for Strategy 1
#FIXED_EXPORT_PRICE = 7.6 / 100  # Fixed export price (£/kWh) for Strategy 1

FIXED_IMPORT_PRICE = 0.1939  # Fixed import price (£/kWh) for Strategy 1
FIXED_EXPORT_PRICE = 0.0769  # Fixed export price (£/kWh) for Strategy 1

STRATEGY = "FIXED"
#STRATEGY = "DYNAMIC"


# Load Dataset
file_path = "research.csv"
file_path = "processed_data.csv"
file_path = "heuristic_optimization_results_v1.csv"


df = pd.read_csv(file_path, skiprows=0, dtype=str, low_memory=False)
#df = pd.read_csv(file_path, skiprows=3)
df.head()
print(df.columns.tolist())
#print(df.head(10))

DATA_PERCENTAGE = 1  # Change to 0.1 for 10%, 0.5 for 50%, etc.

# Calculate number of rows to use
total_rows = len(df)
print(f"☀️  Total rows: {total_rows:.2f}")
print(f"☀️  Total Columns: {total_rows:.2f}")
sampled_rows = int(total_rows * DATA_PERCENTAGE)
print(f"☀️  sampled rows: {sampled_rows:.2f}")
# Option 1: Select first X% of the dataset (sequential)
df_sampled = df.iloc[:5]  # Uses first X% of the year
#print(f"☀️  df sampled: {len(df_sampled):.2f}")
#print(df_sampled.iloc[:2-5])