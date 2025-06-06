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

# Load dataset
ORIGINAL_DATA_FILE = "processed_data.csv"
OUTPUT_FILE = "rl_optimization_results_v4.csv"

# Check if file exists
if not os.path.exists(ORIGINAL_DATA_FILE):
    raise FileNotFoundError(f"❌ ERROR: {ORIGINAL_DATA_FILE} is missing.")

# Load original data
df = pd.read_csv(ORIGINAL_DATA_FILE)

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")

# Ensure required columns exist
required_columns = [
    "time", "ac_primary_load", "dc_ground_1500vdc_power_output", "windflow_33_[500kw]_power_output",
    "p_import_(kw)", "p_export_(kw)", "p_bat_ch_(kw)", "p_bat_dis_(kw)", "soc_(%)"
]
for col in required_columns:
    if col not in df.columns:
        df[col] = 0  # Default to zero if missing

# Convert all required columns to numeric, coercing errors to NaN
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill NaN values with 0 (or another appropriate value)
df.fillna(0, inplace=True)

# Battery and system constraints
BATTERY_CAPACITY = 2000  # kWh
SOC_MIN = 0.05 * BATTERY_CAPACITY  # Min SOC (5%)
SOC_MAX = 1.0 * BATTERY_CAPACITY  # Max SOC (100%)
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95

# Define RL Environment
class EnergyOptimizationEnv(gym.Env):
    def __init__(self, df):
        super(EnergyOptimizationEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(required_columns),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(10)  # More granular actions
        self.battery_soc = SOC_MIN  # Start at minimum SOC (5%)

    def reset(self):
        self.current_step = 0
        self.battery_soc = SOC_MIN  # Reset SOC to minimum (5%)
        return self._next_observation()

    def _next_observation(self):
        max_vals = np.max(self.df[required_columns].values, axis=0)
        max_vals[max_vals == 0] = 1  # Prevent division by zero
        obs = self.df.iloc[self.current_step][required_columns].values / max_vals
        return obs.astype(np.float32)

    def step(self, action):
        done = False
        reward = 0

        # Get current state
        demand = self.df.loc[self.current_step, "ac_primary_load"]
        renewable_gen = self.df.loc[self.current_step, "dc_ground_1500vdc_power_output"] + self.df.loc[self.current_step, "windflow_33_[500kw]_power_output"]
        market_price = self.df.loc[self.current_step, "total_consumption_rate"]
        sellback_price = self.df.loc[self.current_step, "grid_sellback_rate"]

        # Apply action
        if action == 0:  # Charge battery (small)
            charge_power = min((SOC_MAX - self.battery_soc) / CHARGE_EFFICIENCY, 1.0)  # Charge by 1%
            self.battery_soc += charge_power * CHARGE_EFFICIENCY
            reward -= charge_power * 0.1  # Small penalty for charging
        elif action == 1:  # Charge battery (medium)
            charge_power = min((SOC_MAX - self.battery_soc) / CHARGE_EFFICIENCY, 5.0)  # Charge by 5%
            self.battery_soc += charge_power * CHARGE_EFFICIENCY
            reward -= charge_power * 0.1  # Small penalty for charging
        elif action == 2:  # Charge battery (large)
            charge_power = min((SOC_MAX - self.battery_soc) / CHARGE_EFFICIENCY, 10.0)  # Charge by 10%
            self.battery_soc += charge_power * CHARGE_EFFICIENCY
            reward -= charge_power * 0.1  # Small penalty for charging
        elif action == 3:  # Discharge battery (small)
            discharge_power = min((self.battery_soc - SOC_MIN) * DISCHARGE_EFFICIENCY, 1.0)  # Discharge by 1%
            self.battery_soc -= discharge_power / DISCHARGE_EFFICIENCY
            reward += discharge_power * 0.1  # Small reward for discharging
        elif action == 4:  # Discharge battery (medium)
            discharge_power = min((self.battery_soc - SOC_MIN) * DISCHARGE_EFFICIENCY, 5.0)  # Discharge by 5%
            self.battery_soc -= discharge_power / DISCHARGE_EFFICIENCY
            reward += discharge_power * 0.1  # Small reward for discharging
        elif action == 5:  # Discharge battery (large)
            discharge_power = min((self.battery_soc - SOC_MIN) * DISCHARGE_EFFICIENCY, 10.0)  # Discharge by 10%
            self.battery_soc -= discharge_power / DISCHARGE_EFFICIENCY
            reward += discharge_power * 0.1  # Small reward for discharging
        elif action == 6:  # Import from grid
            grid_import = max(demand - renewable_gen - self.battery_soc, 0)
            reward -= grid_import * market_price  # Penalize grid import cost
        elif action == 7:  # Export to grid
            grid_export = min(renewable_gen - demand, self.battery_soc)
            reward += grid_export * sellback_price  # Reward grid export revenue
        elif action == 8:  # Do nothing
            pass
        elif action == 9:  # Maintain SOC within desired range
            if self.battery_soc < 0.2 * BATTERY_CAPACITY:  # If SOC is below 20%, charge
                charge_power = min((0.2 * BATTERY_CAPACITY - self.battery_soc) / CHARGE_EFFICIENCY, 5.0)
                self.battery_soc += charge_power * CHARGE_EFFICIENCY
                reward -= charge_power * 0.1
            elif self.battery_soc > 0.8 * BATTERY_CAPACITY:  # If SOC is above 80%, discharge
                discharge_power = min((self.battery_soc - 0.8 * BATTERY_CAPACITY) * DISCHARGE_EFFICIENCY, 5.0)
                self.battery_soc -= discharge_power / DISCHARGE_EFFICIENCY
                reward += discharge_power * 0.1

        # Ensure SOC stays within limits
        self.battery_soc = max(SOC_MIN, min(self.battery_soc, SOC_MAX))

        # Reward for maintaining SOC within desired range (20% to 80%)
        if 0.2 * BATTERY_CAPACITY <= self.battery_soc <= 0.8 * BATTERY_CAPACITY:
            reward += 1.0  # Reward for maintaining SOC in the desired range

        # Move to next step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        return self._next_observation(), reward, done, {}

# Custom callback for updating the progress bar
class ProgressBarCallback(BaseCallback):
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self.pbar = pbar

    def _on_step(self):
        self.pbar.update(1)  # Update progress bar by 1 step
        return True

# Train PPO Model
def train_rl(pbar):
    start_time = time.time()
    env = EnergyOptimizationEnv(df)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, batch_size=64, gamma=0.99)
    callback = ProgressBarCallback(pbar)  # Use custom callback
    model.learn(total_timesteps=200000, callback=callback)  # Train for more timesteps
    model.save("ppo_energy_model")
    end_time = time.time()
    print(f"⏳ Training Time: {end_time - start_time:.2f} seconds")

# Generate RL-based optimization results
def run_rl_optimization(pbar):
    start_time = time.time()
    env = EnergyOptimizationEnv(df)
    model = PPO.load("ppo_energy_model")  # Load the trained model
    obs = env.reset()
    df["RL_Action"] = 0  # Initialize RL action column with default value
    results = []

    # Progress bar for RL optimization
    for _ in tqdm(range(len(df)), desc="Running RL Optimization", unit="step"):
        if len(results) >= len(df):
            break
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        results.append(action)
        if done:
            break

    # Ensure results match df length
    while len(results) < len(df):
        results.append(0)  # Default action if RL stops early
    df["RL_Action"] = results

    # Ensure required output columns exist before saving
    required_output_columns = [
        "time", "p_import_(kw)", "p_export_(kw)", "p_bat_ch_(kw)", "p_bat_dis_(kw)", "soc_(%)"
    ]

    for col in required_output_columns:
        if col not in df.columns:
            df[col] = 0  # Default value

    df[required_output_columns].to_csv(OUTPUT_FILE, index=False)
    print(f"✅ RL optimization completed. Results saved to {OUTPUT_FILE}")

    end_time = time.time()
    print(f"⏳ RL Optimization Time: {end_time - start_time:.2f} seconds")
    pbar.update(60)  # Update progress bar after optimization

# Run Training and Optimization with a single progress bar
if __name__ == "__main__":
    with tqdm(total=200000 + len(df), desc="Overall Progress", unit="step") as pbar:
        # Train the RL model
        train_rl(pbar)

        # Run RL optimization
        run_rl_optimization(pbar)