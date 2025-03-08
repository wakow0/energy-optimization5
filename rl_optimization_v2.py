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
from tqdm import tqdm

# Load dataset
DATA_FILE = "heuristic_optimization_results_v2.csv"
OUTPUT_FILE = "rl_optimization_results_v2.csv"

# Check if file exists
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"❌ ERROR: {DATA_FILE} is missing.")

df = pd.read_csv(DATA_FILE)
print("Available columns in dataset:", df.columns.tolist())  # Debugging column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")

# Ensure required columns exist
required_columns = [
    "soc_(%)", "p_import_(kw)", "p_export_(kw)", "total_consumption_rate", 
    "dc_ground_1500vdc_power_output", "windflow_33_[500kw]_power_output", "grid_sellback_rate"
]
for col in required_columns:
    if col not in df.columns:
        df[col] = 0  # Default to zero if missing

# Define RL Environment
class EnergyOptimizationEnv(gym.Env):
    def __init__(self, df):
        super(EnergyOptimizationEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(required_columns),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)  # Actions: Charge, Discharge, Import, Export, Do Nothing
        self.battery_soc = 50.0  # Start at 50% SOC (percentage)

    def reset(self):
        self.current_step = 0
        self.battery_soc = 50.0  # Reset SOC to 50%
        return self._next_observation()

    def _next_observation(self):
        max_vals = np.max(self.df[required_columns].values, axis=0)
        max_vals[max_vals == 0] = 1  # Prevent division by zero
        obs = self.df.iloc[self.current_step][required_columns].values / max_vals
        return obs.astype(np.float32)

    def step(self, action):
        done = False
        reward = 0

        if action == 0:  # Charge battery
            self.battery_soc = min(self.battery_soc + 5.0, 100.0)  # Ensure SOC does not exceed 100%
        elif action == 1:  # Discharge battery
            self.battery_soc = max(self.battery_soc - 5.0, 0.0)  # Ensure SOC does not drop below 0%
        elif action == 2:  # Import from grid
            reward -= self.df.loc[self.current_step, "total_consumption_rate"]
        elif action == 3:  # Export to grid
            reward += self.df.loc[self.current_step, "grid_sellback_rate"]
        
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        return self._next_observation(), reward, done, {}


class EnergyOptimizationEnv1(gym.Env):
    def __init__(self, df):
        super(EnergyOptimizationEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(required_columns),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)  # Actions: Charge, Discharge, Import, Export, Do Nothing
        self.battery_soc = 50.0  # Start at 50% SOC (percentage)

    def reset(self):
        self.current_step = 0
        self.battery_soc = 50.0  # Reset SOC to 50%
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
        if action == 0:  # Charge battery
            charge_power = min(100.0 - self.battery_soc, 5.0)  # Charge by 5%
            self.battery_soc += charge_power
            reward -= charge_power * 0.1  # Small penalty for charging
        elif action == 1:  # Discharge battery
            discharge_power = min(self.battery_soc, 5.0)  # Discharge by 5%
            self.battery_soc -= discharge_power
            reward += discharge_power * 0.1  # Small reward for discharging
        elif action == 2:  # Import from grid
            grid_import = max(demand - renewable_gen - self.battery_soc, 0)
            reward -= grid_import * market_price  # Penalize grid import cost
        elif action == 3:  # Export to grid
            grid_export = min(renewable_gen - demand, self.battery_soc)
            reward += grid_export * sellback_price  # Reward grid export revenue

        # Ensure SOC stays within limits
        self.battery_soc = max(0.0, min(self.battery_soc, 100.0))

        # Move to next step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        return self._next_observation(), reward, done, {}


# Train PPO Model
def train_rl():
    start_time = time.time()
    env = EnergyOptimizationEnv(df)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)  # Increased timesteps for better learning
    model.save("ppo_energy_model")
    end_time = time.time()
    print(f"⏳ Training Time: {end_time - start_time:.2f} seconds")

# Generate RL-based optimization results
def run_rl_optimization():
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

# Run Training and Optimization
if __name__ == "__main__":
    train_rl()
    run_rl_optimization()