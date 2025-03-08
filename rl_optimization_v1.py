import pandas as pd
import numpy as np
import os
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from stable_baselines3 import PPO
from tqdm import tqdm

# Load dataset
DATA_FILE = "heuristic_optimization_results_v1.csv"
OUTPUT_FILE = "rl_optimization_results_v1.csv"

# Check if file exists
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"❌ ERROR: {DATA_FILE} is missing.")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Ensure required columns exist
required_columns = [
    "battery_soc", "grid_import", "grid_export", "total_consumption_rate", 
    "dc_ground_1500vdc_power_output", "windflow_33_[500kw]_power_output"
]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"❌ ERROR: Missing column: {col}")

# Define RL Environment
class EnergyOptimizationEnv(gym.Env):
    def __init__(self, df):
        super(EnergyOptimizationEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(required_columns),), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)  # Actions: Charge, Discharge, Import, Export, Do Nothing
        self.battery_soc = 0.5  # Start at 50% SOC

    def reset(self):
        self.current_step = 0
        self.battery_soc = 0.5  # Reset SOC to 50%
        return self._next_observation()

    def _next_observation(self):
        max_vals = np.max(self.df[required_columns].values, axis=0)
        max_vals[max_vals == 0] = 1  # Prevent division by zero
        obs = self.df.iloc[self.current_step][required_columns].values / max_vals
        obs[required_columns.index("battery_soc")] /= 100  # Normalize SOC (0–1 range)

        return obs.astype(np.float32)

    def step(self, action):
        done = False
        reward = 0

        if action == 0:  # Charge battery
            self.battery_soc = min(self.battery_soc + (0.05 * 100), 100.0)  # Charge battery
        elif action == 1:  # Discharge battery
            self.battery_soc = max(self.battery_soc - (0.05 * 100), 0.0)  # Discharge battery
        elif action == 2:  # Import from grid
            reward -= self.df.loc[self.current_step, "total_consumption_rate"]
        elif action == 3:  # Export to grid
            reward += self.df.loc[self.current_step, "grid_sellback_rate"]
        
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        return self._next_observation(), reward, done, {}

# Train PPO Model
import time  # Import time module
def train_rl():
    start_time = time.time()
    env = EnergyOptimizationEnv(df)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_energy_model")
    end_time = time.time()
    print(f"⏳ Training Time: {end_time - start_time:.2f} seconds")


# Generate RL-based optimization results
def run_rl_optimization():
    start_time = time.time()
    env = EnergyOptimizationEnv(df)
    model = PPO.load("ppo_energy_model")  # Ensure model is loaded before predictions
    obs = env.reset()
    df["rl_action"] = 0  # Initialize RL action column with default value
    results = []

    # Progress bar for RL optimization
    for _ in tqdm(range(len(df)), desc="Running RL Optimization", unit="step"): 
        if len(results) >= len(df):
            break
        action, _ = model.predict(obs)  # Model is now properly defined
        obs, _, done, _ = env.step(action)
        results.append(action)
        if done:
            break

    # Ensure results match df length
    while len(results) < len(df):
        results.append(0)

    df["rl_action"] = results

    # Define output columns ensuring compatibility with MILP results
    output_columns = [
        "time", "battery_charge_power", "battery_discharge_power", 
        "grid_import", "grid_export", "battery_soc", "energy_cost", 
        "revenue_from_export"
    ]
    if "rl_action" in df.columns:
        output_columns.append("rl_action")

    df[output_columns].to_csv(OUTPUT_FILE, index=False)
    print(f"✅ RL optimization completed. Results saved to {OUTPUT_FILE}")
    end_time = time.time()
    print(f"⏳ RL Optimization Time: {end_time - start_time:.2f} seconds")


# Run Training and Optimization
if __name__ == "__main__":
    train_rl()
    run_rl_optimization()
