import pandas as pd

# ✅ Load the latest optimization results
file_path = "rl_optimization_results_v3.csv"
file_path = "WorkingCodeVersion1_DYNAMIC_v10_6.csv"
#file_path = "WorkingCodeVersion1_FIXED_v10_5.csv"

file_path = "dynamic_pricing_v11_solution.csv"

file_path = "WorkingCodeVersion1_HOMER_v10.csv"
file_path = "dynamic_pricing_v12_solution.csv"


file_path = "dynamic_pricing_v13_solution.csv"

# ✅ Spike thresholds from Parameters.txt
max_change_limits = {
    "p_import (kw)": 1000,       # max_change_limit
    "p_export (kw)": 1000,       # max_change_limit
    "p_bat_ch (kw)": 1000,       # max_battery_rate_change
    "p_bat_dis (kw)": 1000,      # max_battery_rate_change
    "soc (%)": 100,              # Example value for SoC spikes (adjust as needed)
}

try:
    df = pd.read_csv(file_path, low_memory=False)

    # ✅ Convert column names to lowercase for consistency
    df.columns = df.columns.str.lower().str.strip()

    # ✅ Define the correct column names based on available data
    energy_columns = [
        "p_import (kw)",
        "p_export (kw)",
        "p_bat_ch (kw)",
        "p_bat_dis (kw)",
        "soc (%)"
    ]

    # ✅ Ensure the required columns exist in the dataset
    missing_columns = [col for col in energy_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ ERROR: Missing required columns in {file_path}: {missing_columns}")
    else:
        print("\n🔍 **Spike Check in Key Energy Variables:**")

        for col in energy_columns:
            diffs = df[col].diff().abs()
            rate_spikes = diffs > max_change_limits[col]
            value_spikes = df[col] > max_change_limits[col]

            num_rate_spikes = rate_spikes.sum()
            num_value_spikes = value_spikes.sum()

            # Report both
            if num_rate_spikes > 0:
                print(f"⚠️ {num_rate_spikes} rate-of-change spikes in '{col}' exceeding {max_change_limits[col]} units.")
            else:
                print(f"✅ No rate-of-change spikes in '{col}' (limit: {max_change_limits[col]} units).")

            if num_value_spikes > 0:
                print(f"🚨 {num_value_spikes} value spikes in '{col}' exceeding {max_change_limits[col]} units.")
            else:
                print(f"✅ No absolute value spikes in '{col}' (limit: {max_change_limits[col]} units).")

except FileNotFoundError:
    print(f"❌ ERROR: File '{file_path}' not found. Ensure the optimization has been run successfully.")
