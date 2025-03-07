import pandas as pd

# ✅ Load the latest optimization results
file_path = "WorkingCodeVersion1_FIXED_v10_10.csv"
#file_path = "WorkingCodeVersion1_DYNAMIC_v10_10.csv"

# ✅ Spike thresholds from Parameters.txt
max_change_limits = {
    "p_import (kw)": 500,       # max_change_limit
    "p_export (kw)": 500,       # max_change_limit
    "p_bat_ch (kw)": 300,       # max_battery_rate_change
    "p_bat_dis (kw)": 300,      # max_battery_rate_change
    "soc (%)": 20,              # Example value for SoC spikes (adjust as needed)
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

        # ✅ Check for spikes
        for col in energy_columns:
            diffs = df[col].diff().abs()  # Absolute difference between consecutive rows
            spikes = diffs > max_change_limits[col]
            num_spikes = spikes.sum()

            if num_spikes > 0:
                print(f"⚠️ {num_spikes} spikes detected in '{col}' exceeding {max_change_limits[col]} units.")
            else:
                print(f"✅ No spikes detected in '{col}' (limit: {max_change_limits[col]} units).")

except FileNotFoundError:
    print(f"❌ ERROR: File '{file_path}' not found. Ensure the optimization has been run successfully.")
