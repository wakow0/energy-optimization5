import pandas as pd

# ✅ Load the latest optimization results
file_path = "WorkingCodeVersion1_FIXED_v10_8.csv"
file_path = "generated_decision_variables.csv"

try:
    
    df = pd.read_csv(file_path, low_memory=False)

    # ✅ Convert column names to lowercase for consistency
    df.columns = df.columns.str.lower().str.strip()

    # ✅ Define the correct column names based on available data
    energy_columns = [
        "p_import (kw)",  # Grid Purchases
        "p_export (kw)",  # Grid Sales
        "p_bat_ch (kw)",  # Battery Charge Power
        "p_bat_dis (kw)",  # Battery Discharge Power
        "soc (%)"
    ]

    # ✅ Ensure the required columns exist in the dataset
    missing_columns = [col for col in energy_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ ERROR: Missing required columns in {file_path}: {missing_columns}")
    else:
        # ✅ Check for negative values
        negative_values = df[energy_columns].lt(0).sum()

        print("\n🔍 **Negative Values Found in Key Energy Variables:**")
        print(negative_values)

except FileNotFoundError:
    print(f"❌ ERROR: File '{file_path}' not found. Ensure the optimization has been run successfully.")
