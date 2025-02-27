import pandas as pd

# =======================
# LOAD DATA
# =======================
def load_csv(file_path):
    """Loads a CSV file and ensures correct formatting."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(":", "")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        
        # Convert relevant columns to numeric
        numeric_columns = ["p_import_(kw)", "p_export_(kw)", "p_bat_ch_(kw)", "p_bat_dis_(kw)", "soc_(%)"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        print(f"✅ Loaded {file_path}: {df.shape[0]} rows")
        return df
    except Exception as e:
        print(f"❌ ERROR loading {file_path}: {e}")
        return None

# Load all three datasets
fixed_df = load_csv("WorkingCodeVersion1_FIXED_v1.csv")
dynamic_df = load_csv("WorkingCodeVersion1_DYNAMIC_v1.csv")
homer_df = load_csv("WorkingCodeVersion1_HOMER_v1.csv")

# Ensure all dataframes are valid
if fixed_df is None or dynamic_df is None or homer_df is None:
    raise ValueError("❌ ERROR: One or more CSV files could not be loaded. Check the file format.")

# =======================
# CHECK AND ALIGN ROW COUNTS
# =======================
# Print initial row counts
print(f"\n📊 Initial Row Counts:")
print(f"🔹 Fixed: {len(fixed_df)} rows")
print(f"🔹 Dynamic: {len(dynamic_df)} rows")
print(f"🔹 HOMER: {len(homer_df)} rows")

# Ensure all datasets have the same number of rows
min_rows = min(len(fixed_df), len(dynamic_df), len(homer_df))
fixed_df = fixed_df.iloc[:min_rows]
dynamic_df = dynamic_df.iloc[:min_rows]
homer_df = homer_df.iloc[:min_rows]

print(f"✅ Trimmed all datasets to {min_rows} rows for alignment.")

# Ensure consistent column formatting for HOMER dataset
homer_df.columns = homer_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(":", "")

# Now rename columns in HOMER dataset
homer_df = homer_df.rename(columns={
    "p_import_(kw)": "p_import_(kw)_homer",
    "p_export_(kw)": "p_export_(kw)_homer",
    "p_bat_ch_(kw)": "p_bat_ch_(kw)_homer",
    "p_bat_dis_(kw)": "p_bat_dis_(kw)_homer",
    "soc_(%)": "soc_(%)_homer"
})

# Now merge safely
comparison_df = fixed_df.merge(dynamic_df, on="time", suffixes=("_fixed", "_dynamic"))
comparison_df = comparison_df.merge(homer_df, on="time", how="left")  # Ensure HOMER data is correctly merged

# Print row count after merging
print(f"\n📊 Row Counts After Merging:")
print(f"🔹 Fixed: {len(fixed_df)} rows")
print(f"🔹 Dynamic: {len(dynamic_df)} rows")
print(f"🔹 HOMER: {len(homer_df)} rows")
print(f"🔹 Comparison DataFrame: {len(comparison_df)} rows")

# If row counts don't match, print a warning
if len(comparison_df) < min_rows:
    print(f"⚠️ WARNING: Some rows were dropped during merging! Expected {min_rows}, got {len(comparison_df)}.")

# =======================
# CHECK FOR MISSING TIMESTAMPS
# =======================
missing_in_homer = fixed_df[~fixed_df["time"].isin(homer_df["time"])]
missing_in_fixed = homer_df[~homer_df["time"].isin(fixed_df["time"])]

if not missing_in_homer.empty:
    print(f"❌ {len(missing_in_homer)} rows missing in HOMER but present in Fixed/Dynamic.")

if not missing_in_fixed.empty:
    print(f"❌ {len(missing_in_fixed)} rows missing in Fixed/Dynamic but present in HOMER.")

# Print a few missing timestamps for debugging
if not missing_in_homer.empty:
    print(f"🔍 Sample missing timestamps in HOMER: {missing_in_homer['time'].head().tolist()}")
if not missing_in_fixed.empty:
    print(f"🔍 Sample missing timestamps in Fixed/Dynamic: {missing_in_fixed['time'].head().tolist()}")

# =======================
# CONTINUE WITH ANALYSIS
# =======================
print("\n✅ All row checks completed. Proceeding with analysis...")
