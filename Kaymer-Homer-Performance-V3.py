import pandas as pd
import matplotlib.pyplot as plt

# =======================
# LOAD DATA
# =======================
def load_csv(file_path):
    """Loads a CSV file and ensures correct formatting."""
    try:
        df = pd.read_csv(file_path, dtype=str, low_memory=False)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(":", "")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        
        # Convert relevant columns to numeric
        numeric_columns = ["p_import_(kw)", "p_export_(kw)", "p_bat_ch_(kw)", "p_bat_dis_(kw)", "soc_(%)"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df
    except Exception as e:
        print(f"❌ ERROR loading {file_path}: {e}")
        return None

# Load all three datasets
fixed_df = load_csv("WorkingCodeVersion1_FIXED_v10_8.csv")
dynamic_df = load_csv("WorkingCodeVersion1_DYNAMIC_v10_8.csv")
homer_df = load_csv("WorkingCodeVersion1_HOMER_v10_5.csv")

# Ensure all three datasets have the same number of rows before merging
min_rows = min(len(fixed_df), len(dynamic_df), len(homer_df))

fixed_df = fixed_df.iloc[:min_rows]  # Trim to match minimum row count
dynamic_df = dynamic_df.iloc[:min_rows]
homer_df = homer_df.iloc[:min_rows]

print(f"✅ Trimmed all datasets to {min_rows} rows for alignment.")




# Ensure all dataframes are valid
if fixed_df is None or dynamic_df is None or homer_df is None:
    raise ValueError("❌ ERROR: One or more CSV files could not be loaded. Check the file format.")

# =======================
# ALIGN DATA FOR COMPARISON
# =======================
# Merge all three datasets on "time"
# Ensure proper suffixes for HOMER dataset before merging


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

# Print available columns to verify correctness
print("\n📋 Available columns in comparison_df:", list(comparison_df.columns))


# =======================
# ANALYSIS: ENERGY IMPORT/EXPORT
# =======================
print("\n📊 **Energy Import/Export Comparison**")
comparison_df["import_diff_fixed_vs_dynamic"] = comparison_df["p_import_(kw)_fixed"] - comparison_df["p_import_(kw)_dynamic"]
comparison_df["import_diff_fixed_vs_homer"] = comparison_df["p_import_(kw)_fixed"] - comparison_df["p_import_(kw)_homer"]
comparison_df["export_diff_fixed_vs_dynamic"] = comparison_df["p_export_(kw)_fixed"] - comparison_df["p_export_(kw)_dynamic"]
comparison_df["export_diff_fixed_vs_homer"] = comparison_df["p_export_(kw)_fixed"] - comparison_df["p_export_(kw)_homer"]

print(comparison_df[["time", "import_diff_fixed_vs_dynamic", "import_diff_fixed_vs_homer",
                     "export_diff_fixed_vs_dynamic", "export_diff_fixed_vs_homer"]].describe())



# =======================
# ANALYSIS: STATE OF CHARGE (SOC)
# =======================
print("\n⚡ **Battery SOC Analysis**")
comparison_df["soc_diff_fixed_vs_dynamic"] = comparison_df["soc_(%)_fixed"] - comparison_df["soc_(%)_dynamic"]
comparison_df["soc_diff_fixed_vs_homer"] = comparison_df["soc_(%)_fixed"] - comparison_df["soc_(%)_homer"]

print(comparison_df[["time", "soc_diff_fixed_vs_dynamic", "soc_diff_fixed_vs_homer"]].describe())

# =======================
# VISUALIZATION
# =======================
def plot_comparison(x, y1, y2, y3, ylabel, title):
    """Plots Fixed vs Dynamic vs HOMER results."""
    plt.figure(figsize=(12, 5))
    plt.plot(x, y1, label="Fixed", linestyle="-")
    plt.plot(x, y2, label="Dynamic", linestyle="--")
    plt.plot(x, y3, label="HOMER", linestyle=":")
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

# Plot Energy Import Comparison
plot_comparison(comparison_df["time"], 
                comparison_df["p_import_(kw)_fixed"], 
                comparison_df["p_import_(kw)_dynamic"], 
                comparison_df["p_import_(kw)_homer"], 
                "Power (kW)", "Energy Import Comparison")

# Plot Energy Export Comparison
plot_comparison(comparison_df["time"], 
                comparison_df["p_export_(kw)_fixed"], 
                comparison_df["p_export_(kw)_dynamic"], 
                comparison_df["p_export_(kw)_homer"], 
                "Power (kW)", "Energy Export Comparison")

# Plot Battery Charge Power
plot_comparison(comparison_df["time"], 
                comparison_df["p_bat_ch_(kw)_fixed"], 
                comparison_df["p_bat_ch_(kw)_dynamic"], 
                comparison_df["p_bat_ch_(kw)_homer"], 
                "Power (kW)", "Battery Charging Comparison")

# Plot Battery Discharge Power
plot_comparison(comparison_df["time"], 
                comparison_df["p_bat_dis_(kw)_fixed"], 
                comparison_df["p_bat_dis_(kw)_dynamic"], 
                comparison_df["p_bat_dis_(kw)_homer"], 
                "Power (kW)", "Battery Discharging Comparison")

# Plot SOC Comparison
plot_comparison(comparison_df["time"], 
                comparison_df["soc_(%)_fixed"], 
                comparison_df["soc_(%)_dynamic"], 
                comparison_df["soc_(%)_homer"], 
                "SOC (%)", "Battery State of Charge (SOC) Comparison")

# =======================
# SAVE ANALYSIS RESULTS
# =======================
comparison_df.to_csv("Comparison_Results.csv", index=False)
print("\n✅ Comparison analysis saved as 'Comparison_Results.csv'")
