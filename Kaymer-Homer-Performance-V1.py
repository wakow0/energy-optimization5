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

        return df
    except Exception as e:
        print(f"❌ ERROR loading {file_path}: {e}")
        return None

# Load all three datasets
fixed_df = load_csv("WorkingCodeVersion1_FIXED_v7.csv")
dynamic_df = load_csv("WorkingCodeVersion1_DYNAMIC_v7.csv")
homer_df = load_csv("WorkingCodeVersion1_HOMER_v7.csv")

# Ensure all dataframes are valid
if fixed_df is None or dynamic_df is None or homer_df is None:
    raise ValueError("❌ ERROR: One or more CSV files could not be loaded. Check the file format.")

# =======================
# ALIGN DATA FOR COMPARISON
# =======================
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

# Print available columns to verify correctness
print("\n📋 Available columns in comparison_df:", list(comparison_df.columns))











# =======================
# PERFORMANCE METRICS
# =======================

# 🔹 Cost Savings (%)
cost_savings = ((comparison_df["p_import_(kw)_homer"].sum() - comparison_df["p_import_(kw)_fixed"].sum()) / 
                comparison_df["p_import_(kw)_homer"].sum()) * 100

# 🔹 Grid Dependency Reduction (%)
grid_reduction = ((comparison_df["p_import_(kw)_homer"].sum() - comparison_df["p_import_(kw)_dynamic"].sum()) / 
                  comparison_df["p_import_(kw)_homer"].sum()) * 100

# 🔹 Battery Utilization
battery_utilization_fixed = (comparison_df["p_bat_dis_(kw)_fixed"].sum() / comparison_df["p_import_(kw)_fixed"].sum()) * 100
battery_utilization_dynamic = (comparison_df["p_bat_dis_(kw)_dynamic"].sum() / comparison_df["p_import_(kw)_dynamic"].sum()) * 100
battery_utilization_homer = (comparison_df["p_bat_dis_(kw)_homer"].sum() / comparison_df["p_import_(kw)_homer"].sum()) * 100

# 🔹 Revenue Comparison
revenue_fixed = comparison_df["p_export_(kw)_fixed"].sum()
revenue_dynamic = comparison_df["p_export_(kw)_dynamic"].sum()
revenue_homer = comparison_df["p_export_(kw)_homer"].sum()

# 🔹 Compute Battery Utilization Difference
battery_utilization_diff_fixed = battery_utilization_fixed - battery_utilization_homer
battery_utilization_diff_dynamic = battery_utilization_dynamic - battery_utilization_homer

# ✅ Print the results
print("\n=== Key Performance Metrics ===")
print(f"🔹 Cost Savings: {cost_savings:.2f}%")
print(f"🔹 Grid Dependency Reduction: {grid_reduction:.2f}%")
print(f"🔹 Battery Utilization (Fixed): {battery_utilization_fixed:.2f}%")
print(f"🔹 Battery Utilization (Dynamic): {battery_utilization_dynamic:.2f}%")
print(f"🔹 Battery Utilization (HOMER): {battery_utilization_homer:.2f}%")
print(f"🔹 Revenue (Fixed): {revenue_fixed:.2f} kW sold")
print(f"🔹 Revenue (Dynamic): {revenue_dynamic:.2f} kW sold")
print(f"🔹 Revenue (HOMER): {revenue_homer:.2f} kW sold")
print("================================\n")

# =======================
# DETERMINE OVERALL WINNER
# =======================
print("\n==== OVERALL WINNER ====")

# 🔹 Cost Comparison
if cost_savings > 0:
    print(f"✅ Kaymer REDUCES total energy costs compared to HOMER by {cost_savings:.2f}%")
elif cost_savings < 0:
    print(f"❌ Kaymer INCREASES total energy costs by {abs(cost_savings):.2f}%")
else:
    print("🔸 Kaymer and HOMER have the SAME total energy costs.")

# 🔹 Grid Dependency Comparison
if grid_reduction > 0:
    print(f"✅ Kaymer REDUCES GRID DEPENDENCE compared to HOMER by {grid_reduction:.2f}%")
elif grid_reduction < 0:
    print(f"❌ Kaymer RELIES MORE on the grid by {abs(grid_reduction):.2f}%")
else:
    print("🔸 Kaymer and HOMER have the SAME level of grid dependence.")

# 🔹 Revenue Comparison
if revenue_fixed > revenue_homer:
    print(f"✅ Fixed Strategy GENERATES MORE REVENUE than HOMER ({revenue_fixed - revenue_homer:.2f} kW more)")
elif revenue_fixed < revenue_homer:
    print(f"❌ HOMER GENERATES MORE REVENUE than Fixed Strategy ({revenue_homer - revenue_fixed:.2f} kW more)")
else:
    print("🔸 Fixed and HOMER have the SAME revenue.")

if revenue_dynamic > revenue_homer:
    print(f"✅ Dynamic Strategy GENERATES MORE REVENUE than HOMER ({revenue_dynamic - revenue_homer:.2f} kW more)")
elif revenue_dynamic < revenue_homer:
    print(f"❌ HOMER GENERATES MORE REVENUE than Dynamic Strategy ({revenue_homer - revenue_dynamic:.2f} kW more)")
else:
    print("🔸 Dynamic and HOMER have the SAME revenue.")

# 🔹 Battery Utilization Comparison
if battery_utilization_diff_fixed > 0:
    print(f"✅ Fixed Strategy UTILIZES battery storage MORE than HOMER by {battery_utilization_diff_fixed:.2f}%")
elif battery_utilization_diff_fixed < 0:
    print(f"❌ Fixed Strategy UNDERUTILIZES battery storage compared to HOMER by {abs(battery_utilization_diff_fixed):.2f}%")

if battery_utilization_diff_dynamic > 0:
    print(f"✅ Dynamic Strategy UTILIZES battery storage MORE than HOMER by {battery_utilization_diff_dynamic:.2f}%")
elif battery_utilization_diff_dynamic < 0:
    print(f"❌ Dynamic Strategy UNDERUTILIZES battery storage compared to HOMER by {abs(battery_utilization_diff_dynamic):.2f}%")

print("======================================================")
