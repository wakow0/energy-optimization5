# Re-import necessary libraries since execution state was reset
import pandas as pd
import numpy as np

# Load the latest Fixed and Dynamic pricing solution files for analysis
fixed_file = "solution_output_FIXED_v15.csv"
dynamic_file = "solution_output_DYNAMIC_v15.csv"

# Read the CSV files
df_fixed = pd.read_csv(fixed_file)
df_dynamic = pd.read_csv(dynamic_file)

# Convert time column to datetime for better analysis
df_fixed["time"] = pd.to_datetime(df_fixed["time"])
df_dynamic["time"] = pd.to_datetime(df_dynamic["time"])

# Merge Fixed and Dynamic data for direct comparison
df_comparison = df_fixed.copy()
df_comparison = df_comparison.rename(columns={
    "P_import (kW)": "Fixed_P_import",
    "P_export (kW)": "Fixed_P_export",
    "P_bat_ch (kW)": "Fixed_P_bat_ch",
    "P_bat_dis (kW)": "Fixed_P_bat_dis",
    "SOC (%)": "Fixed_SOC"
})

df_comparison = df_comparison.merge(df_dynamic, on="time", suffixes=("", "_Dynamic"))

# Calculate differences between Fixed and Dynamic Pricing
df_comparison["Import_Diff (kW)"] = df_comparison["P_import (kW)"] - df_comparison["Fixed_P_import"]
df_comparison["Export_Diff (kW)"] = df_comparison["P_export (kW)"] - df_comparison["Fixed_P_export"]
df_comparison["Charge_Diff (kW)"] = df_comparison["P_bat_ch (kW)"] - df_comparison["Fixed_P_bat_ch"]
df_comparison["Discharge_Diff (kW)"] = df_comparison["P_bat_dis (kW)"] - df_comparison["Fixed_P_bat_dis"]
df_comparison["SOC_Diff (%)"] = df_comparison["SOC (%)"] - df_comparison["Fixed_SOC"]

# Display key summary statistics for Fixed vs. Dynamic Pricing
summary_stats = df_comparison[["Import_Diff (kW)", "Export_Diff (kW)", "Charge_Diff (kW)", "Discharge_Diff (kW)", "SOC_Diff (%)"]].describe()

# Display comparative analysis results
# Display comparative analysis results using Pandas instead of ace_tools
import matplotlib.pyplot as plt

# Plot the differences between Fixed and Dynamic pricing for key parameters
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(df_comparison["time"], df_comparison["Import_Diff (kW)"], label="Import Difference", color="blue")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.ylabel("Import Diff (kW)")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(df_comparison["time"], df_comparison["Export_Diff (kW)"], label="Export Difference", color="green")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.ylabel("Export Diff (kW)")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(df_comparison["time"], df_comparison["Charge_Diff (kW)"], label="Charge Difference", color="orange")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.ylabel("Charge Diff (kW)")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(df_comparison["time"], df_comparison["SOC_Diff (%)"], label="SOC Difference", color="red")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.ylabel("SOC Diff (%)")
plt.legend()

plt.suptitle("Comparison of Fixed vs Dynamic Pricing Differences")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
