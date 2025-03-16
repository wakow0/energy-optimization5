import pandas as pd
import matplotlib.pyplot as plt

# ✅ Load datasets
solution_a = pd.read_csv("WorkingCodeVersion1_DYNAMIC_v10_6.csv")
solution_FIXED = pd.read_csv("generated_decision_variables_FIXED_v3.csv")
solution_DYNAMIC = pd.read_csv("generated_decision_variables_DYNAMIC_v3.csv")
solution_HOMER = pd.read_csv("WorkingCodeVersion1_HOMER_v10_6.csv")

# ✅ Convert 'time' to datetime
for df in [solution_a, solution_FIXED, solution_DYNAMIC, solution_HOMER]:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

# ✅ Define Column Mapping
column_mapping = {
    "P_import (kW)": "import",
    "P_export (kW)": "export",
    "P_bat_ch (kW)": "charge",
    "P_bat_dis (kW)": "discharge",
    "SOC (%)": "soc"
}

# ✅ Rename Columns for Consistency
for df in [solution_a, solution_FIXED, solution_DYNAMIC, solution_HOMER]:
    df.rename(columns=column_mapping, inplace=True)

# ✅ Compute Total Import, Export, Charge, Discharge, and SOC for Each Solution
metrics = {
    "Solution A": {
        "Total Import": solution_a["import"].sum(),
        "Total Export": solution_a["export"].sum(),
        "Total Charge": solution_a["charge"].sum(),
        "Total Discharge": solution_a["discharge"].sum(),
        "Average SOC": solution_a["soc"].mean(),
    },
    "Solution (Fixed)": {
        "Total Import": solution_FIXED["import"].sum(),
        "Total Export": solution_FIXED["export"].sum(),
        "Total Charge": solution_FIXED["charge"].sum(),
        "Total Discharge": solution_FIXED["discharge"].sum(),
        "Average SOC": solution_FIXED["soc"].mean(),
    },
    "Solution (Dynamic)": {
        "Total Import": solution_DYNAMIC["import"].sum(),
        "Total Export": solution_DYNAMIC["export"].sum(),
        "Total Charge": solution_DYNAMIC["charge"].sum(),
        "Total Discharge": solution_DYNAMIC["discharge"].sum(),
        "Average SOC": solution_DYNAMIC["soc"].mean(),
    },
    "Solution (HOMER)": {
        "Total Import": solution_HOMER["import"].sum(),
        "Total Export": solution_HOMER["export"].sum(),
        "Total Charge": solution_HOMER["charge"].sum(),
        "Total Discharge": solution_HOMER["discharge"].sum(),
        "Average SOC": solution_HOMER["soc"].mean(),
    }
}

# ✅ Convert to DataFrame for Display
df_metrics = pd.DataFrame(metrics).T
df_metrics["Import Reduction"] = df_metrics["Total Import"].max() - df_metrics["Total Import"]
df_metrics["Export Increase"] = df_metrics["Total Export"] - df_metrics["Total Export"].min()
df_metrics["Overall Score"] = df_metrics["Export Increase"] - df_metrics["Import Reduction"]

# ✅ Determine Best Solution
best_solution = df_metrics["Overall Score"].idxmax()

print("\n📊 **Comparison Metrics:**")
print(df_metrics)
print(f"\n🏆 Best Solution: **{best_solution}** (Lowest Import & Highest Export)")

# ✅ 📊 Bar Chart: Total Import, Export, Charge, Discharge, and SOC Comparison
df_metrics[["Total Import", "Total Export", "Total Charge", "Total Discharge", "Average SOC"]].plot(
    kind="bar", figsize=(10, 6), colormap="coolwarm", edgecolor="black"
)
plt.ylabel("Total Energy (kW) / Average SOC (%)")
plt.title("Comparison of Total Import, Export, Charge, Discharge, and SOC")
plt.xticks(rotation=0)
plt.legend(loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("Total_Comparison_with_SOC.png")
plt.show()

# ✅ 📈 Import, Export, Charge, Discharge & SOC Trends Over Time
fig, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True)

# 📉 Import Over Time
axs[0].plot(solution_a["time"], solution_a["import"], linestyle="--", label="Solution A - Import", color="blue")
axs[0].plot(solution_FIXED["time"], solution_FIXED["import"], linestyle="-", label="Solution (Fixed) - Import", color="green")
axs[0].plot(solution_DYNAMIC["time"], solution_DYNAMIC["import"], linestyle="-", label="Solution (Dynamic) - Import", color="red")
axs[0].plot(solution_HOMER["time"], solution_HOMER["import"], linestyle="-", label="Solution (HOMER) - Import", color="orange")
axs[0].set_ylabel("Import (kW)")
axs[0].set_title("Grid Import Over Time")
axs[0].legend(loc="upper right")
axs[0].grid(True)

# 📈 Export Over Time
axs[1].plot(solution_a["time"], solution_a["export"], linestyle="--", label="Solution A - Export", color="blue")
axs[1].plot(solution_FIXED["time"], solution_FIXED["export"], linestyle="-", label="Solution (Fixed) - Export", color="green")
axs[1].plot(solution_DYNAMIC["time"], solution_DYNAMIC["export"], linestyle="-", label="Solution (Dynamic) - Export", color="red")
axs[1].plot(solution_HOMER["time"], solution_HOMER["export"], linestyle="-", label="Solution (HOMER) - Export", color="orange")
axs[1].set_ylabel("Export (kW)")
axs[1].set_title("Grid Export Over Time")
axs[1].legend(loc="upper right")
axs[1].grid(True)

# 🔋 Charge Over Time
axs[2].plot(solution_a["time"], solution_a["charge"], linestyle="--", label="Solution A - Charge", color="blue")
axs[2].plot(solution_FIXED["time"], solution_FIXED["charge"], linestyle="-", label="Solution (Fixed) - Charge", color="green")
axs[2].plot(solution_DYNAMIC["time"], solution_DYNAMIC["charge"], linestyle="-", label="Solution (Dynamic) - Charge", color="red")
axs[2].plot(solution_HOMER["time"], solution_HOMER["charge"], linestyle="-", label="Solution (HOMER) - Charge", color="orange")
axs[2].set_ylabel("Charge (kW)")
axs[2].set_title("Battery Charging Over Time")
axs[2].legend(loc="upper right")
axs[2].grid(True)

# 🔋 Discharge Over Time
axs[3].plot(solution_a["time"], solution_a["discharge"], linestyle="--", label="Solution A - Discharge", color="blue")
axs[3].plot(solution_FIXED["time"], solution_FIXED["discharge"], linestyle="-", label="Solution (Fixed) - Discharge", color="green")
axs[3].plot(solution_DYNAMIC["time"], solution_DYNAMIC["discharge"], linestyle="-", label="Solution (Dynamic) - Discharge", color="red")
axs[3].plot(solution_HOMER["time"], solution_HOMER["discharge"], linestyle="-", label="Solution (HOMER) - Discharge", color="orange")
axs[3].set_ylabel("Discharge (kW)")
axs[3].set_title("Battery Discharging Over Time")
axs[3].legend(loc="upper right")
axs[3].grid(True)

# 📈 SOC Over Time
axs[4].plot(solution_a["time"], solution_a["soc"], linestyle="--", label="Solution A - SOC", color="blue")
axs[4].plot(solution_FIXED["time"], solution_FIXED["soc"], linestyle="-", label="Solution (Fixed) - SOC", color="green")
axs[4].plot(solution_DYNAMIC["time"], solution_DYNAMIC["soc"], linestyle="-", label="Solution (Dynamic) - SOC", color="red")
axs[4].plot(solution_HOMER["time"], solution_HOMER["soc"], linestyle="-", label="Solution (HOMER) - SOC", color="orange")
axs[4].set_ylabel("SOC (%)")
axs[4].set_title("State of Charge (SOC) Over Time")
axs[4].legend(loc="upper right")
axs[4].grid(True)

# ✅ Save & Show Plots
plt.savefig("Full_Grid_Comparison_with_SOC.png")
plt.show()
