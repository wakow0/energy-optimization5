import pandas as pd
import matplotlib.pyplot as plt

# ✅ Load datasets
solution_a = pd.read_csv("WorkingCodeVersion1_DYNAMIC_v10_6.csv")
solution_b_lr = pd.read_csv("WorkingCodeVersion_LR_v3.csv")
solution_b_xg = pd.read_csv("WorkingCodeVersion_XG_v3.csv")
solution_b_lstm = pd.read_csv("WorkingCodeVersion_LSTM_v3.csv")

# ✅ Convert 'time' to datetime
for df in [solution_a, solution_b_lr, solution_b_xg, solution_b_lstm]:
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
for df in [solution_a, solution_b_lr, solution_b_xg, solution_b_lstm]:
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
    "Solution B (LR)": {
        "Total Import": solution_b_lr["import"].sum(),
        "Total Export": solution_b_lr["export"].sum(),
        "Total Charge": solution_b_lr["charge"].sum(),
        "Total Discharge": solution_b_lr["discharge"].sum(),
        "Average SOC": solution_b_lr["soc"].mean(),
    },
    "Solution B (XG)": {
        "Total Import": solution_b_xg["import"].sum(),
        "Total Export": solution_b_xg["export"].sum(),
        "Total Charge": solution_b_xg["charge"].sum(),
        "Total Discharge": solution_b_xg["discharge"].sum(),
        "Average SOC": solution_b_xg["soc"].mean(),
    },
    "Solution B (LSTM)": {
        "Total Import": solution_b_lstm["import"].sum(),
        "Total Export": solution_b_lstm["export"].sum(),
        "Total Charge": solution_b_lstm["charge"].sum(),
        "Total Discharge": solution_b_lstm["discharge"].sum(),
        "Average SOC": solution_b_lstm["soc"].mean(),
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
axs[0].plot(solution_b_lr["time"], solution_b_lr["import"], linestyle="-", label="Solution B (LR) - Import", color="green")
axs[0].plot(solution_b_xg["time"], solution_b_xg["import"], linestyle="-", label="Solution B (XG) - Import", color="red")
axs[0].plot(solution_b_lstm["time"], solution_b_lstm["import"], linestyle="-", label="Solution B (LSTM) - Import", color="orange")
axs[0].set_ylabel("Import (kW)")
axs[0].set_title("Grid Import Over Time")
axs[0].legend(loc="upper right")
axs[0].grid(True)

# 📈 Export Over Time
axs[1].plot(solution_a["time"], solution_a["export"], linestyle="--", label="Solution A - Export", color="blue")
axs[1].plot(solution_b_lr["time"], solution_b_lr["export"], linestyle="-", label="Solution B (LR) - Export", color="green")
axs[1].plot(solution_b_xg["time"], solution_b_xg["export"], linestyle="-", label="Solution B (XG) - Export", color="red")
axs[1].plot(solution_b_lstm["time"], solution_b_lstm["export"], linestyle="-", label="Solution B (LSTM) - Export", color="orange")
axs[1].set_ylabel("Export (kW)")
axs[1].set_title("Grid Export Over Time")
axs[1].legend(loc="upper right")
axs[1].grid(True)

# 🔋 Charge Over Time
axs[2].plot(solution_a["time"], solution_a["charge"], linestyle="--", label="Solution A - Charge", color="blue")
axs[2].plot(solution_b_lr["time"], solution_b_lr["charge"], linestyle="-", label="Solution B (LR) - Charge", color="green")
axs[2].plot(solution_b_xg["time"], solution_b_xg["charge"], linestyle="-", label="Solution B (XG) - Charge", color="red")
axs[2].plot(solution_b_lstm["time"], solution_b_lstm["charge"], linestyle="-", label="Solution B (LSTM) - Charge", color="orange")
axs[2].set_ylabel("Charge (kW)")
axs[2].set_title("Battery Charging Over Time")
axs[2].legend(loc="upper right")
axs[2].grid(True)

# 🔋 Discharge Over Time
axs[3].plot(solution_a["time"], solution_a["discharge"], linestyle="--", label="Solution A - Discharge", color="blue")
axs[3].plot(solution_b_lr["time"], solution_b_lr["discharge"], linestyle="-", label="Solution B (LR) - Discharge", color="green")
axs[3].plot(solution_b_xg["time"], solution_b_xg["discharge"], linestyle="-", label="Solution B (XG) - Discharge", color="red")
axs[3].plot(solution_b_lstm["time"], solution_b_lstm["discharge"], linestyle="-", label="Solution B (LSTM) - Discharge", color="orange")
axs[3].set_ylabel("Discharge (kW)")
axs[3].set_title("Battery Discharging Over Time")
axs[3].legend(loc="upper right")
axs[3].grid(True)

# 📈 SOC Over Time
axs[4].plot(solution_a["time"], solution_a["soc"], linestyle="--", label="Solution A - SOC", color="blue")
axs[4].plot(solution_b_lr["time"], solution_b_lr["soc"], linestyle="-", label="Solution B (LR) - SOC", color="green")
axs[4].plot(solution_b_xg["time"], solution_b_xg["soc"], linestyle="-", label="Solution B (XG) - SOC", color="red")
axs[4].plot(solution_b_lstm["time"], solution_b_lstm["soc"], linestyle="-", label="Solution B (LSTM) - SOC", color="orange")
axs[4].set_ylabel("SOC (%)")
axs[4].set_title("State of Charge (SOC) Over Time")
axs[4].legend(loc="upper right")
axs[4].grid(True)

# ✅ Save & Show Plots
plt.savefig("Full_Grid_Comparison_with_SOC.png")
plt.show()
