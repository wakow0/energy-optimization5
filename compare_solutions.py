import pandas as pd
import matplotlib.pyplot as plt

# ✅ Define Correct Column Mappings (For Both Solution A & Solution B)
column_mapping = {
    "P_import (kW)": "import",
    "P_export (kW)": "export",
    "P_bat_ch (kW)": "charge",
    "P_bat_dis (kW)": "discharge",
    "SOC (%)": "soc"
}

# ✅ Load Solution A
solution_a = pd.read_csv("WorkingCodeVersion1_DYNAMIC_v10_5.csv")

# ✅ Load datasets
solution_b_lr = pd.read_csv("WorkingCodeVersion_LR_v1.csv")
solution_b_xg = pd.read_csv("WorkingCodeVersion_XG_v1.csv")
solution_b_lstm = pd.read_csv("WorkingCodeVersion_LSTM_v1.csv")

# ✅ Ensure 'time' is converted to datetime
solution_a["time"] = pd.to_datetime(solution_a["time"])
solution_b_lr["time"] = pd.to_datetime(solution_b_lr["time"])
solution_b_xg["time"] = pd.to_datetime(solution_b_xg["time"])
solution_b_lstm["time"] = pd.to_datetime(solution_b_lstm["time"])

# ✅ Check for datetime conversion success
print("✅ Time column types:", solution_a["time"].dtype, solution_b_lr["time"].dtype)

# ✅ Rename columns in Solution A to match expected names
solution_a.rename(columns=column_mapping, inplace=True)

# ✅ Ensure Solution A has the correct columns
print("✅ Columns in Solution A after renaming:", solution_a.columns.tolist())

# ✅ Load and Fix Solution B CSVs
def load_and_clean_solution_b(filename):
    df = pd.read_csv(filename)

    # ✅ Rename columns in Solution B
    df.rename(columns=column_mapping, inplace=True)

    # ✅ Merge Solution B with Solution A using the 'time' column
    df = pd.concat([solution_a[['time']], df], axis=1)

    return df

solution_b_lr = load_and_clean_solution_b("WorkingCodeVersion_LR_v1.csv")
solution_b_xg = load_and_clean_solution_b("WorkingCodeVersion_XG_v1.csv")
solution_b_lstm = load_and_clean_solution_b("WorkingCodeVersion_LSTM_v1.csv")

# ✅ Ensure Solution B is in "good condition" (Filter out negative values)
def filter_good_condition(df):
    return df[(df["import"] >= 0) & (df["export"] >= 0) & 
              (df["charge"] >= 0) & (df["discharge"] >= 0) & 
              (df["soc"] >= 5)]

solution_b_lr = filter_good_condition(solution_b_lr)
solution_b_xg = filter_good_condition(solution_b_xg)
solution_b_lstm = filter_good_condition(solution_b_lstm)

print(f"✅ Filtered Solution B - LR: {len(solution_b_lr)} rows")
print(f"✅ Filtered Solution B - XGBoost: {len(solution_b_xg)} rows")
print(f"✅ Filtered Solution B - LSTM: {len(solution_b_lstm)} rows")

# ✅ Save the cleaned versions
solution_b_lr.to_csv("Filtered_WorkingCodeVersion_LR_v1.csv", index=False)
solution_b_xg.to_csv("Filtered_WorkingCodeVersion_XG_v1.csv", index=False)
solution_b_lstm.to_csv("Filtered_WorkingCodeVersion_LSTM_v1.csv", index=False)

print("✅ Filtered Solution B versions saved.")

# ✅ Plot comparisons
def plot_comparison(metric, ylabel, title, filename):
    plt.figure(figsize=(12, 6))
    
    # ✅ Plot each solution for the current metric
    plt.plot(solution_a["time"], solution_a[metric], linestyle="--", label=f"Solution A - {metric.capitalize()}", color="blue")
    plt.plot(solution_b_lr["time"], solution_b_lr[metric], linestyle="-", label=f"Solution B (LR) - {metric.capitalize()}", color="green")
    plt.plot(solution_b_xg["time"], solution_b_xg[metric], linestyle="-", label=f"Solution B (XG) - {metric.capitalize()}", color="red")
    plt.plot(solution_b_lstm["time"], solution_b_lstm[metric], linestyle="-", label=f"Solution B (LSTM) - {metric.capitalize()}", color="orange")

    # ✅ Labels and title
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)

    # ✅ Ensure only one legend is added per figure
    plt.legend(loc="upper left")
    plt.grid(True)

    # ✅ Save and close plot
    plt.savefig(filename)
    plt.show()
    plt.close()  # Ensure figure is closed to prevent overlapping legends

# ✅ Plot for each metric
plot_comparison("import", "Power (kW)", "Comparison of Grid Import", "Comparison_Import.png")

metrics = ["export", "charge", "discharge", "soc"]
for metric in metrics:
    ylabel = "Power (kW)" if metric != "soc" else "SOC (%)"
    title = f"Comparison of {metric.capitalize()}"
    filename = f"Comparison_{metric.capitalize()}.png"
    plot_comparison(metric, ylabel, title, filename)

print("✅ All comparisons plotted and saved.")