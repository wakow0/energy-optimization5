import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime


def plot_results(results_file="WorkingCodeVersion1_FIXED_v4.csv"):
    """Generate and save plots automatically from optimization results."""

    if not os.path.exists(results_file):
        raise FileNotFoundError(f"❌ ERROR: '{results_file}' is missing.")

    print(f"✅ Loading results from {results_file} for plotting...")

    # Load the results
    df = pd.read_csv(results_file)

    # Standardize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )

    print("\n✅ DEBUG: Columns in CSV file:")
    print(df.columns.tolist())

    # ✅ Now your columns are:
    # ['time', 'p_import_kw', 'p_export_kw', 'p_bat_ch_kw', 'p_bat_dis_kw', 'soc_%']

    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"plots/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output folder created: {output_dir}")

    # Plot 1: Battery Discharge vs. Grid Purchases
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["p_bat_dis_kw"], label="Battery Discharge", color="blue")
    plt.plot(df["time"], df["p_import_kw"], label="Grid Purchases", color="red")
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.title("Battery Discharge vs. Grid Purchases")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.savefig(f"{output_dir}/battery_vs_grid.png")
    plt.close()

    # Plot 2: Grid Purchases vs. Sales
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["p_import_kw"], label="Grid Purchases", color="green")
    plt.plot(df["time"], df["p_export_kw"], label="Grid Sales", color="purple")
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.title("Grid Purchases vs. Sales")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.savefig(f"{output_dir}/grid_purchases_vs_sales.png")
    plt.close()

    # Plot 3: All Metrics Overview
    metrics = ["p_import_kw", "p_export_kw", "p_bat_ch_kw", "p_bat_dis_kw", "soc_%"]
    colors = ["blue", "red", "green", "orange", "purple"]
    titles = [
        "Grid Import",
        "Grid Export",
        "Battery Charging",
        "Battery Discharging",
        "Battery State of Charge (SOC)",
    ]

    plt.figure(figsize=(12, 12))
    for i, (metric, color, title) in enumerate(zip(metrics, colors, titles)):
        plt.subplot(5, 1, i + 1)
        plt.plot(df["time"], df[metric], label=title, color=color)
        plt.xlabel("Time")
        plt.ylabel(metric)
        plt.title(title)
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(f"{output_dir}/all_metrics.png")
    plt.close()

    print(f"✅ All plots saved in '{output_dir}/'.")


if __name__ == "__main__":
    plot_results()
