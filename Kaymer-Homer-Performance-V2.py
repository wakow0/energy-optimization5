import pandas as pd

# ✅ Load necessary constants from Parameters.txt
FIXED_IMPORT_PRICE = 0.1939  # £/kWh
FIXED_EXPORT_PRICE = 0.0769  # £/kWh

# ✅ Load the CSV files
opt_results_fixed = pd.read_csv("solution_output_FIXED_v9.csv")
opt_results_dynamic = pd.read_csv("solution_output_DYNAMIC_v9.csv")
opt_results_homer = pd.read_csv("WorkingCodeVersion1_HOMER_v10_5.csv")
processed_data = pd.read_csv("processed_data.csv")

#df = pd.read_csv("research.csv", skiprows=3)

# ✅ Ensure consistent column names
#df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
opt_results_fixed.columns = opt_results_fixed.columns.str.strip().str.lower().str.replace(" ", "_")
opt_results_dynamic.columns = opt_results_dynamic.columns.str.strip().str.lower().str.replace(" ", "_")
opt_results_homer.columns = opt_results_homer.columns.str.strip().str.lower().str.replace(" ", "_")

# ✅ Align lengths
min_length = min(len(opt_results_fixed), len(opt_results_dynamic), len(processed_data))
opt_results_fixed = opt_results_fixed.iloc[:min_length].reset_index(drop=True)
opt_results_dynamic = opt_results_dynamic.iloc[:min_length].reset_index(drop=True)
opt_results_homer = opt_results_homer.iloc[:min_length].reset_index(drop=True)
#df = df.iloc[:min_length].reset_index(drop=True)

# ✅ Fixed Price Strategy
print(f"\nSelected Strategy: Fixed Price Strategy")

total_battery_activity = opt_results_fixed["p_bat_ch_(kw)"].sum() + opt_results_fixed["p_bat_dis_(kw)"].sum()
total_energy_demand = opt_results_fixed["p_import_(kw)"].sum() + opt_results_fixed["p_export_(kw)"].sum() + total_battery_activity
battery_usage_percentage = (total_battery_activity / total_energy_demand) * 100 if total_energy_demand > 0 else 0
print(f"🔋 Battery Usage Percentage: {battery_usage_percentage:.2f}%")

battery_savings_fixed = opt_results_fixed["p_bat_dis_(kw)"].sum() * FIXED_IMPORT_PRICE
renewable_savings_fixed = (processed_data["dc_ground_1500vdc_power_output"].sum() + processed_data["windflow_33_[500kw]_power_output"].sum()) * FIXED_IMPORT_PRICE
total_export_revenue_fixed = opt_results_fixed["p_export_(kw)"].sum() * FIXED_EXPORT_PRICE
total_import_cost_fixed = opt_results_fixed["p_import_(kw)"].sum() * FIXED_IMPORT_PRICE
total_cost_saving_fixed = battery_savings_fixed + renewable_savings_fixed + total_export_revenue_fixed - total_import_cost_fixed

print(f"🔋 Cost Savings from Battery Usage: £{battery_savings_fixed:.2f}")
print(f"☀️ Cost Savings from Renewable Energy: £{renewable_savings_fixed:.2f}")
print(f"💰 Total Export Revenue: £{total_export_revenue_fixed:.2f}")
print(f"⚡ Total Import Cost: £{total_import_cost_fixed:.2f}")
print(f"📊 Total Cost Saving: £{total_cost_saving_fixed:.2f}")

# ✅ Dynamic Price Strategy
print(f"\nSelected Strategy: Dynamic Price Strategy")

total_battery_activity = opt_results_dynamic["p_bat_ch_(kw)"].sum() + opt_results_dynamic["p_bat_dis_(kw)"].sum()
total_energy_demand = opt_results_dynamic["p_import_(kw)"].sum() + opt_results_dynamic["p_export_(kw)"].sum() + total_battery_activity
battery_usage_percentage = (total_battery_activity / total_energy_demand) * 100 if total_energy_demand > 0 else 0
print(f"🔋 Battery Usage Percentage: {battery_usage_percentage:.2f}%")

battery_savings_dynamic = (opt_results_dynamic["p_bat_dis_(kw)"] * processed_data["total_consumption_rate"]).sum()
renewable_savings_dynamic = (processed_data["dc_ground_1500vdc_power_output"] + processed_data["windflow_33_[500kw]_power_output"]).sum() * processed_data["total_consumption_rate"].mean()
total_export_revenue_dynamic = (opt_results_dynamic["p_export_(kw)"] * processed_data["grid_sellback_rate"]).sum()
total_import_cost_dynamic = (opt_results_dynamic["p_import_(kw)"] * processed_data["total_consumption_rate"]).sum()
total_cost_saving_dynamic = battery_savings_dynamic + renewable_savings_dynamic + total_export_revenue_dynamic - total_import_cost_dynamic

print(f"🔋 Cost Savings from Battery Usage: £{battery_savings_dynamic:.2f}")
print(f"☀️ Cost Savings from Renewable Energy: £{renewable_savings_dynamic:.2f}")
print(f"💰 Total Export Revenue: £{total_export_revenue_dynamic:.2f}")
print(f"⚡ Total Import Cost: £{total_import_cost_dynamic:.2f}")
print(f"📊 Total Cost Saving: £{total_cost_saving_dynamic:.2f}")

# ✅ HOMER GRID Strategy
print(f"\nSelected Strategy: HOMER GRID")

total_battery_activity = opt_results_homer["p_bat_ch_(kw)"].sum() + opt_results_homer["p_bat_dis_(kw)"].sum()
total_energy_demand = opt_results_homer["p_import_(kw)"].sum() + opt_results_homer["p_export_(kw)"].sum() + total_battery_activity
battery_usage_percentage = (total_battery_activity / total_energy_demand) * 100 if total_energy_demand > 0 else 0
print(f"🔋 Battery Usage Percentage: {battery_usage_percentage:.2f}%")

battery_savings_homer = (opt_results_homer["p_bat_dis_(kw)"] * processed_data["total_consumption_rate"]).sum()
renewable_savings_homer = (processed_data["dc_ground_1500vdc_power_output"] + processed_data["windflow_33_[500kw]_power_output"]).sum() * processed_data["total_consumption_rate"].mean()
total_export_revenue_homer = (opt_results_homer["p_export_(kw)"] * processed_data["grid_sellback_rate"]).sum()
total_import_cost_homer = (opt_results_homer["p_import_(kw)"] * processed_data["total_consumption_rate"]).sum()
total_cost_saving_homer = battery_savings_homer + renewable_savings_homer + total_export_revenue_homer - total_import_cost_homer


print(f"🔋 Cost Savings from Battery Usage: £{battery_savings_homer:.2f}")
print(f"☀️ Cost Savings from Renewable Energy: £{renewable_savings_homer:.2f}")
print(f"💰 Total Export Revenue: £{total_export_revenue_homer:.2f}")
print(f"⚡ Total Import Cost: £{total_import_cost_homer:.2f}")
print(f"📊 Total Cost Saving: £{total_cost_saving_homer:.2f}")
