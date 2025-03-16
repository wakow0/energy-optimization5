import pandas as pd
import numpy as np
import time  
from tqdm import tqdm  

# Load the extracted input data
file_path = "extracted_inputs.csv"
df = pd.read_csv(file_path)

# Battery & Grid Constraints
battery_capacity = 2000  # kWh
battery_max_charge = 1000  # kW
battery_max_discharge = 1000  # kW
inverter_capacity = 1000  # kW
soc_min = 5  # 5% SoC limit
soc_max = 100  # 100% SoC
eta_ch = 0.95  # Charging efficiency
eta_dis = 0.95  # Discharging efficiency

# Define Fixed Pricing Constants
FIXED_IMPORT_PRICE = 0.1939  # £/kWh
FIXED_EXPORT_PRICE = 0.0769  # £/kWh

# Determine price thresholds for dynamic pricing adjustments
Q1_import = df["total_consumption_rate"].quantile(0.25)  
Q3_import = df["total_consumption_rate"].quantile(0.75)  
Q1_export = df["grid_sellback_rate"].quantile(0.25)  
Q3_export = df["grid_sellback_rate"].quantile(0.75)  

# Initialize separate DataFrames for Fixed & Dynamic pricing
df_fixed = df.copy()
df_dynamic = df.copy()

# Initialize new decision variable columns for both strategies
for df_version in [df_fixed, df_dynamic]:
    df_version["P_import (kW)"] = 0.0
    df_version["P_export (kW)"] = 0.0
    df_version["P_bat_ch (kW)"] = 0.0
    df_version["P_bat_dis (kW)"] = 0.0
    df_version["SOC (%)"] = np.nan  # Initialize SoC in percentage

# Set initial SoC using the previous day's last value
df_fixed.loc[0, "SOC (%)"] = 50.0
df_dynamic.loc[0, "SOC (%)"] = 50.0

# Time Step: 30 minutes (0.5 hours)
time_step = 0.5

# Start time measurement
start_time = time.time()

# Process each time step with a progress bar
for t in tqdm(range(1, len(df)), desc="Processing Fixed & Dynamic Solutions (v4)", unit="step"):
    for strategy, df_version in [("Fixed", df_fixed), ("Dynamic", df_dynamic)]:
        available_renewable = df_version.loc[t, "dc_ground_1500vdc_power_output"] + df_version.loc[t, "windflow_33_[500kw]_power_output"]
        demand = df_version.loc[t, "ac_primary_load"]
        
        # Determine Pricing (Fixed vs Dynamic)
        if strategy == "Fixed":
            import_price = FIXED_IMPORT_PRICE
            export_price = FIXED_EXPORT_PRICE
        else:  # Dynamic Pricing
            import_price = df_version.loc[t, "total_consumption_rate"]
            export_price = df_version.loc[t, "grid_sellback_rate"]

        # Apply dynamic pricing adjustments
        reduce_import = import_price >= Q3_import  # Reduce import if price is high
        prioritize_export = export_price >= Q3_export  # Increase export if sellback price is high
        encourage_charge = import_price <= Q1_import  # Charge when import price is low

        # Charge if excess renewable is available and import price is low
        charge_power = min(available_renewable, battery_max_charge) if encourage_charge else 0

        # Discharge if demand is higher than renewable supply or if export price is high
        discharge_power = (
            min(battery_max_discharge, demand - available_renewable) 
            if demand > available_renewable else 0
        )
        if prioritize_export and df_version.loc[t-1, "SOC (%)"] > 20:
            discharge_power = min(battery_max_discharge, df_version.loc[t-1, "SOC (%)"] / 100 * battery_capacity)

        # Grid Import only if necessary and price isn't too high
        grid_import = max(0, demand - available_renewable - discharge_power) if not reduce_import else 0

        # Grid Export if excess renewable after charging
        grid_export = max(0, available_renewable - demand - charge_power)
        if prioritize_export:
            grid_export = max(grid_export, discharge_power)  

        # Apply mutual exclusivity rules
        if grid_import > 0:
            grid_export = 0
        if charge_power > 0:
            discharge_power = 0

        # Enforce Energy Balance (Final Correction Step)
        energy_balance = available_renewable + grid_import + discharge_power - grid_export - charge_power
        correction_needed = demand - energy_balance

        if abs(correction_needed) > 1e-2:
            # Adjust variables to ensure balance
            if correction_needed > 0:  # Not enough supply, increase import or discharge
                if discharge_power < battery_max_discharge and df_version.loc[t-1, "SOC (%)"] > soc_min:
                    discharge_power += correction_needed
                else:
                    grid_import += correction_needed
            else:  # Excess supply, increase export or charge
                if charge_power < battery_max_charge:
                    charge_power -= correction_needed
                else:
                    grid_export -= correction_needed

        # Update SoC in percentage format
        prev_soc = df_version.loc[t-1, "SOC (%)"]
        soc_change = ((charge_power * eta_ch * time_step) - (discharge_power * time_step / eta_dis)) / battery_capacity * 100
        new_soc = prev_soc + soc_change
        new_soc = np.clip(new_soc, soc_min, soc_max)  # Ensure within 5% - 100%

        # Store computed values
        df_version.loc[t, "P_import (kW)"] = grid_import
        df_version.loc[t, "P_export (kW)"] = grid_export
        df_version.loc[t, "P_bat_ch (kW)"] = charge_power
        df_version.loc[t, "P_bat_dis (kW)"] = discharge_power
        df_version.loc[t, "SOC (%)"] = new_soc  # Store SoC as percentage

# End time measurement
end_time = time.time()
execution_time = end_time - start_time

# Save the generated decision variable data
df_fixed.to_csv("generated_decision_variables_FIXED_v4.csv", index=False)
df_dynamic.to_csv("generated_decision_variables_DYNAMIC_v4.csv", index=False)

# Save only the required solution columns for both Fixed & Dynamic with "v4" naming
solution_columns = ["time", "P_import (kW)", "P_export (kW)", "P_bat_ch (kW)", "P_bat_dis (kW)", "SOC (%)"]

df_fixed[solution_columns].to_csv("solution_output_FIXED_v4.csv", index=False)
df_dynamic[solution_columns].to_csv("solution_output_DYNAMIC_v4.csv", index=False)

print("✅ Solutions saved separately for both pricing strategies with correct column names.")
print(f"⏳ Total Execution Time: {execution_time:.2f} seconds")
