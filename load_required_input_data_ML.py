import pandas as pd

# Define file path (Update if needed)
file_path = "processed_data.csv"  

# Load the dataset
try:
    df = pd.read_csv(file_path)
    print("✅ Successfully loaded 'processed_data.csv'!")
except FileNotFoundError:
    print("❌ ERROR: 'processed_data.csv' not found. Please check the file location.")

# Extract necessary columns
required_columns = [
    "time",
    "dc_ground_1500vdc_power_output",  # Solar Generation
    "windflow_33_[500kw]_power_output",  # Wind Generation
    "ac_primary_load",  # Load Demand
    "total_consumption_rate",  # Import Price (Buy Price)
    "grid_sellback_rate"  # Export Price (Sell Price)
]

# Check if all required columns exist
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"❌ ERROR: Missing required columns: {missing_columns}")
else:
    # Extract the required columns
    df_required = df[required_columns]
    
    # Display the first few rows for verification
    print("\n✅ Extracted Data Preview:")
    print(df_required.head())

    # Save extracted data to a new CSV for reference
    df_required.to_csv("extracted_inputs.csv", index=False)
    print("\n✅ Extracted data saved as 'extracted_inputs.csv' for verification.")
