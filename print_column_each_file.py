import pandas as pd



# File paths
files = ["WorkingCodeVersion1_FIXED_v4.csv", "WorkingCodeVersion1_DYNAMIC_v4.csv", "WorkingCodeVersion1_HOMER_v4.csv"]

# Check row counts
for file in files:
    df = pd.read_csv(file)
    print(f"📂 {file}: {df.shape[0]} rows")







for file in files:
    df = pd.read_csv(file, nrows=1)  # Read only the first row to check headers
    print(f"\n📂 {file} Columns:")
    print(df.columns.tolist())
