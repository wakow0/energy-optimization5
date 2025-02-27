import pandas as pd
import os
import csv

def detect_delimiter(file_path):
    """Auto-detect CSV delimiter."""
    with open(file_path, "r", encoding="utf-8") as f:
        sample = f.readline()
    delimiters = [",", ";", "\t"]
    for delim in delimiters:
        if delim in sample:
            return delim
    return ","  # Default to comma

def load_csv_data(file_path: str, sample_percentage: float = 100, output_file="processed_data.csv"):
    """
    Loads and preprocesses the dataset from a CSV file, ensuring correct formatting.

    Parameters:
    - file_path (str): Path to the CSV file.
    - sample_percentage (float): Percentage of data to process (1-100).
    - output_file (str): Output filename for the processed dataset.

    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    try:
        # ✅ Detect and handle `sep=,` issue
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines or len(lines) < 3:
            raise ValueError("❌ ERROR: CSV file is empty or has too few rows!")

        if lines[0].startswith("sep="):
            lines = lines[1:]  # Remove the first line

        # ✅ Detect delimiter automatically
        delimiter = detect_delimiter(file_path)

        # ✅ Extract column names from the first row
        column_names = lines[0].strip().replace('"', '').split(delimiter)

        # ✅ Write the cleaned content to a temporary file
        temp_file_path = "cleaned_" + file_path
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.writelines(lines[2:])  # Skip unit row

        # ✅ Manually Read CSV and Skip Bad Rows
        valid_rows = []
        with open(temp_file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if len(row) == len(column_names):  # Only keep rows with correct column count
                    valid_rows.append(row)

        # ✅ Convert List to DataFrame
        df = pd.DataFrame(valid_rows, columns=column_names)

        # ✅ Ensure dataframe is valid before proceeding
        if df.empty or len(df.columns) < 2:
            raise ValueError("❌ ERROR: Dataframe is empty after filtering! Check the CSV file format.")

        # ✅ Standardize column names (lowercase, underscores, remove `:`)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(":", "")

        # ✅ Ensure "time" column exists
        if "time" not in df.columns:
            raise ValueError("❌ ERROR: 'time' column is missing! Check the CSV structure.")

        # ✅ Convert Time Column
        df["time"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

        # ✅ Remove NaN values in 'time' column
        df.dropna(subset=["time"], inplace=True)

        # ✅ Apply sampling if needed
        if 0 < sample_percentage < 100:
            df = df.sample(frac=sample_percentage / 100, random_state=42)

        # ✅ Sort by time and reset index
        df = df.sort_values(by="time").reset_index(drop=True)

        # ✅ Save the dataset only if valid
        if df.empty:
            raise ValueError("❌ ERROR: Data processing resulted in an empty dataset!")

        df.to_csv(output_file, index=False, encoding="utf-8")

        return df  # ✅ Return the processed DataFrame

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

# ✅ Run parsing when script is executed directly
if __name__ == "__main__":
    file_path = "research.csv"  # Replace with any CSV file
    
    # ✅ Call function with column enforcement enabled (default)
    df_cleaned = load_csv_data(file_path, sample_percentage=100)

    # ✅ Ensure the dataset is actually saved
    if df_cleaned is not None:
        print(f"✅ Processed data successfully saved as 'processed_data.csv'!")
