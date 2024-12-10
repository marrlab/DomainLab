"""
merge csv files inside a directory into one csv file, optinally, select
column value to be fitting a regular expression, e.g. params={key1:val1,
key2:val2}, let pattern=val2
"""
import os
import sys
import argparse
import glob
import re
import pandas as pd


# Set up argument parsing
parser = argparse.ArgumentParser(
    description="merge all CSV files in a specified directory.")

parser.add_argument('directory', type=str,
                    help="The directory to search for CSV files")

parser.add_argument('--pattern', type=str,
                    help="The pattern to match in the specified column")

parser.add_argument('--out_file_name', type=str,
                    default="merged.csv",
                    help="out csv name")

parser.add_argument('--column_name', type=str,
                    default=None,
                    help="The column name to filter by")
parser.add_argument('--filter_value', type=str,
                    default=None,
                    help="The value to match in the specified column")

# Parse the arguments
args = parser.parse_args()

# Use the directory passed from the command line
directory = args.directory
column_name = args.column_name
filter_value = args.filter_value

# List of CSV files to be merged
csv_files = glob.glob(os.path.join(directory, "*.csv"))

if not column_name:
    merged_df = pd.concat([pd.read_csv(file) for file in csv_files])

else:
    dfs = []
    if args.pattern:
        pattern_re = re.compile(args.pattern)
    # Loop through each file and filter the rows based on the column value
    for file in csv_files:
        df = pd.read_csv(file, skipinitialspace=True)
        # Check if the column exists in the current file
        if column_name in df.columns:
            # Filter rows where the column value matches the filter value
            if filter_value:
                filtered_df = df[df[column_name] == filter_value]
            else:
                filtered_df = \
                    df[df[column_name].astype(str).apply(
                        lambda x: bool(pattern_re.search(x)))]
            # Append the filtered dataframe to the list if it's not empty
            if not filtered_df.empty:
                dfs.append(filtered_df)
        else:
            raise RuntimeError(f"column name {column_name} not found in {file}")

    # Combine all files in the list
    if dfs:
        merged_df = pd.concat(dfs)
    else:
        print("Merg CSV failed")
        sys.exit()


# Save the merged file
merged_df.to_csv(args.out_file_name, index=False)

print(f"Merged CSV saved as {args.out_file_name}")
