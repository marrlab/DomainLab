#!/bin/bash
# $1 should be a folder with only csv files
# $2 is the output file name

# Define the directory containing the text files
directory=$1

# Define the output CSV file
output_file="${2:-merged_data.csv}"

# Initialize the merged CSV file with the header from the first file
find "$directory" -maxdepth 1 -type f -name "*.csv" | sort | head -1 | xargs head -n 1 > "$output_file"

# Merge data from all text files into the output CSV file
find "$directory" -maxdepth 1 -type f -name "*.csv" | sort | while read -r file; do
    # Skip the header line, merge the remaining lines into the output CSV file
    tail -n +2 "$file" >> "$output_file"
done

echo "Merged data saved to $output_file"
