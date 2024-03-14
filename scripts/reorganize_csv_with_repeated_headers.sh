#!/bin/bash

# Define the input file
input_file=$1

# Define the output CSV file
output_file="merged_data.csv"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Input file not found: $input_file"
    exit 1
fi

# Initialize the output CSV file with the header from the first section
header=$(awk '/param_index/{print; exit}' "$input_file")
echo "$header" > "$output_file"

# Merge data from all sections into the output CSV file
awk '/param_index/{next} NF' "$input_file" | sed '1d' >> "$output_file"

echo "Merged data saved to $output_file"
