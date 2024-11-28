cat $1/rule_results/*.csv > temp_result.csv
sh reorganize_csv_with_repeated_headers.sh temp_result_header_cleaned.csv
mkdir -p temp_agg
python main_out.py --gen_plots merged_data.csv --outp_dir ./temp_agg
