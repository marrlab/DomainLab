cat $1/rule_results/*.csv > temp_results.csv
sh reorganize_csv_with_repeated_headers.sh merged_data.csv
mkdir -p temp_agg
python main_out.py --gen_plots merged_data.csv --outp_dir ./temp_agg
