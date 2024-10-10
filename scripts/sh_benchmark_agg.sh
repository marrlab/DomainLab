cat $1/rule_results/*.csv > temp_results.csv
mkdir -p temp_agg
python main_out.py --gen_plots temp_results.csv --outp_dir ./temp_agg
