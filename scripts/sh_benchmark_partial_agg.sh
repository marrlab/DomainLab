# $1 should be the rule_results folder which contains several csv files
# $2 default to be merged_data.csv

file_na_merged_csv="${2:-merged_data.csv}"
sh scripts/merge_csvs.sh $1 $file_na_merged_csv
python scripts/generate_latex_table.py $file_na_merged_csv
python main_out.py --gen_plots $file_na_merged_csv  --outp_dir partial_agg_plots
