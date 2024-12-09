# mkdir $2
sh scripts/merge_csvs.sh $1
python main_out.py --gen_plots merged_data.csv  --outp_dir partial_agg_plots