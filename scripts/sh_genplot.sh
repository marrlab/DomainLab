mkdir $2
merge_csvs.sh
python main_out.py --gen_plots merged_data.csv  --outp_dir $2
