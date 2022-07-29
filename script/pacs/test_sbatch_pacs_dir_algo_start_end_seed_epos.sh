set -e  # exit upon first error
batchsize="2"
paths="--tpath=examples/tasks/demo_task_path_list_small.py --npath=examples/nets/resnet.py"
algoconf="sh_algo_conf.sh"
epos=2

bash sbatch_pacs_dir_algo_start_end_seed_epos.sh ../.. matchdg 0 1 "$epos" $algoconf "$paths" "$batchsize"
bash sbatch_pacs_dir_algo_start_end_seed_epos.sh ../.. hduva 0 1 "$epos" $algoconf "$paths" "$batchsize"
bash sbatch_pacs_dir_algo_start_end_seed_epos.sh ../.. deepall 0 1 "$epos" $algoconf "$paths" "$batchsize"
bash sbatch_pacs_dir_algo_start_end_seed_epos.sh ../.. diva 0 1 "$epos" $algoconf "$paths" "$batchsize"
