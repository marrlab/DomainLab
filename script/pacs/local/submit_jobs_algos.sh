#!/bin/bash
repodir="../.."
echo "source code directory is $repodir"
echo "seed range: $1:$2"
startseed=$1
endseed=$2
mkdir -p logs
currenttime="`date`"

scriptname='sbatch_pacs_dir_algo_start_end_seed_epos.sh'
epos=1

batchsize="2"
paths="--tpath=examples/tasks/demo_task_path_list_small.py --npath=examples/nets/resnet.py"
algoconf="sh_algo_conf.sh"

#

echo "current time: $currenttime"

# each job correspond to one algorithm
for algo in "deepall" "diva" "hduva" "matchdg"
do
    echo "algo: $algo"
    JOBNAME="$algo"
    sbatch --error="logs/${JOBNAME}_error_${currenttime}" --output="logs/${JOBNAME}_output_${currenttime}" --job-name="${JOBNAME}" $scriptname $repodir $algo $startseed $endseed $epos $algoconf "$paths" "$batchsize"
done
