repodir="../.."
echo "source code directory is $repodir"
echo "seed range: $1:$2"
startseed=$1
endseed=$2
mkdir -p logs
currenttime="`date`"

scriptname='sbatch_pacs_dir_algo_start_end_seed_epos.sh'
epos=1000  # Let hduva and diva run enough epochs

batchsize="20"  # FIXME
paths="--tpath=examples/tasks/not_a_demo_task_path_list.py --npath=examples/nets/resnet.py"   # FIXME
algoconf="sh_algo_conf_cluster.sh"   # FIXME

#

echo "current time: $currenttime"

# each job correspond to one algorithm
for algo in "hduva" "diva" "matchdg" "deepall"
do
    echo "algo: $algo"
    JOBNAME="$algo"
    sbatch --error="logs/${JOBNAME}_error_${currenttime}" --output="logs/${JOBNAME}_output_${currenttime}" --job-name="${JOBNAME}" $scriptname $repodir $algo $startseed $endseed $epos $algoconf "$paths" "$batchsize"
done
