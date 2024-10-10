# change the following two lines if needed
VENV="domainlab_py39"
BASHRC="~/.bashrc"  # source ~/.bash_profile

##
JOB_NAME="submit"
PATH_CODE=$1
PATH_OUT_BASE="${PATH_CODE}/submit_job_logs"
mkdir -p $PATH_OUT_BASE
PATH_YAML=$2
START_SEED=$3
ACTIVE_TIME="3-00:00:00"

job_file="${PATH_OUT_BASE}/${JOB_NAME}.cmd"


# echo the following line to ${job_file}
echo "#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${PATH_OUT_BASE}/${JOB_NAME}.out
#SBATCH -e ${PATH_OUT_BASE}/${JOB_NAME}.err
#SBATCH -p cpu_p
#SBATCH -t ${ACTIVE_TIME}
#SBATCH -c 20
#SBATCH --mem=32G
#SBATCH --qos=cpu_normal
#SBATCH --no-requeue
#SBATCH --nice=10000

source ${BASHRC}
conda activate ${VENV}

${PATH_CODE}/run_benchmark_slurm.sh ${PATH_CODE}/${PATH_YAML} ${START_SEED}
" > ${job_file}
# end of echo

sbatch ${job_file}
