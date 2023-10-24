JOB_NAME="fbopt_jigen"
PATH_OUT_BASE="/lustre/groups/imm01/workspace/felix.drost/fbopt_jigen/job_logs"
PATH_CODE="/lustre/groups/imm01/workspace/felix.drost/fbopt_jigen"


job_file="${PATH_OUT_BASE}/${JOB_NAME}.cmd"
echo "#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${PATH_OUT_BASE}/${JOB_NAME}.out
#SBATCH -e ${PATH_OUT_BASE}/${JOB_NAME}.err
#SBATCH -p cpu_p 
#SBATCH -t 2-00:00:00
#SBATCH -c 20
#SBATCH --mem=32G
#SBATCH --qos=cpu_normal
#SBATCH --no-requeue
#SBATCH --nice=10000

source ~/.bash_profile
conda activate fbopt

${PATH_CODE}/run_benchmark_slurm.sh ${PATH_CODE}/examples/benchmark/pacs_jigen_fbopt_and_others.yaml 0
" > ${job_file}
sbatch ${job_file}



