rm -f -R .snakemake
DIR=$(pwd)
echo $DIR
bash ./sbatch4submit_slurm_cpu_3days.sh $DIR $1 $2
