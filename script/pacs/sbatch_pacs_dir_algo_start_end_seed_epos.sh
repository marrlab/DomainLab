#!/bin/bash
#SBATCH -p gpu_p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH --nice=10000 
set -e
echo "Starting stuff at `date`"
# source $HOME/.bashrc    # FIXME
nvidia-smi


###################Start of Parse Args##################################################################
echo "command line arguments:"
echo "directory of repository for source code: $1"

ALGO="$2"
echo "algo: $ALGO"

STARTSEED="$3"
ENDSEED="$4"
echo "seed from $STARTSEED to $ENDSEED"

EPOS=$5   # FIXME: must be Capital!
echo "number of epochs: $EPOS"

#EXTRAARGS="$5"
#echo "Extra args $EXTRAARGS" # You can put arbitrary unix commands here, call other scripts, etc...
algoconffile=$6
source $algoconffile    # bash script file that contains algorithm specific configurations

tpathnpath=$7
batchsize=$8


###################End of Parse Args####################################################################


# change to source code directory
echo "going to enter directory: $1"
cd "$1"
echo "current directory `pwd`"


for seed in `seq $STARTSEED $ENDSEED`   # outter loop: seed is less important than domain
do
echo "outter loop seed: $seed"
for testdomain in "sketch" "cartoon" "photo" "art_painting"   # inner loop
do
  echo "test domain:$testdomain"
  echo "algorithm specific arguments: ${dict_algo_args[$ALGO]}"
  echo "batchsize: $batchsize"
  python main_out.py --te_d="$testdomain"  --bs=$batchsize --aname="$ALGO" $tpathnpath --epos="$EPOS" --seed="$seed" ${dict_algo_args[$ALGO]}
done

# nvidia-smi
done

echo "Ending stuff at `date`"
