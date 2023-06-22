#!/bin/bash -x -v
set -e  # exit upon first error
starttime=`date +%s`

# run examples
bash -x -v ci_run_examples.sh

# run test
sh ci_pytest_cov.sh

# run benchmark
./run_benchmark_local_conf_seed2_gpus.sh examples/benchmark/demo_benchmark.yaml

# update documentation
# if git status | grep -q 'master'; then
  # echo "in master branch"
sh gen_doc.sh
# fi

endtime=`date +%s`
runtime=$((endtime-starttime))
echo "total time used:"
echo "$runtime"


