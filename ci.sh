#!/bin/bash -x -v
set -e  # exit upon first error
starttime=`date +%s`
sh ci_run_examples.sh
sh ci_pytest_cov.sh
sh gen_doc.sh
endtime=`date +%s`
runtime=$((endtime-starttime))
echo "total time used:"
echo $runtime


