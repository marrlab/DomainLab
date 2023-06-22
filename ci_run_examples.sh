#!/bin/bash -x -v
set -e  # exit upon first error
sed 's/`//g' docs/doc_examples.md >> ./sh_temp.sh
sed 's/`//g' docs/doc_MNIST_classification.md >> ./sh_temp.sh
sed 's/`//g' docs/doc_benchmark.md >> ./sh_temp.sh
sed 's/`//g' docs/doc_custom_nn.md >> ./sh_temp.sh
sed 's/`//g' docs/doc_extend_contribute.md >> ./sh_temp.sh
sed 's/`//g' docs/doc_tasks.md >> ./sh_temp.sh
sed 's/`//g' README.md >> ./sh_temp.sh
bash -x -v -e sh_temp.sh
