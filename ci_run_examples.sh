#!/bin/bash -x -v
set -e  # exit upon first error
# >> append content
# > erase original content
sed -n '/```shell/,/```/ p' docs/doc_examples.md | sed '/^```/ d' > ./sh_temp_example.sh
bash -x -v -e sh_temp_example.sh

sed -n '/```shell/,/```/ p' docs/doc_MNIST_classification.md | sed '/^```/ d' > ./sh_temp_mnist.sh
bash -x -v -e sh_temp_mnist.sh
echo "mnist example done"

sed -n '/```shell/,/```/ p' docs/doc_benchmark.md | sed '/^```/ d' > ./sh_temp_benchmark.sh
bash -x -v -e sh_temp_benchmark.sh
echo "benchmark  done"

sed -n '/```shell/,/```/ p' docs/doc_custom_nn.md | sed '/^```/ d' > ./sh_temp_nn.sh
bash -x -v -e sh_temp_nn.sh

sed -n '/```shell/,/```/ p' docs/doc_tasks.md | sed '/^```/ d' > ./sh_temp_task.sh
bash -x -v -e sh_temp_task.sh

sed -n '/```shell/,/```/ p' README.md | sed '/^```/ d' > ./sh_temp_readme.sh
bash -x -v -e sh_temp_readme.sh

# sed -n '/```shell/,/```/ p' docs/doc_extend_contribute.md | sed '/^```/ d' > ./sh_temp_extend.sh
#bash -x -v -e sh_temp_extend.sh
