#!/bin/bash -x -v
set -e  # exit upon first error
sed -n '/```shell/,/```/ p' docs/doc_examples.md | sed '/^```/ d' >> ./sh_temp.sh
sed -n '/```shell/,/```/ p' docs/doc_MNIST_classification.md | sed '/^```/ d' >> ./sh_temp.sh
sed -n '/```shell/,/```/ p' docs/doc_benchmark.md | sed '/^```/ d' >> ./sh_temp.sh
sed -n '/```shell/,/```/ p' docs/doc_custom_nn.md | sed '/^```/ d' >> ./sh_temp.sh
sed -n '/```shell/,/```/ p' docs/doc_extend_contribute.md | sed '/^```/ d' >> ./sh_temp.sh
sed -n '/```shell/,/```/ p' docs/doc_tasks.md | sed '/^```/ d' >> ./sh_temp.sh
sed -n '/```shell/,/```/ p' README.md | sed '/^```/ d' >> ./sh_temp.sh
bash -x -v -e sh_temp.sh
