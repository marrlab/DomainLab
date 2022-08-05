#!/bin/bash -x -v
set -e  # exit upon first error
cat docs/doc_examples.md > ./sh_temp.sh
bash -x -v sh_temp.sh
