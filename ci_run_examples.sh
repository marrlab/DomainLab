#!/bin/bash -x -v
set -e  # exit upon first error
sed 's/`//g' docs/doc_examples.md > ./sh_temp.sh
bash -x -v -e sh_temp.sh
