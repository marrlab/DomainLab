#!/bin/bash -x -v
# step 1: in case for a new download of the repository, get the token via https://pypi.org/manage/project/domainlab/settings/,  then do
# poetry config pypi-token.pypi [my-token] 
# step 2: change the version in pyproject.toml
poetry build  # step 3
poetry publish # step 4
