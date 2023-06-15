#!/bin/bash -x -v
# get the token via https://pypi.org/manage/project/domainlab/settings/
poetry config pypi-token.pypi [my-token]
poetry publish
