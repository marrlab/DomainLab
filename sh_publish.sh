#!/bin/bash -x -v
poetry config pypi-token.pypi [my-token]
poetry publish
