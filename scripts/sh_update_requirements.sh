#!/bin/bash -x -v
poetry update
poetry export -f requirements.txt --output requirements.txt --without-hashes
