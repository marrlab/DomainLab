#!/bin/bash
poetry update
poetry export -f requirements.txt --output requirements.txt --without-hashes --without-urls --without snakemake diatrie

# only snamkemake poetry export --without-hashes -f requirements.txt | grep -v colorama > requirements.txt
