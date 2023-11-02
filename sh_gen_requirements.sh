#!/bin/bash
poetry update
poetry export -f requirements.txt --output requirements.txt --without-hashes --without-urls
