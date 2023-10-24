#!/bin/bash
export CUDA_VISIBLE_DEVICES=""   
# although garbage collector has been explicitly called, sometimes there is still CUDA out of memory error
# so it is better not to use GPU to do the pytest to ensure every time there is no CUDA out of memory error occuring
# --cov-report term-missing to show in console file wise coverage and lines missing
python -m pytest --cov=domainlab --cov-report html
