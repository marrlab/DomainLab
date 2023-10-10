#!/bin/bash
export CUDA_VISIBLE_DEVICES=""   
# although garbage collector has been explicitly called, sometimes there is still CUDA out of memory error
# so it is better not to use GPU to do the pytest to ensure every time there is no CUDA out of memory error occuring
# pytest -s tests/test_fbopt.py
python main_out.py --te_d=caltech --task=mini_vlcs --bs=8 --aname=diva --trainer=fbopt --nname=alexnet --nname_dom=alexnet --gamma_d=3  --gamma_y=3 --epos=200 --es=100
