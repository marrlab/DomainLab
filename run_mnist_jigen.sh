#!/bin/bash
# export CUDA_VISIBLE_DEVICES=""   
# although garbage collector has been explicitly called, sometimes there is still CUDA out of memory error
# so it is better not to use GPU to do the pytest to ensure every time there is no CUDA out of memory error occuring
# pytest -s tests/test_fbopt.py
python main_out.py --te_d=caltech --task=mini_vlcs --bs=16 --aname=jigen --trainer=fbopt --nname=alexnet --epos=200 --es=200 --mu_init=1.0 --coeff_ma_output=0 --coeff_ma_setpoint=0 --coeff_ma_output=0 