#!/bin/bash
# export CUDA_VISIBLE_DEVICES=""   
# although garbage collector has been explicitly called, sometimes there is still CUDA out of memory error
# so it is better not to use GPU to do the pytest to ensure every time there is no CUDA out of memory error occuring
# pytest -s tests/test_fbopt.py

python main_out.py --te_d=0 --tr_d 1 2 --task=mnistcolor10 --bs=16 --model=diva --trainer=fbopt --nname=conv_bn_pool_2 --epos=5000 --es=5 --mu_init=1e-6 --gamma_y=1.0 --k_i_gain_ratio=0.9 --coeff_ma_output_state=0 --coeff_ma_setpoint=0 --epos_min=1000 --force_setpoint_change_once
