#!/bin/bash
# export CUDA_VISIBLE_DEVICES=""   
# although garbage collector has been explicitly called, sometimes there is still CUDA out of memory error
# so it is better not to use GPU to do the pytest to ensure every time there is no CUDA out of memory error occuring
# pytest -s tests/test_fbopt.py
python main_out.py --te_d=sketch --tpath=examples/tasks/task_pacs_path_list.py --model=jigen --trainer=fbopt --bs=64 --epos=200 --es=200 --npath=examples/nets/resnet50domainbed.py --mu_init=1e-6 --lr=5e-5 --coeff_ma_output_state=0.1
