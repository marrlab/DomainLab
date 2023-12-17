#!/bin/bash
# export CUDA_VISIBLE_DEVICES=""   
# although garbage collector has been explicitly called, sometimes there is still CUDA out of memory error
# so it is better not to use GPU to do the pytest to ensure every time there is no CUDA out of memory error occuring
# pytest -s tests/test_fbopt.py
python main_out.py --te_d=sketch --tpath=examples/tasks/demo_task_path_list_small.py --bs=16 --model=jigen --trainer=fbopt --nname=alexnet --epos=200 --es=100 --init_mu=0.01
