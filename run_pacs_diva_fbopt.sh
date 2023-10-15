#!/bin/bash
# export CUDA_VISIBLE_DEVICES=""   
# although garbage collector has been explicitly called, sometimes there is still CUDA out of memory error
# so it is better not to use GPU to do the pytest to ensure every time there is no CUDA out of memory error occuring
# pytest -s tests/test_fbopt.py
python main_out.py --te_d=sketch --bs=64 --aname=diva --trainer=fbopt --epos=200 --es=200 --npath_dom=examples/nets/resnet50domainbed.py --tpath=examples/tasks/task_pacs_path_list.py --npath_dom=examples/nets/resnet50domainbed.py
