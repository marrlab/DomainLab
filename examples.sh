# Hard coded tasks
python main_out.py --te_d=0 --task=mnistcolor10 --keep_model
python main_out.py --te_d 0 1 2 --tr_d 3 4 5 6 7 8 9 --task=mnistcolor10 --keep_model

# 224 size image testing
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=diva
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=diva --npath=examples/nets/resnet.py
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --aname=deepall
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --aname=deepall --npath=examples/nets/resnet.py
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=20 --aname=matchdg --epochs_ctr=3 --epochs_erm=3   
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=20 --aname=matchdg --epochs_ctr=3 --epochs_erm=3 --npath=examples/nets/resnet.py
# learning rate is crutial for having NAN

# User defined tasks

# Folder Task
python main_out.py --te_d=caltech --tpath=examples/task_vlcs.py --debug --bs=2

# ImagePath Task
python main_out.py --te_d=sketch --tpath=examples/demo_task_path_list.py --debug --bs=2


# HDUVA uses lots of memory can only be tested on large mem
#python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=hduva --nocu
