# colored version of mnist where each color represent one domain, in total 10 colors
python main_out.py --te_d=0 --task=mnistcolor10 --keep_model
python main_out.py --te_d 0 1 2 --tr_d 3 4 5 6 7 8 9 --task=mnistcolor10 --keep_model

# Default tasks (mini_vlcs) (224*224 sized image)
## model diva with default neural network
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=diva
## model diva with custom neural network
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=diva --npath=examples/nets/resnet.py

## model deepall with default neural network
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --aname=deepall
## model deepall with custom neural network
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --aname=deepall --npath=examples/nets/resnet.py
## model matchdg with default neural network
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=20 --aname=matchdg --epochs_ctr=3 --epochs_erm=3   
## model matchdg with custom neural network
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=20 --aname=matchdg --epochs_ctr=3 --epochs_erm=3 --npath=examples/nets/resnet.py

###### matchdg learning rate is crutial for not having NAN

# User defined tasks

## Folder Task
### Folder Task with default neural network
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --debug --bs=2
### Folder Task with externally user defined neural network
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --debug --bs=2 --npath=examples/nets/resnet.py


## ImagePath Task
### ImagePath Task with default algorithm
python main_out.py --te_d=sketch --tpath=examples/tasks/demo_task_path_list_small.py --debug --bs=2
### ImagePath Task with externally user defined neural network
python main_out.py --te_d=sketch --tpath=examples/tasks/demo_task_path_list_small.py --debug --bs=2 --npath=examples/nets/resnet.py


# HDUVA uses lots of memory can only be tested on large mem
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=hduva --nocu

# Custom algorithm defined in external python file
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --apath=examples/algos/demo_custom_algo_builder.py --aname=custom

