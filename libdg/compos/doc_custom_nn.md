To use a custom neural network, the user has to implement the following signature in a python file and feed into this library via `--npath`. 

```
def build_feat_extract_net(dim_y, remove_last_layer):
```

See examples below from `--npath=examples/nets/resnet.py`

# Task 'mini_vlcs'

## algorithm 'diva' with custom neural network

```
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=diva --npath=examples/nets/resnet.py
```

## algorithm 'deepall' with custom neural network

```
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --aname=deepall --npath=examples/nets/resnet.py
```

## algorithm 'matchdg' with custom neural network

```
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=20 --aname=matchdg --epochs_ctr=3 --epochs_erm=3 --npath=examples/nets/resnet.py
```

# ImagePath Task
## ImagePath Task with externally user defined neural network

```
python main_out.py --te_d=sketch --tpath=examples/tasks/demo_task_path_list_small.py --debug --bs=2 --npath=examples/nets/resnet.py
```
# FolderTask
## Folder Task with externally user defined neural network

```
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --debug --bs=2 --npath=examples/nets/resnet.py
```
