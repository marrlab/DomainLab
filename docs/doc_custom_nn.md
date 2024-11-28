# Specify neural network in command line

To use a custom neural network in command line with DomainLab, the user has to implement the following signature in a python file and specify the file path via `--npath`


```python
def build_feat_extract_net(dim_y, remove_last_layer=False):
```
The user could choose to ignore argument `remove_last_layer` since this argument is only used in fair benchmark comparison.

See examples below from `--npath=examples/nets/resnet.py` where the examples can be found in the examples folder of the code repository.
<https://github.com/marrlab/DomainLab/blob/master/examples/nets/resnet.py>

## Example use case
### model 'erm' with custom neural network

```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm --npath=examples/nets/resnet.py
```

### trainer 'matchdg' with custom neural network

```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm --trainer=matchdg --epochs_ctr=3 --epos=6 --npath=examples/nets/resnet.py
```


### model erm with custom neural network
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --model=erm --npath=examples/nets/resnet.py
```

## Larger images:

### model erm with implemented neural network
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --model=erm --nname=alexnet
```

### model dann with implemented neural network
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --model=dann --nname=alexnet
```

## Custom algorithm defined in external python file
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=3 --apath=examples/models/demo_custom_model.py --model=custom --nname_argna2val my_custom_arg_name --nname_argna2val alexnet
```

```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=3 --apath=examples/models/demo_custom_model.py --model=custom --npath_argna2val my_custom_arg_name --npath_argna2val examples/nets/resnet.py
```
