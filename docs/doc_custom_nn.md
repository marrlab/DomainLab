# Custom Neural Network

To use a custom neural network, the user has to implement the following signature in a python file and feed into this library via `--npath`. 


```
def build_feat_extract_net(dim_y, remove_last_layer):
```

See examples below from `--npath=examples/nets/resnet.py` where the examples can be found in the examples folder of the code repository.

## Custom Neural Network Examples

<https://github.com/marrlab/DomainLab/blob/master/examples/nets/resnet.py>

### algorithm 'deepall' with custom neural network

```
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --aname=deepall --npath=examples/nets/resnet.py
```

### algorithm 'matchdg' with custom neural network

```
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --aname=matchdg --epochs_ctr=3 --epos=6 --npath=examples/nets/resnet.py
```
