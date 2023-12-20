# Custom Neural Network

To use a custom neural network, the user has to implement the following signature in a python file and feed into this library via `--npath`. 


```python
def build_feat_extract_net(dim_y, remove_last_layer):
```

See examples below from `--npath=examples/nets/resnet.py` where the examples can be found in the examples folder of the code repository.

## Custom Neural Network Examples

<https://github.com/marrlab/DomainLab/blob/master/examples/nets/resnet.py>

### model 'erm' with custom neural network

```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm --npath=examples/nets/resnet.py
```

### trainer 'matchdg' with custom neural network

```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm --trainer=matchdg --epochs_ctr=3 --epos=6 --npath=examples/nets/resnet.py
```
