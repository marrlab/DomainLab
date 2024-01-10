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
