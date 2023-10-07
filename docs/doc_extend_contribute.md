# Extend or contribute with a custom domain generalization algorithm

## External extension by implementing your custom models

Look at this example

```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --apath=examples/models/demo_custom_model.py --aname=custom --nname_argna2val my_custom_arg_name --nname_argna2val alexnet

python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --apath=examples/models/demo_custom_model.py --aname=custom --npath_argna2val my_custom_arg_name --npath_argna2val examples/nets/resnet.py
```

where the example python file corresponding to `--apath` defines a model which contains several modules that can be customized to different neural networks via `--npath_argna2val` if the neural network is specified in a python file
or `nname_argna2val` if the neural network is already implemented.


### Design
![Design Diagram](libDG.svg)
