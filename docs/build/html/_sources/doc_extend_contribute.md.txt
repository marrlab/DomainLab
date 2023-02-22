# Extend or contribute with a custom domain generalization algorithm

## External extension by implementing your custom models

Look at this example

```
cd ..
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=3 --apath=examples/algos/demo_custom_algo_builder.py --aname=custom --nname_argna2val my_custom_arg_name --nname_argna2val alexnet

python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=3 --apath=examples/algos/demo_custom_algo_builder.py --aname=custom --npath_argna2val my_custom_arg_name --npath_argna2val examples/nets/resnet.py
```

where the example python file corresponding to `--apath` defines a model which contains several modules that can be customized to different neural networks via `--npath_argna2val` if the neural network is specified in a python file
or `nname_argna2val` if the neural network is already implemented.

## Internal extension for more advanced contribution
- implement `domainlab/algos/builder_your-algorithm-name.py`
- add your algorithm into `domainlab/algos/zoo_algos.py` by adding `chain = NodeAlgoBuilder[your-algorithm-name](chain)`
- note that all algorithms will be converted to lower case!
- make a pull request


### Design
![Design Diagram](libDG.svg)
