### Basic usage
DomainLab comes with some minimal toy-dataset to test its basis functionality, see [A minimal subsample of the VLCS dataset](../data/vlcs_mini) and [an example configuration file for vlcs_mini](../examples/tasks/task_vlcs.py)

Suppose you have cloned the repository and have the dependencies ready, change directory to this repository:

To train a domain invariant model on the vlcs_mini task

```shell
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --config=examples/yaml/demo_config_single_run_diva.yaml
```
where `--tpath` specifies the path of a user specified python file which defines the domain generalization task [see here](../examples/tasks/task_vlcs.py), `--te_d` specifies the test domain name (or index starting from 0), `--config` specifies the configurations of the domain generalization algorithms, [see here](../examples/yaml/demo_config_single_run_diva.yaml)

#### Further usage
Alternatively, in a verbose mode without using the algorithm configuration file:

```shell
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --debug --bs=2 --model=diva --gamma_y=7e5 --gamma_d=1e5 --nname=alexnet --nname_dom=conv_bn_pool_2
```

where `--model` specifies which algorithm to use, `--bs` specifies the batch size, `--debug` restrain only running for 2 epochs and save results with prefix 'debug'. For DIVA, the hyper-parameters include `--gamma_y=7e5` which is the relative weight of ERM loss compared to ELBO loss, and `--gamma_d=1e5`, which is the relative weight of domain classification loss compared to ELBO loss.
`--nname` is to specify which neural network to use for feature extraction for classification, `--nname_dom` is to specify which neural network to use for feature extraction of domains.
For usage of other arguments, check with

```shell
python main_out.py --help
```

See also [Examples](./doc_examples.md).

### Custom Neural network

where the user could plugin different neural network architectures for the same algorithm. See
[Specify Custom Neural Networks for an algorithm](./doc_custom_nn.md)

### Output structure (results storage) and Performance Measure
[Output structure and Performance Measure](./doc_output.md)


## Software Design Pattern, Extend or Contribution
[Extend or Contibute](./doc_extend_contribute.md)
