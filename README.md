# DomainLab: train robust neural networks using domain generalization algorithms on your data

![GH Actions CI ](https://github.com/marrlab/DomainLab/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/marrlab/DomainLab/branch/master/graph/badge.svg)](https://app.codecov.io/gh/marrlab/DomainLab)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/bc22a1f9afb742efb02b87284e04dc86)](https://www.codacy.com/gh/marrlab/DomainLab/dashboard)
[![Documentation](https://img.shields.io/badge/Documentation-Here)](https://marrlab.github.io/DomainLab/)

## Domain Generalization and DomainLab

Domain Generalization aims at learning domain invariant features by utilizing data from multiple domains (data sites, corhorts, batches, vendors) so the learned feature can generalize to new unseen domains. 

DomainLab is a software platform with state-of-the-art domain generalization algorithms implemented, designed by maximal decoupling of different software componets thus enhances maximal code reuse.

- To evaluate a domain generalization algorithm's performance on your custom data, the user only need to specify ways to access the data in the form of a standalone python file. See [Task Specification](./docs/doc_tasks.md).

- To benchmark several algorithms on your dataset, a single line command along with a benchmark configuration files is sufficient. See [Benchmarks](./docs/doc_benchmark.md)



## Getting started

### Installation
#### Development version (recommended)

Suppose you have cloned the repository and have changed directory to the cloned repository.

```shell
pip install -r requirements.txt
```
then 

`python setup.py install`

#### Windows installation details

To install DomainLab on Windows, please remove the `snakemake` dependency from the `requirements.txt` file.
Benchmarking is currently not supported on Windows due to the dependency on Snakemake.

#### Dependencies management
-   [python-poetry](https://python-poetry.org/) and use the configuration file `pyproject.toml` in this repository.
 
#### Release
- Install via `pip install domainlab`

### Basic usage
DomainLab comes with some minimal toy-dataset to test its basis functionality, see [A minimal subsample of the VLCS dataset](./data/vlcs_mini) and [an example configuration file for vlcs_mini](./examples/tasks/task_vlcs.py)

Suppose you have cloned the repository and have the dependencies ready, change directory to this repository:

To train a domain invariant model on the vlcs_mini task

```shell
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --config=examples/yaml/demo_config_single_run_diva.yaml 
```
where `--tpath` specifies the path of a user specified python file which defines the domain generalization task [see here](./examples/tasks/task_vlcs.py), `--te_d` specifies the test domain name (or index starting from 0), `--config` specifies the configurations of the domain generalization algorithms, [see here](./examples/yaml/demo_config_single_run_diva.yaml)

#### Further usage
Alternatively, in a verbose mode without using the algorithm configuration file:

```shell
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --debug --bs=2 --aname=diva --gamma_y=7e5 --gamma_d=1e5 --nname=alexnet --nname_dom=conv_bn_pool_2
```

where `--aname` specifies which algorithm to use, see [Available Algorithms](./docs/doc_algos.md), `--bs` specifies the batch size, `--debug` restrain only running for 2 epochs and save results with prefix 'debug'. For DIVA, the hyper-parameters include `--gamma_y=7e5` which is the relative weight of ERM loss compared to ELBO loss, and `--gamma_d=1e5`, which is the relative weight of domain classification loss compared to ELBO loss.
`--nname` is to specify which neural network to use for feature extraction for classification, `--nname_dom` is to specify which neural network to use for feature extraction of domains.
For usage of other arguments, check with

```shell
python main_out.py --help
```

See also [Examples](./docs/doc_examples.md).

### Output structure (results storage) and Performance Measure
[Output structure and Performance Measure](./docs/doc_output.md)

## Custom Usage

### Define your task
Do you have your own data that comes from different domains? Create a task for your data and benchmark different domain generlization algorithms according to the following example. See
[Task Specification](./docs/doc_tasks.md)

### Custom Neural network
This library decouples the concept of algorithm (model) and neural network architecture where the user could plugin different neural network architectures for the same algorithm. See
[Specify Custom Neural Networks for an algorithm](./docs/doc_custom_nn.md)

## Software Design Pattern, Extend or Contribution
[Extend or Contibute](./docs/doc_extend_contribute.md)
