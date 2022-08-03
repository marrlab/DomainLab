DomainLab: Playground for Domain Generalization
================================================

![GH Actions CI ](https://github.com/marrlab/DomainLab/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/marrlab/DomainLab/branch/main/graph/badge.svg)](https://app.codecov.io/gh/marrlab/DomainLab)


## Domain Generalization and DomainLab

### Domain Generalization

Domain Generalization aims at learning domain invariant features by utilizing data from multiple domains so the learned feature can generalize to new unseen domains. 


### Why a dedicated package?

Domain generalization algorithm try to learn domain invariant features by adding regularization upon the ERM (Emperical Risk Minimization) loss. A typical setting of evaluating domain generalization algorithm is the so called leave-one-domain-out scheme, where one dataset is collected from each distribution. Each time, one dataset/domain is left as test-set to estimate the generalization performance of a model trained upon the rest of domains/datasets.


Once you came across a claim,  that a domain generalization algorithm A can generate a "better" model  h upon some datasets D with "better" performance compared to other algorithms, have you ever wondered:

- Is this mostly attributed to a more "powerful" neural network architecture of model A compared to others? What will happen if I change the backbone neural network of algorithm A from ResNet to AlexNet?
- Is this mostly attributed the protocol of estimating the generalization performance? e.g. dataset split, Will this algorithm "work" for my datasets?
- Is this mostly attributed to the "clever" regularization algorithm or a special loss function A has used for the neural network?

To maximally decouple these attributing factors, DomainLab was implemented with software design patterns, where

- Domain generalization algorithms was implemented in a way that keeps the underlying neural network architecture transparent, i.e. the concrete neural network architecture can be replaced like a plugin through specifying a custom neural network architecture implemented in a python file. See [Specify Custom Neural Networks for an algorithm](./docs/doc_custom_nn.md) 

- To evaluate a domain generalization algorithm's performance, the user can specify a "Task" in the form of custom python file and feed into the command line argument, thus it is at the user's discretion on how to evaluate an algorithm, so that all domain generalization algorithms could be compared fairly. See [Task Specification](./docs/doc_tasks.md).

- To simply test an algorithm's performance, there is no need to change any code inside this repository, the user only need to extend this repository to fit their custom need.

# Getting started
## Installation

- Install via python-poetry:
Read the python-poetry documentation https://python-poetry.org/ and use the configuration file in this repository.

- **Or** only install dependencies via pip
Suppose you have cloned the repository and have changed directory to the cloned repository.
```
pip install -r requirements.txt
```

## Basic usage
Suppose you have cloned the repository and the dependencies ready, change directory to the repository:
DomainLab comes with some minimal toy-dataset to test its basis functionality. To train a domain generalization model with a user-specified task, one can execute a command similar to the following.
```
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --debug --bs=2 --aname=diva --gamma_y=7e5 --gamma_d=1e5 --nname=alexnet --nname_dom=conv_bn_pool_2
```
where `--tpath` specifies the path of a user specified python file which defines the domain generalization task, see Example in [Task Specification](./docs/doc_tasks.md). `--aname` specifies which algorithm to use, see [Available Algorithms](./docs/doc_algos.md), `--bs` specifies the batch size, `--debug` restrain only running for 2 epochs and save results with prefix 'debug'. For DIVA, the hyper-parameters include `--gamma_y=7e5` which is the relative weight of ERM loss compared to ELBO loss, and  `--gamma_d=1e5`, which is the relative weight of domain classification loss compared to ELBO loss.
`--nname` is to specify which neural network to use for feature extraction for classification, `--nname_dom` is to specify which neural network to use for feature extraction of domains.
For usage of other arguments, check with 

```
python main_out.py --help
```

See also [Examples](./examples.sh).

### Output structure (results storage) and Performance Measure
[Output structure and Performance Measure](./docs/doc_output.md)

## Custom Usage

### Define your task 
Do you have your own data that comes from different domains? Create a task for your data and benchmark different domain generlization algorithms according to the following example. See
[Task Specification](./docs/doc_tasks.md) 

### Custom Neural network 
This library decouples the concept of algorithm (model) and neural network architecture where the user could plugin different neural network architectures for the same algorithm. See
[Specify Custom Neural Networks for an algorithm](./docs/doc_custom_nn.md) 

# Software Design Pattern, Extend or Contribution, Credits
[Extend or Contibute](./docs/doc_extend_contribute.md)
