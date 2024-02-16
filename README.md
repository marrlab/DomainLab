# DomainLab: modular python package for training domain invariant neural networks

![GH Actions CI ](https://github.com/marrlab/DomainLab/actions/workflows/ci.yml/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/marrlab/DomainLab/branch/master/graph/badge.svg)](https://app.codecov.io/gh/marrlab/DomainLab)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/bc22a1f9afb742efb02b87284e04dc86)](https://www.codacy.com/gh/marrlab/DomainLab/dashboard)
[![Documentation](https://img.shields.io/badge/Documentation-Here)](https://marrlab.github.io/DomainLab/)
[![pages-build-deployment](https://github.com/marrlab/DomainLab/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/marrlab/DomainLab/actions/workflows/pages/pages-build-deployment)

## Distribution shifts, domain generalization and DomainLab

Neural networks trained using data from a specific distribution (domain) usually fails to generalize to novel distributions (domains). Domain generalization aims at learning domain invariant features by utilizing data from multiple domains (data sites, corhorts, batches, vendors) so the learned feature can generalize to new unseen domains (distributions).

DomainLab is a software platform with state-of-the-art domain generalization algorithms implemented, designed by maximal decoupling of different software components thus enhances maximal code reuse.

### DomainLab
DomainLab decouples the following concepts or objects:
- task $M$: In DomainLab, a task is a container for datasets from different domains. (e.g. from distribution $D_1$ and $D_2$). Task offer a static protocol to evaluate the generalization performance of a neural network: which dataset(s) is used for training, wich dataset(s) used for testing.  
- neural network: a map $\phi$ from the input data to the feature space and a map $\varphi$ from feature space to output $\hat{y}$ (e.g. decision variable).
- model: structural risk in the form of $\ell() + \mu R()$  where
  - $\ell(Y, \hat{y}=\varphi(\phi(X)))$ is the task specific empirical loss (e.g. cross entropy for classification task).
  - $R(\phi(X))$ is the penalty loss to boost domain invariant feature extraction using $\phi$.
  - $\mu$ is the corresponding multiplier to each penalty function factor.
- trainer:  an object that guides the data flow to model and append further domain invariant losses
like inter-domain feature alignment.

We offer detailed documentation on how these models and trainers work in our documentation page: https://marrlab.github.io/DomainLab/

DomainLab makes it possible to combine models with models, trainers with models, and trainers with trainers in a decorator pattern like line of code `Trainer A(Trainer B(Model C(Model D(network E), network E, network F)))` which correspond to $\ell() + \mu_a R_a() + \mu_b R_b + \mu_c R_c() + \mu_d R_d()$, where Model C and Model D share neural network E, but Model C has an extra neural network F. All models share the same neural network for feature extraction, but can have different auxilliary networks for $R()$.

<div style="align: center; text-align:center;">
<figure>  
<img src="https://github.com/marrlab/DomainLab/blob/master/docs/figs/invarfeat4dg.png?raw=true" style="width:300px;"/> 
</figure>
</div>

## Getting started

### Installation
For development version in Github, see [Installation and Dependencies handling](./docs/doc_install.md)

We also offer a PyPI version here https://pypi.org/project/domainlab/  which one could install via `pip install domainlab` and it is recommended to create a virtual environment for it.

### Task specification
We offer various ways for the user to specify a scenario to evaluate the generalization performance via training on a limited number of datasets. See detail in
[Task Specification](./docs/doc_tasks.md) 

### Example and usage

#### Command line

`domainlab -c ./examples/conf/vlcs_diva_mldg_dial.yaml` (if you install via pip)

or if you clone this the code repository for DomainLab

`python main_out.py -c ./examples/conf/vlcs_diva_mldg_dial.yaml`

where the configuration file below can be downloaded [here](https://raw.githubusercontent.com/marrlab/DomainLab/master/examples/conf/vlcs_diva_mldg_dial.yaml)
```
te_d: caltech                       # domain name of test domain
tpath: examples/tasks/task_vlcs.py  # python file path to specify the task
bs: 2                               # batch size
model: dann_diva                    # combine model DANN with DIVA
epos: 1                             # number of epochs
trainer: mldg_dial                  # combine trainer MLDG and DIAL
gamma_y: 700000.0                   # hyperparameter of diva
gamma_d: 100000.0                   # hyperparameter of diva
npath: examples/nets/resnet.py      # neural network for class classification
npath_dom: examples/nets/resnet.py  # neural network for domain classification
```
See details in [Command line usage](./docs/doc_usage_cmd.md)

#### or Programm against DomainLab API

See example here: [Transformer as feature extractor, decorate JIGEN with DANN, training using MLDG decorated by DIAL](https://github.com/marrlab/DomainLab/blob/master/examples/api/jigen_dann_transformer.py)


### Benchmark different methods
DomainLab provides a powerful benchmark functionality.
To benchmark several algorithms(combination of neural networks, models, trainers and associated hyperparameters), a single line command along with a benchmark configuration files is sufficient. See details in [benchmarks documentation and tutorial](./docs/doc_benchmark.md)

One could simply run
`bash run_benchmark_slurm.sh your_benchmark_configuration.yaml` to launch different experiments with specified configuraiton.


For example,  the following result (without any augmentation like flip) is for PACS dataset using ResNet. 

<div style="align: center; text-align:center;">
<figure>  
<img src="https://github.com/marrlab/DomainLab/blob/master/docs/figs/stochastic_variation_two_rows.png?raw=true" style="width:800px;"/>
<div class="caption" style="align: center; text-align:center;">
<figcaption>Benchmark results plot generated from DomainLab, where each rectangle represent one model trainer combination, each bar inside the rectangle represent a unique hyperparameter index associated with that method combination, each dot represent a random seeds.</figcaption>
</div>
</figure>
</div>


### Temporary citation

```bibtex
@manual{domainlab,
  title={{DomainLab: modular python package for training domain invariant neural networks}},
  author={{Xudong Sun, et.al.}},
  organization={{Institute of AI for Health}},
  year={2023},
  url={https://github.com/marrlab/DomainLab},
  note={temporary citation for domainlab}
}
```
