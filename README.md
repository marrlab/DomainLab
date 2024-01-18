# DomainLab: modular python package for training domain invariant neural networks

![GH Actions CI ](https://github.com/marrlab/DomainLab/actions/workflows/ci.yml/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/marrlab/DomainLab/branch/master/graph/badge.svg)](https://app.codecov.io/gh/marrlab/DomainLab)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/bc22a1f9afb742efb02b87284e04dc86)](https://www.codacy.com/gh/marrlab/DomainLab/dashboard)
[![Documentation](https://img.shields.io/badge/Documentation-Here)](https://marrlab.github.io/DomainLab/)
[![pages-build-deployment](https://github.com/marrlab/DomainLab/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/marrlab/DomainLab/actions/workflows/pages/pages-build-deployment)

## Distribution shifts, domain generalization and DomainLab

Neural networks trained using data from a specific distribution (domain) usually fails to generalize to novel distributions (domains). Domain generalization aims at learning domain invariant features by utilizing data from multiple domains (data sites, corhorts, batches, vendors) so the learned feature can generalize to new unseen domains (distributions).

<div style="align: center; text-align:center;">
<img src="https://github.com/marrlab/DomainLab/blob/master/docs/figs/invarfeat4dg.png?raw=true" style="width:400px;"/>
</div>

DomainLab is a software platform with state-of-the-art domain generalization algorithms implemented, designed by maximal decoupling of different software components thus enhances maximal code reuse.

DomainLab decouples the following concepts or objects:
- task $M$: a combination of datasets (e.g. from distribution $D_1$ and $D_2$)
- neural network: a map $\phi$ from the input data to the feature space and a map $\varphi$ from feature space to output $\hat{y}$ (e.g. decision variable).
- model: structural risk in the form of $\ell() + \mu R()$  where
  - $\ell(Y, \hat{y}=\varphi(\phi(X)))$ is the task specific empirical loss (e.g. cross entropy for classification task).
  - $R(\phi(X))$ is the penalty loss to boost domain invariant feature extraction using $\phi$.
  - $\mu$ is the corresponding multiplier to each penalty function factor.
- trainer:  an object that guides the data flow to model and append further domain invariant losses
like inter-domain feature alignment.

We offer detailed documentation on how these models and trainers work in our documentation page: https://marrlab.github.io/DomainLab/

DomainLab makes it possible to combine models with models, trainers with models, and trainers with trainers in a decorator pattern like line of code `Trainer A(Trainer B(Model C(Model D(network E), network E, network F)))` which correspond to $\ell() + \mu_a R_a() + \mu_b R_b + \mu_c R_c() + \mu_d R_d()$, where Model C and Model D share neural network E, but Model C has an extra neural network F. All models share the same neural network for feature extraction, but can have different auxilliary networks for $R()$.

## Getting started

### Installation
For development version in Github, see [Installation and Dependencies handling](./docs/doc_install.md)

We also offer a PyPI version here https://pypi.org/project/domainlab/  which one could install via `pip install domainlab` and it is recommended to create a virtual environment for it.


#### Guide for Helmholtz GPU cluster
```
conda create --name domainlab_py39 python=3.9
conda activate domainlab_py39
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install torchmetrics==0.10.3
git checkout fbopt
pip install -r requirements_notorch.txt
conda install tensorboard
```

#### Download PACS

step 1:

use the following script to download PACS to your local laptop and upload it to your cluster

https://github.com/marrlab/DomainLab/blob/fbopt/data/script/download_pacs.py

step 2:
make a symbolic link following the example script in https://github.com/marrlab/DomainLab/blob/master/sh_pacs.sh

where `mkdir -p data/pacs` is executed under the repository directory,

`ln -s /dir/to/yourdata/pacs/raw  ./data/pacs/PACS`
will create a symbolic link under the repository directory


### Example and usage

#### Either clone this repo and use command line

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


For example,  the following result (without any augmentation like flip) is for PACS dataset.

<div style="align: center; text-align:center;">
<img src="https://github.com/marrlab/DomainLab/blob/master/docs/figs/stochastic_variation_two_rows.png?raw=true" style="width:800px;"/>
</div>
where each rectangle represent one model trainer combination, each bar inside the rectangle represent a unique hyperparameter index associated with that method combination, each dot represent a random seeds.
