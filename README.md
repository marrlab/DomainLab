LibDG: Library of Domain Generalization
================================================
## Domain Generalization and libDG

### Domain Generalization

Domain Generalization aims at learning domain invariant features by utilizing data from multiple domains so the learned feature can generalize to new unseen domains. 


### Why a dedicated library?

Domain generalization algorithm try to learn domain invariant features by adding regularization upon the ERM (Emperical Risk Minimization) loss. A typical setting of evaluating domain generalization algorithm is the so called leave-one-domain-out scheme, where one dataset is collected from each distribution. Each time, one dataset/domain is left as test-set to estimate the generalization performance of a model trained upon the rest of domains/datasets.


Once you came across a claim,  that a domain generalization algorithm A can generate a "better" model  h upon some datasets D with "better" performance compared to other algorithms, have you ever wondered:

- Is this mostly attributed to a more "powerful" neural network architecture of model A compared to others? What will happen if I change the backbone neural network of algorithm A from ResNet to AlexNet?
- Is this mostly attributed the protocol of estimating the generalization performance? e.g. dataset split, Will this algorithm "work" for my datasets?
- Is this mostly attributed to the "clever" regularization algorithm A has used?

To maximally decouple these attributing factors, LibDG was implemented with software design patterns, where

- Domain generalization algorithms was implemented in a way that keeps the underlying neural network architecture transparent, i.e. the concrete neural network architecture can be replaced like a plugin through specifying a custom neural network architecture implemented in a python file.
- To evaluate a domain generalization algorithm's performance, the user can specify a "Task" in the form of custom python file and feed into the command line argument. See [Task Specification](libdg/tasks/README.md) 

# Getting started
## Basic usage
Clone the repository if you do not want to install it
```
git clone git@github.com:smilesun/libDG.git
cd libDG
```
LibDG comes with some minimal toy-dataset to test its basis functionality. To train a domain generalization model with a user-specified task, one can execute a command similar to the following.
```
python main_out.py --te_d=caltech --tpath=./examples/task_vlcs.py --debug --bs=20 --aname=diva
```
where `--tpath` specifies the path of a user specified python file which defines the domain generalization task, see Example in [Task Specification](libdg/tasks/README.md). `--aname` specifies which algorithm to use, see [Available Algorithms](libdg/algos/README.md), `--bs` specifies the batch size, `--debug` restrain only running for 2 epochs and save results with prefix 'debug'.

For usage of other arguments, check with `python main_out.py --help`

### Output structure
By default, libDG generate outputs to a folder called "zoutput" relative to the current working directory. The structure is something similar as follows. ([] means the folder might or might not exist)

```
zoutput/
├── aggrsts (aggregation of results)
│   ├── task1_test_domain1_tagName.csv
│   ├── task2_test_domain3_tagName.csv
│   
│  
├── [gen] (counterfactual image generation)
│   ├── task1_test_domain1
│   
└── saved_models (persisted pytorch model)
    ├── task1_algo1_git-commit-hashtag1_seed_1.model
    ├── task1_algo1_git-commit-hashtag1_seed_1.model_oracle
```

# Design
![Design Diagram](libDG.svg)

# Extend libDG with a domain generalization algorithm

## Playground extension
Look at this example:
```
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --apath=./examples/algos/builder_deepall_copy.py --aname=deepall2
```

## Contribute to libDG by implementing a serious algorithm
- implement libdg/algos/builder_your-algorithm-name.py
- add your algorithm into libdg/algos/zoo_algos.py by adding `chain = NodeAlgoBuilder[your-algorithm-name](chain)`
- note that all algorithms will be converted to lower case!
- make a pull request

# Credits
Contact: Xudong Sun (smilesun.east@gmail.com, Institute of AI for Health, Helmholtz Munich, Germany)

Please cite our paper if you use this code in your research:
```
@inproceedings{sun2021hierarchical,
  title={Hierarchical Variational Auto-Encoding for Unsupervised Domain Generalization},
  author={Sun, Xudong and Buettner, Florian},
  booktitle={ICLR 2021 RobustML workshop, https://arxiv.org/pdf/2101.09436.pdf},
  year={2021}
}
```
