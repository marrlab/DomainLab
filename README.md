# DomainLab: modular python package for training domain invariant neural networks

![GH Actions CI ](https://github.com/marrlab/DomainLab/actions/workflows/ci.yml/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/marrlab/DomainLab/branch/master/graph/badge.svg)](https://app.codecov.io/gh/marrlab/DomainLab)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/bc22a1f9afb742efb02b87284e04dc86)](https://www.codacy.com/gh/marrlab/DomainLab/dashboard)
[![Documentation](https://img.shields.io/badge/Documentation-Here)](https://marrlab.github.io/DomainLab/)
[![pages-build-deployment](https://github.com/marrlab/DomainLab/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/marrlab/DomainLab/actions/workflows/pages/pages-build-deployment)

## Distribution shifts, domain generalization and DomainLab

Neural networks trained using data from a specific distribution (domain) usually fails to generalize to novel distributions (domains). Domain generalization aims at learning domain invariant features by utilizing data from multiple domains (data sites, corhorts, batches, vendors) so the learned feature can generalize to new unseen domains (distributions). 

DomainLab is a software platform with state-of-the-art domain generalization algorithms implemented, designed by maximal decoupling of different software componets thus enhances maximal code reuse.

As an input to the software, the user need to provide 
- the neural network to be trained for the task (e.g. classification)
- task specification which contains dataset(s) from domain(s). 

DomainLab decouples the following concepts or objects:
- neural network: a map from the input data to the feature space and output.
- model: structural risk in the form of $\ell() + \mu R()$  where $\ell()$ is the task specific empirical loss (e.g. cross entropy for classification task) and $R()$ is the penalty loss for inter-domain alignment (domain invariant regularization).
- trainer:  an object that guides the data flow to model and append further domain invariant losses.

DomainLab makes it possible to combine models with models, trainers with models, and trainers with trainers in a decorator pattern like `Trainer A(Trainer B(Model C(Model D(network E), network F)))` which correspond to $\ell() + \mu_a R_a() + \mu_b R_b + \mu_c R_c() + \mu_d R_d()$ 

## Getting started

### Installation
For development version in Github, see [Installation and Dependencies handling](./docs/doc_intall.md)

We also offer a PyPI version here https://pypi.org/project/domainlab/  which one could install via `pip install domainlab` and it is recommended to create a virtual environment for it. 

### Task specification
In DomainLab, a task is a container for datasets from different domains. See detail in
[Task Specification](./docs/doc_tasks.md)

### Example and usage

#### Either clone this repo and use command line 
See details in [Command line usage](./docs/doc_usage_cmd.md)

#### or Programm against DomainLab API

As a user, you need to define neural networks you want to train. As an example, here we define a transformer neural network for classification in the following code. 
```
from torch import nn                                                                                     
from torchvision.models import vit_b_16                                                                  
from torchvision.models.feature_extraction import create_feature_extractor

class VIT(nn.Module):                                                                                    
    def __init__(self, num_cls, freeze=True,                                                             
                 list_str_last_layer=['getitem_5'],                                                      
                 len_last_layer=768):                                                                    
        super().__init__()                                                                               
        self.nets = vit_b_16(pretrained=True)                                                            
        if freeze:                                                                                                                                    
            for param in self.nets.parameters():                                                         
                param.requires_grad = False                                                              
        self.features_vit_flatten = create_feature_extractor(self.nets, return_nodes=list_str_last_layer)           
        self.fc = nn.Linear(len_last_layer, num_cls)                                                     
                                                                                                         
    def forward(self, tensor_x):                                                                         
        """
        compute logits predicts
        """
        x = self.features_vit_flatten(tensor_x)['getitem_5']
        out = self.fc(x)
        return out
```
Then we plug this neural network in our model:
```
from domainlab.mk_exp import mk_exp                                                                      
from domainlab.tasks import get_task                                                                     
from domainlab.models.model_deep_all import mk_deepall

task = get_task("mini_vlcs")
nn = VIT(num_cls=task.dim_y, freeze=True)
model = mk_deepall()(nn)
# use trainer MLDG, DIAL
exp = mk_exp(task, model, trainer="mldg,dial",   # combine two trainers
             test_domain="caltech", batchsize=2, nocu=True)
exp.execute(num_epochs=2)
```


### Benchmark different methods
DomainLab provides a powerful benchmark functionality. 
To benchmark several algorithms, a single line command along with a benchmark configuration files is sufficient. See details in [Benchmarks](./docs/doc_benchmark.md)
