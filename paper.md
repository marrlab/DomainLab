# DomainLab: A PyTorch library for causal domain generalization

## Summary
Deep learning (DL) models have solved real-world challenges in various areas, such as computer vision, natural language processing, medical imaging, and healthcare. While generalizing to unseen test domains comes naturally to humans, it’s still a major obstacle for machines. By design,  most DL models assume that training and testing distributions are the same, causing them to fail when this isn’t the case. Instead, domain generalization aims at training domain invariant models that are robust to distribution shifts. We introduce DomainLab, a PyTorch based Python package for domain generalization. DomainLab focuses on causal domain generalization and probabilistic methods, while offering easy extensibility to a wide range of other methods including adversarial methods, self-supervised learning and other training paradigms. 

Compared to existing solutions, DomainLab aims at maximum decoupling of different contributing factors to the performance of domain generalization, i.e. how the data get split, and the evaluation protocol is decoupled from the loss model, the exact neural network architecture which represents different parts of a mathematical model, and how the weights are updated.  Such that the user can make free combinations of the aforementioned factors. 

DomainLab’s documentation is hosted on https://marrlab.github.io/DomainLab and its source code can be found at https://github.com/marrlab/DomainLab. 

## Statement of need 

Over the past years, various methods have been proposed addressing different aspects of domain generalization. However, their implementations are often limited to proof-of-concept code, interspersed with custom code for data access, pre-processing, evaluation, etc. This limits the applicability of these methods, affects reproducibility, and restricts the ability to perform comparisons with other state-of-the-art methods. 

DomainBed for the first time provided a common codebase for benchmarking domain generalization methods (Gulrajani and Lopez-Paz 2020), however applying its algorithms to new use-cases requires extensive adaptation of its source code. More recently, Dassl provides a Python package to benchmark different algorithms such as semi-supervised learning, domain adaptation and domain generalization (Zhou et al. 2021). Its design is more modular than DomainBed but not yet modularized to the level of DomainLab and our package focuses on generative modeling. 


With DomainLab, we introduce a fully modular PyTorch based Python package for domain generalization that follows best practices in software design and includes extensive documentation to enable the community to understand and contribute to the code. It contains extensive unit tests as well as end-to-end tests to verify the implemented functionality. 

For an ideal package for domain generalization, the package should offer decoupling of different contributing factors: for an algorithm with good performance, it should discretize:
Is this mostly attributed to a more appropriate neural network architecture of model A compared to others? 
Is this mostly attributed to the protocol of estimating the generalization performance? e.g. dataset split. 
Is this mostly attributed to a special loss function which offers a better regularization to the training of the neural network?

## Description
### General Design 
To address software design issues of existing code bases like DomainBed (Gulrajani and Lopez-Paz 2020) and Dassl (Zhou et al. 2021), and to maximally decouple factors that might affect the performance of domain generalization algorithms, we designed DomainLab with the following features:

First, the package offers the user a standalone facility to specify the data, data split, pre-processing, etc, which will not modify the code base of DomainLab. (connect user’s data to algorithms)

Domain generalization algorithms were implemented with a transparent underlying neural network architecture.  The concrete neural network architecture can thus be replaced like a plugin through specifying an architecture implemented in a python file or specifying a string of some existing neural network like AlexNet, via command line arguments.

Selection of algorithms, neural network components, as well as other components like training procedure are done via chain-of-responsibility method. Other design patterns like observer pattern, visitor pattern, etc. are also used to improve decoupling.  

Instead of modifying code across several python files, the package is closed to modification and open to extension. To simply test an algorithm’s performance on the user’s data, there is no need to change any code inside this repository, the user only needs to extend this repository to fit their custom need by providing custom python files. 

It offers a framework for generating combinations by simply letting the user select elements through command line arguments. (combine tasks, neural network architectures)
With the above design, DomainLab offers users the flexibility, by constructing custom tasks incorporating their data, writing custom neural network architectures, and even trying their own algorithms by specifying a python file with custom loss functions. There is no need to change the original code of DomainLab when the user needs to use the domain generalization method to their own application, extend the method with custom neural network and try to discriminate the most significant factor that affects performance. 

### Components  
To achieve the above design goals of decoupling, we used the following components:

Models refer to a PyTorch module with a specified loss function containing regularization effect of several domains plus the task-specific loss, which is classification loss for classification task, but stay transparent with respect to the exact neural network architecture, which can be configured by the user via command line arguments. 

There are two types of models
- implemented models from publications in the field of domain generalization using causality and probabilistic model based methods
- custom models, where the user only needs to specify a python file defining the custom loss function, while remain transparent of the exact neural network used for each submodule. 

	The common classification loss calculation is done via a parent model class, thus the individual models representing different domain regularization can be reused for other tasks like segmentation by simply inheriting another task loss. 
  
Tasks refer to a component, where the user specifies different datasets from different domains and preprocessing specified upon them. There are several types of tasks in DomainLab:
Built-in tasks like Color-Mnist, subsampled version of PACS, VLCS, as test utility of algorithms.
TaskFolder: If the data is already organized in a root folder, with different subfolders containing data from different domains and a further level of sub-sub-folders containing data from different classes.   
TaskPathFile: This allows the user to specify each domain a text file indicating the path and label for each image. Thus, the user can choose which portion of images to use as training, validation and test. 

Trainer is the component that directs data flow to the model to calculate loss and back-propagation to update the parameters, several models can share a common trainer. A specific trainer can also be a visitor to models to update coefficients in models during training to implement techniques like warm-up. This follows the visitor design pattern. 

Following the observer pattern, we use separate classes to conduct operations needed to be done after each epoch (e.g. deciding whether to execute early stopping) and after training finishes. 

Following the Builder Pattern, we construct each component needed to conduct a domain generalization experiment, including 
constructing a trainer which guides the data flow.
constructing a concrete neural network architecture and feeding into the model.
constructing the evaluator as a callback of what to do after each epoch.

## Availability
Domainlab is free and open source. It is published under the MIT License. You can download the source code at https://github.com/marrlab/DomainLab. Extensive documentation can be found here at https://marrlab.github.io/DomainLab. DomainLab can be installed using python-poetry or pip.

## Contributions (optional)
Xudong Sun has designed the package with software design patterns and implemented the algorithms. He initiated and made major contributions to other aspects of the package development. 

[References]
- DIVA: Domain Invariant Variational Autoencoders, https://arxiv.org/pdf/1905.10427.pdf
- Hierarchical Variational Auto-Encoding for Unsupervised Domain Generalization:https://arxiv.org/pdf/2101.09436.pdf'
- Domain Generalization using Causal Matching, https://arxiv.org/abs/2006.07500',
- Domain adversarial invariant feature learning
- Domain Generalization by Solving Jigsaw Puzzles, https://arxiv.org/abs/1903.06864
- Gulrajani, Ishaan, and David Lopez-Paz. 2020. In Search of Lost Domain Generalization.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2007.01434.
- Wang, Jindong, Cuiling Lan, Chang Liu, Yidong Ouyang, Tao Qin, Wang Lu, Yiqiang Chen, Wenjun Zeng, and Philip S. Yu. 2021. Generalizing to Unseen Domains: A Survey on Domain Generalization. arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2103.03097.
- Zhou, Kaiyang, Ziwei Liu, Yu Qiao, Tao Xiang, and Chen Change Loy. 2022. Domain Generalization: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence PP (August). https://doi.org/10.1109/TPAMI.2022.3195549.
