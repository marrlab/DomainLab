---
title: 'DomainLab: A PyTorch library for domain generalization in deep learning'
tags:
  - Python
  - deep learning
  - distribution shift
  - domain generalization
  - causality
authors:
  - name: Xudong Sun
    orcid: 0000-0001-9234-4932
    equal-contrib: False
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Alexej Gossmann
    orcid: 0000-0001-9068-3877
    equal-contrib: False
    affiliation: 2
  - name: Carla Feistner
    orcid: 0000-0003-0111-3835
    equal-contrib: False
    affiliation: 4
  - name: Georg Schwarz
    orcid: 0000-0002-1431-7725
    equal-contrib: False
    affiliation: 4
  - name: Rahul Babu Shrestha
    orcid: 0000-0002-6429-4402
    equal-contrib: False
    affiliation: 4
  - name: Xinyue Zhang
    orcid: 0000-0003-4806-4049
    equal-contrib: False
    affiliation: 4
  - name: Sayedali Shetab Boushehri
    orcid: 0000-0003-3391-9294
    equal-contrib: False 
    affiliation: "1,3,4"
  - name: Umer Muhammad Rao
    orcid: 0000-0001-6179-5829
    equal-contrib: False 
    affiliation: "1"
  - name: Patrick Rockenschaub
    orcid: 0000-0002-6499-7933
    equal-contrib: False
    affiliation: 6
  - name: Armin Gruber
    orcid: 0000-0002-5015-6543
    equal-contrib: False 
    affiliation: "1,5"
  - name: Florian Buettner
    orcid: 0000-0001-5587-6761
    equal-contrib: False 
    affiliation: "7,8,9,10"   
  - name: Carsten Marr
    orcid: 0000-0003-2154-4552
    corresponding: true #  
    affiliation: "1"
    
affiliations:
 - name: Institute of AI for Health, Helmholtz Munich, Munich, Germany
   index: 1
 - name: Division of Imaging, Diagnostics, and Software Reliability, CDRH, U.S Food and Drug Administration, Silver Spring, MD 20993, USA
   index: 2
 - name: Data Science, Pharmaceutical Research and Early Development Informatics (pREDi), Roche Innovation Center Munich, Germany
   index: 3
 - name: Technical University of Munich, Munich, Germany 
   index: 4
 - name: Ludwig Maximilian University of Munich
   index: 5
 - name: Charité Lab for AI in Medicine, Charité Universitätsmedizin Berlin, Berlin, Germany
   index: 6
 - name: German Cancer Research Center (DKFZ), Heidelberg, Germany
   index: 7 
 - name: Frankfurt Cancer Institute, Frankfurt, Germany
   index: 8
 - name: German Cancer Consortium (DKTK), Germany
   index: 9 
 - name: Goethe University Frankfurt, Germany
   index: 10    
date: 17 Feb 2023
bibliography: paper.bib
---


# Summary

Deep learning (DL) models have showed success in tackling real-world challenges in various areas, such as computer vision and medical imaging or computational pathology. However, while generalizing to unseen data domains comes naturally to humans, it's still a major obstacle for machines. By design, most DL models assume that training and testing distributions are well aligned, causing them to fail when this is violated. Instead, domain generalization aims at training domain invariant models that are robust to distribution shifts [@wang2022generalizing].

We introduce DomainLab, a Python package for domain generalization. Compared to existing solutions, DomainLab excels at the extent of modularization by decoupling the various factors that contribute to the performance of a domain generalization method: How the domain invariant regularization loss is computed, which neural network architectures are used for each component, how the neural network weights are updated, how the data are fed into neural network and which evaluation protocol is applied are defined independently. The user can mix and match different combinations of the individual components and evaluate the impact on generalization performance.

DomainLab currently supports Pytorch backend.

Additionally, thanks to the modularized design patterns of DomainLab, domain generalization methods like causal domain generalization and probabilistic methods [@ilse2020diva], [@mahajan2021domain], [@sun2021hierarchical] which does not exist in other existing solutions (possibly due to software design limitations), have been implemented, which can be easily integrated with adversarial methods [@levi2021domain], [@ganin2016domain], [@akuzawa2020adversarial], self-supervised learning [@carlucci2019domain] and other training paradigms [@rame2022fishr]. 

DomainLab's documentation is hosted at <https://marrlab.github.io/DomainLab> and its source code can be found at <https://github.com/marrlab/DomainLab>.

# Statement of need 

Over the past years, various methods have been proposed addressing different aspects of domain generalization. However, their implementations are often limited to proof-of-concept code, interspersed with custom code for data access, pre-processing, evaluation, etc. This limits the applicability of these methods, affects reproducibility, and restricts the ability to perform comparisons with other state-of-the-art methods.

*DomainBed* for the first time provided a common codebase for benchmarking domain generalization methods [@gulrajani2020search], however applying its algorithms to new use-cases requires extensive adaptation of its source code, including, for instance, that the neural network backbones are hardcoded. All components of an algorithm have to be initialized in the construction function, which is not suitable for complex algorithms that require flexibility and extensibility of its components. A more recent concurrent work, *Dassl* [@dassl2022github], provides a Python package to benchmark different machine learning algorithms, such as DL models for semi-supervised learning, domain adaptation and domain generalization [@zhou2021domain]. Its design is more modular than DomainBed. However, the documentation does not contain enough details about algorithm implementation and the code base does not appear to be well tested. In addition, the authors have not clarified a plan for maintaining their Python module, while we aim at a long term maintenance of our package 

With DomainLab, we introduce a fully modular Python package for domain generalization with a Pytorch backend that follows best practices in software design and includes extensive documentation, which enables the research community to understand and contribute to the code. The DomainLab codebase contains extensive unit tests as well as end-to-end tests to verify the implemented functionality.

An ideal package for domain generalization should decouple the factors that affect model performance. This way, the components that contributed most to a promising result can be isolated, allowing for better comparability between methods. Such decoupling would allow to answer many important research questions, including for example the following.
Can the results be ascribed to a more appropriate neural network architecture?
Is the performance impacted by the protocol used to estimate the generalization performance, e.g., the dataset split?
Does the model benefit from a specific loss function, e.g., because it offers a better regularization to the training of the neural network?

# Description

## General Design

To address software design issues of existing code bases, such as DomainBed [@gulrajani2020search] and Dassl [@zhou2021domain], and to maximally decouple factors that might affect the performance of domain generalization algorithms, we designed DomainLab with the following features.

First, the package offers the user a standalone application to specify the data, data split protocol, pre-processing, neural network backbone, and model loss function, which will not modify the code base of DomainLab. That is, it connects a user's data to algorithms.
Domain generalization algorithms, such as DIVA [@ilse2020diva], HDUVA [@sun2021hierarchical], and MatchDG [@mahajan2021domain] among others, were implemented with a transparent underlying neural network architecture. The concrete neural network architecture can thus be replaced by plugging in an architecture implemented in a Python file or by specifying some of the already implemented architectures, such as AlexNet [@krizhevskyImageNetClassificationDeep2012], via command line arguments. The collection of domain generalization algorithms and neural network architectures implemented in DomainLab is steadily increasing over time.

Selection of algorithms, neural network components, as well as other components, such as the training procedure, are done via the chain-of-responsibility method [@gamma1995design]. Other design patterns, including the observer pattern, visitor pattern, etc., are also used to improve the decoupling of different factors contributing to the performance of an algorithm (see also Section "Components" below).

Instead of modifying code across several Python files, the package is closed to modification and open to extension. To simply test an algorithm's performance on a user's data, there is no need to change any code inside this repository. The user only needs to extend this repository to fit their requirements by providing custom python files.
DomainLab offers a framework for generating combinations (tasks, neural network architectures, etc.) by simply letting the user select elements through command line arguments.

With the above design, DomainLab offers users the flexibility to construct custom tasks with their own data, writing custom neural network architectures for use with the already implemented domain generalization algorithms, and even constructing their own domain generalization algorithms on top of the existing components by specifying a Python file with custom models and loss functions. There is no need to change the original code of DomainLab when the user wants to apply a domain generalization method to their own data, extend the method with custom neural networks, or try discriminating the most significant factors that affects performance.

## Components

To achieve the above design goals of decoupling, we used the following components.

*Models* refer to a PyTorch module with a specified loss function containing regularization effects of several domains as well as the task-specific losses (e.g., cross entropy loss for classification tasks). However, the exact neural network architecture can be configured by the user via command line arguments. There are two types of models implemented: (1) models from publications in the field of domain generalization (e.g., [@ilse2020diva], [@sun2021hierarchical], [@mahajan2021domain], etc.), including those using causality and probabilistic model based methods, and (2) custom models, where the user only needs to specify a Python file defining the custom loss function, while remaining flexible with respect to the exact neural network used for each submodule.

The common classification loss calculation is done via a parent model class. Thus, the individual models (e.g., representing different domain regularization) can be reused for other tasks (e.g., segmentation) by simply inheriting another task loss.

*Tasks* refer to a component, where the user specifies different datasets from different domains as well as preprocessing steps. There are several types of tasks in DomainLab:
(1) Built-in tasks, e.g., Color-Mnist, subsampled version of PACS, VLCS, etc., to provide a test utility for algorithms.
(2) "TaskFolder" which can be used if the data is already organized in a root folder with different subfolders containing data from different domains and a further level of sub-sub-folders containing data from different classes.
(3) "TaskPathFile" which allows the user to specify each domain in a text file indicating the path and label for each observation, so that, the user can choose which portion of the sample to use for training, validation and testing.

*Trainer* is the component that directs data flow to the model to calculate the  losses and to update the model parameters. Several models can share a common trainer. A specific trainer can also be a *Visitor* to models to update coefficients in models during training and to implement techniques such as warm-up which follows the visitor design pattern from software engineering.

Following the *Observer* pattern, we use separate classes to conduct operations needed to be done after each epoch (e.g., deciding whether to execute early stopping) and any operations performed after training finishes.

Following the *Builder* pattern, we construct each component needed to conduct a domain generalization experiment, including 
constructing a trainer which guides the data flow, 
constructing a concrete neural network architecture and feeding it into the model, 
constructing the evaluator as a callback of what to do after each epoch.

# Availability
Domainlab is open source and freely available. It is published under the MIT License. You can download the source code at <https://github.com/marrlab/DomainLab>. Extensive documentation can be found here at <https://marrlab.github.io/DomainLab>. DomainLab can be installed using the [python-poetry](https://python-poetry.org/) or [pip](https://pypi.org/project/pip/) utilities.

# Contributions
XS has designed the package with software design patterns and implemented the framework, algorithms and other components of the package. He initiated and made major contributions to other aspects of the package development. AGossmann made significant contributions to the software design and code, testing through use in independent research projects, and writing of the manuscript. CM initiated the project with XS, contributed to the code style enhancement and paper description of Domainlab and supervised the project. PR contributed to the package design, code quality, and paper description. SSB has helped in improvement of the framework, finding new usecases, and enhanced the readability of the code. FB contributed to the paper description and code documentation. GS added the possibility to benchmark different algorithms using a snakemake pipeline and contributed minor enhancements. CF added sanity checks for the datasets and implemented the chart generation for graphical evaluation of the benchmark results. AGruber tested and evaluated the library with real life medical datasets and pointed out important issues and their solutions. RS added a feature to specify command line arguments with YAML files. XZ contributed to the printing and saving of the confusion matrix, the fixes of two bugs joint work with XS and CF, and the implementation of chart generation for the graphical evaluation of the benchmark results.

# Funding
SSB has received funding by F. Hoffmann-la Roche LTD (No grant number is applicable) and supported by the Helmholtz Association under the joint research school ‘Munich School for Data Science - MUDS’. PR was supported through an Alexander von Humboldt Foundation postdoctoral fellowship (Grant Nr. 1221006). CM has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant agreement No. 866411).

# References
