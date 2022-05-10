* Submit using the following website: https://joss.theoj.org/papers/new  (login with orcid! Please provide your orcid!)
* https://joss.theoj.org/papers/10.21105/joss.03957
* https://github.com/openjournals/joss-reviews/issues/3574  (look here for an example paper!) 
* https://www.sphinx-doc.org/en/master/
* https://joss.readthedocs.io/en/latest/review_checklist.html (make sure the checklist is met!)
* paper format: 1-2 page Markdown text file with name paper.md
* Make sure each author must commit to the repository! (Please commit paper.md!)


LibDG: PyTorch library for domain generalization
Distribution shifts greatly harm the widespread use of deep learning models for real world applications. Traditional domain adaptation methods map new data at deployment into an invariant distribution domain similar to the distribution domain from the  training data. Domain generalization, however, aims at training models robust against distribution shifts by utilizing data from different domains and producing domain invariant models directly. Over the past years, various methods have been proposed addressing different aspects of the domain generalization problem and remedying shortcomings of previous approaches. The respective papers are usually published with proof-of-concept level code, entangled with data access, pre-processing, evaluation, etc. This leads to two shortcomings: First, proof-of-concept level code greatly restricts AI practitioners from easily applying these methods in their everyday work. Second, it hamperes a systematic benchmarking of methods and thus limits our ability to predict how algorithms will behave on a new set data. 


To address these issues, we introduce LibDG, a pytorch library for domain generalization. LibDG  is designed to achieve maximum decoupling between models, loss functions, neural network architectures, and tasks. It offers a framework for generating combinations by simply letting the user select elements through command line arguments. 


In LibDG,
* models refer to a PyTorch module with a forward pass and a defined loss function, but stay transparent with respect to the exact neural network architecture, which can be configured by the user and be chosen according to the specific dataset bundle which is called task (see task). There are two types of models
   * implemented models from publications in the field of domain generalization
   * custom models, where the user only needs to specify a python file created on his/her own with instruction on how to deal with inflowing data and different domains. 
* tasks refer to a python class where different datasets are composites of the object. There are several types of tasks in LibDG:
   * Build-in tasks like Color-Mnist.
   * TaskFolder: User constructed tasks where the data source folder is specified in a python file. See README.md (**add link!**)
   * TaskPathFile: A user-specified CSV file indicating the path for each file, instead of letting each folder represent one class of files/images as in TaskFolder.
* trainer refers to a python class that directs data flow from a task into the model to update the parameters. 
* evaluator refers to actions that need to be done after each epoch (e.g. deciding whether to execute early stopping) and after training finishes. This is called the observer in software design patterns.
* algorithm builder is responsible for
   * constructing a concrete neural network architecture and feeding into the model.
   * constructing a trainer which guides the data flow.
   * constructing the evaluator as a callback of what to do after each epoch.


With the above facilities, LibDG offers users the freedom to combine their own use case, by constructing custom tasks incorporating their data, writing custom neural network architectures, and even trying their own algorithms by specifying a python file. 


[References]
* DIVA: Domain Invariant Variational Autoencoders, https://arxiv.org/pdf/1905.10427.pdf
* Hierarchical Variational Auto-Encoding for Unsupervised Domain Generalization:https://arxiv.org/pdf/2101.09436.pdf'
* Domain Generalization using Causal Matching, https://arxiv.org/abs/2006.07500',
* Domain adversarial invariant feature learning
* Domain Generalization by Solving Jigsaw Puzzles, https://arxiv.org/abs/1903.06864