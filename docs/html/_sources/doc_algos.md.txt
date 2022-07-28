# Implemented algorithms (models)
Note this library decomposes the concept of model from network. That is, the model only defines the loss of the neural network, while the architecture of the neural network can be specified independently.

## algorithm "deepall"
Pool all domains together and train an ERM (empirical risk minimization) model


## algorithm "matchdg"
Domain Generalization using Causal Matching, https://arxiv.org/abs/2006.07500, ICML 2020.

## algorithm "diva"
DIVA: Domain Invariant Variational Autoencoders, https://arxiv.org/pdf/1905.10427.pdf, ICLR 2019 workshop.

## algorithm "hduva"
Hierarchical Variational Auto-Encoding for Unsupervised Domain Generalization:https://arxiv.org/pdf/2101.09436.pdf, ICLR 2020 RobustML workshop.

## Others
model 'dann': 'Domain adversarial invariant feature learning'

model 'jigsaw': 'Domain Generalization by Solving Jigsaw Puzzles, https://arxiv.org/abs/1903.06864', CVPR 2019.
