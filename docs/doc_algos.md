# Implemented algorithms (models)

Note this library decomposes the concept of model from network. That is, the model only defines the loss of the neural network, while the architecture of the neural network can be specified independently.

## algorithm "deepall"
Pool all domains together and train an ERM (empirical risk minimization) model


## algorithm "matchdg"
Domain Generalization using Causal Matching, https://arxiv.org/abs/2006.07500, ICML 2020.

## algorithm "diva"
DIVA: Domain Invariant Variational Autoencoders, https://arxiv.org/pdf/1905.10427.pdf, ICLR 2019 workshop.

DIVA approaches the domain generalization problem with a variational autoencoder
with three latent variables, using three independent encoders. By encouraging the network to store the domain,
class and residual features in one of the latent space respectively the class specific information
shall be disentangled. In order to obtain marginally independent latent variables, the density of the domain
and class latent space is conditioned on the domain and class respectively. These densities are parameterized
by learnable parameters. During training, all three latent variables are then fed into a single decoder
reconstructing the image. Additionally, two classifiers are trained, predicting the domain and class label
from the respective latent variable, emphasizing to capture these information in this variables.
This leads to an overall large network. However, during inference only the class encoder and classifier
is used. The experiments are showing that it is indeed possible to disentangle the information in their
test setting. Also in a semi-supervised setting, where class labels are missing for some data or domains,
the classification accuracy can be further improved. This is an advantage in the sense that prediction
accuracy turns out to be notably better if the training contained a domain close to the test domain.
Therefore, this semi-supervised setup allows to prepare for new domains by some additional training
needing only samples of the new domain, but no class labels. In the end however, DIVA always needs domain
labels and does not support a fully unsupervised setting. Since it is not always clear what different domains
actually exist, this can lead to problems and decreased performance.

**Model parameters:**
the following model parameters can be specified: 

- Size of latent space for domain-specific information: zd_dim 
- Size of latent space for residual variance: zx_dim
- Size of latent space for class-specific information: zy_dim
- Multiplier for y classifier ($\alpha_y$ of eq. (2)): gamma_y
- Multiplier for d classifier ($\alpha_d$ of eq. (2)): gamma_d

## algorithm "hduva"
Hierarchical Variational Auto-Encoding for Unsupervised Domain Generalization:https://arxiv.org/pdf/2101.09436.pdf, ICLR 2020 RobustML workshop.

## Others
model 'dann': 'Domain adversarial invariant feature learning'

model 'jigsaw': 'Domain Generalization by Solving Jigsaw Puzzles, https://arxiv.org/abs/1903.06864', CVPR 2019.
