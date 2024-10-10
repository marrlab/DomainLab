# Model DIVA
## Domain Invariant Variational Autoencoders

DIVA addresses the domain generalization problem with a variational autoencoder
with three latent variables, using three independent encoders.

By encouraging the network to store each the domain,
class and residual features in one of the latent spaces, the class-specific information
is disentangled.

In order to obtain marginally independent latent variables, the densities of the domain
and class latent spaces are conditioned on the domain and the class, respectively. These densities are then
parameterized by learnable parameters. During training, all three latent variables are fed into a single decoder
reconstructing the input image.

Additionally, two classifiers are trained, predicting the domain and class label
from the respective latent variable.
This leads to an overall large network. However, during inference only the class encoder and classifier
are used.

DIVA can improve the classification accuracy also in a semi-supervised setting, where class labels
are missing for some data or domains. This is an advantage, as prediction
accuracy turns out to be notably better if the training data contains a domain close to the test domain.
Therefore, this semi-supervised setup allows to prepare for new domains by some additional training,
needing only samples of the new domain, but no class labels.
However, DIVA always needs domain labels and does not support a fully unsupervised setting.
Since it is not always clear which different domains actually exist, this can lead to problems and a
decreased performance.

### Model parameters
The following hyperparameters can be specified:

- `zd_dim`: size of latent space for domain-specific information
- `zx_dim`: size of latent space for residual variance
- `zy_dim`: size of latent space for class-specific information
- `gamma_y`: multiplier for y classifier ($\alpha_y$ of eq. (2) in paper below)
- `gamma_d`: multiplier for d classifier ($\alpha_d$ of eq. (2) in paper below)

Furthermore, the user can specify the neural networks for the class and domain classifiers using
- `nname`/`npath`
- `nname_dom`/`npath_dom`


## Examples
### model diva with implemented neural network
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --model=diva --nname=alexnet --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5
```

### model diva with custom neural network
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --model=diva --npath=examples/nets/resnet.py --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5
```
### generation of images
```shell
python main_out.py --te_d=0 --task=mnistcolor10 --keep_model --model=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=10e5 --gamma_d=1e5 --gen
```
## Colored version of MNIST

### leave one domain out
```shell
python main_out.py --te_d=0 --task=mnistcolor10 --keep_model --model=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=10e5 --gamma_d=1e5
```

### choose train and test
```shell
python main_out.py --te_d 0 1 2 --tr_d 3 7 --task=mnistcolor10 --model=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=7e5 --gamma_d=1e5
```




_Reference:_
DIVA: Domain Invariant Variational Autoencoders, https://arxiv.org/pdf/1905.10427.pdf, Medical Imaging with Deep Learning. PMLR, 2020.
