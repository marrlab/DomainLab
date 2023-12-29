# Model HDUVA
## HDUVA: HIERARCHICAL VARIATIONAL AUTO-ENCODING FOR UNSUPERVISED DOMAIN GENERALIZATION

HDUVA builds on a generative approach within the framework of variational autoencoders to facilitate generalization to new domains without supervision. HDUVA learns representations that disentangle domain-specific information from class-label specific information even in complex settings where domain structure is not observed during training. 

## Model Overview
More specifically, HDUVA is based on three latent variables that are used to model distinct sources of variation and are denoted as $z_y$, $z_d$ and $z_x$. $z_y$ represents class specific information, $z_d$ represents domain specific information and $z_x$ models residual variance of the input. We introduce an additional hierarchical level and use a continuous latent representation s to model (potentially unobserved) domain structure. This means that we can encourage disentanglement of the latent variables through conditional priors without the need of conditioning on a one-hot-encoded, observed domain label. The model along with its parameters and hyperparameters is shown in Figure 1: 

<div style="align: center; text-align:center;">
 <img src="figs/tikz_hduva.svg" alt="PGM for HDUVA" style="height: 300px; width:500px;"/> 
 <div class="caption">Figure 1: Probabilistic graphical model for HDUVA:Hierarchical Domain Unsupervised Variational Autoencoding. </div>
</div>



Note that as part of the model a latent representation of $X$ is concatentated with $s$ and $z_d$ (dashed arrows), requiring respecive encoder networks.

## Evidence lower bound and overall loss
The ELBO of the model can be decomposed into 4 different terms: 

Likelihood: $E_{q(z_d, s|x), q(z_x|x), q(z_y|x)}\log p_{\theta}(x|s, z_d, z_x, z_y)$ 

KL divergence weighted as in the Beta-VAE: $-\beta_x KL(q_{\phi_x}(z_x|x)||p_{\theta_x}(z_x)) - \beta_y KL(q_{\phi_y}(z_y|x)||p_{\theta_y}(z_y|y))$ 

Hierarchical KL loss (domain term): $- \beta_d E_{q_{\phi_s}(s|x), q_{\phi_d}(z_d|x, s)} \log \frac{q_{\phi_d}(z_d|x, s)}{p_{\theta_d}(z_d|s)}$

Hierarchical KL loss  (topic term): $-\beta_t E_{q_{\phi_s}(s|x)}KL(q_{\phi_s}(s|x)||p_{\theta_s}(s|\alpha))$

In addition, we construct the overall loss by adding an auxiliary classsifier, by adding an additional term to the ELBO loss, weighted with $\gamma_y$:


## Hyperparameters loss function
For fitting the model, we need to specify the 4 $\beta$-weights related to the the different terms of the ELBO ( $\beta_x$ , $\beta_y$, $\beta_d$, $\beta_t$)  as well as $\gamma_y$. 

## Model hyperparameters
In addition to these hyperparameters, the following model parameters can be specified: 

-   `zd_dim`: size of latent space for domain-specific information
-   `zx_dim`: size of latent space for residual variance
-   `zy_dim`: size of latent space for class-specific information
-   `topic_dim`: size of dirichlet distribution for topics $s$

The user need to specify at least two neural networks for the **encoder** part via 

- `npath_encoder_x2topic_h`:  the python file path of a neural network that maps the image (or other
modal of data to a one dimensional (`topic_dim`) hidden representation serving as input to Dirichlet encoder: `X->h_t(X)->alpha(h_t(X))` where `alpha` is the neural network to map a 1-d hidden layer to dirichlet concentration parameter.
- `npath_encoder_sandwich_x2h4zd`: the python file path of a neural network that maps the
image to a hidden representation (`img_h_dim`), which will be used to infere the posterior distribution of `z_d`: `topic(X), X -> [topic(X), h_d(X)] -> zd_mean, zd_scale`

For the path: `topic(X), X -> [topic(X), h_d(X)] -> zd_mean, zd_scale`, we need an extra output dimension for neural network `h_d`:
-   `img_h_dim`: output size of `h_d(X)`

Alternatively, one could use an existing neural network in DomainLab using `nname` instead of `npath`:
-   `nname_encoder_x2topic_h`
-   `nname_encoder_sandwich_x2h4zd`


## Hyperparameter for warmup
Finally, the number of epochs for hyper-parameter warm-up can be specified via the argument `warmup`.

Please cite our paper if you find it useful!
```text
@inproceedings{sun2021hierarchical,
  title={Hierarchical Variational Auto-Encoding for Unsupervised Domain Generalization},
  author={Sun, Xudong and Buettner, Florian},
  booktitle={ICLR 2021 RobustML workshop, https://arxiv.org/pdf/2101.09436.pdf},
  year={2021}
}
```
