# MIRO: Mutual-Information Regularization
## Mutual Information Regularization with Oracle (MIRO).

### Pre-requisite: Variational lower bound on mutual information

Barber, David, and Felix Agakov. "The im algorithm: a variational approach to information maximization." Advances in neural information processing systems 16, no. 320 (2004): 201.

$$I(X,Y)=H(Y)-H(Y|X)=-<\log p_y(Y)>_{p_y(Y)}+<\log p(Y|X)>_{p(X,Y)}$$

Given variational distribution of $q(x|y)$ as decoder (i.e. $Y$ encodes information from $X$)


Since

$$KL\left(p(X|Y)|q(X|Y)\right)=<\log p(X|Y)>_{p(X|Y)}-<\log q(X,Y)>_{p(X|Y)} >0$$

We have

$$<\log p(X|Y)>_{p(X|Y)}><\log q(X,Y)>_{p(X|Y)}$$

Then

$$I(X,Y)=-<\log p_y(Y)>_{p_y(Y)}+<\log p(Y|X)>_{p(X,Y)}>-<\log p_y(Y)>_{p_y(Y)}+<\log q(X,Y)>_{p(X|Y)}$$

with the lower bound being

$$-<\log p_y(Y)>_{p_y(Y)}+<\log q(X,Y)>_{p(X|Y)}$$

To optimize the lower bound, one can iterate

- fix decoder $q(X|Y)$ and optimize encoder $Y=g(X;\theta) + \epsilon$
- fix encoder parameter $\theta$, tune decoder to alleviate the lower bound

#### Laplace approximation

decoding posterior: 

$$p(X|Y) \sim Gaussian(Y|[\Sigma^{-1}]_{ij}=\frac{\partial^2 \log p(x|y)}{\partial x_i\partial x_j})$$ 

when $|Y|$ is large (large deviation from zero contains more information, which must be explained by non-typical $X$)

#### Linear Gaussian


The bound $H(X)+<\log q(X|Y)>_{p(x,y)}$ becomes

$$\sum_i <|X_i-m(Y_i)|_{|\Sigma^{-1}|(Y_i)} + \log det(\Sigma(Y_i))>_{p(Y|X)}$$


## MIRO

MIRO try to match the pre-trained model's features layer by layer to the target neural network we want to train for domain invariance in terms of mutual information. They use a constant identity encoder on feature from target neural network, then a population variance $\Sigma$ (forced to be diagonal). 

Let $z$ denote the intermediate features of each layer, let $f_0$ be the pre-trained model, $f$ be the target neural network. Let $x$ be the input data.

$$z_f=f(x)$$

$$z_{f_0}=f^{(0)}(x)$$

the lower bound  for Mutual information for instance $i$ is


$$\log|\Sigma| + ||z^{(i)}_{f_0}-id(z^{i})||_{\Sigma}^{-1}$$

where $id$ is the mean map

For diagonal $\Sigma$, determinant is simply multiplication of all diagonal values,

$$\log|\Sigma|=\sum_{k} \log \sigma_k + ||{z_k}^{(i)}_{f_0}-z_k^{i}||{\sigma_k}^{-1}$$

