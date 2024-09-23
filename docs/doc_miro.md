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

decoding posterior: $$p(X|Y)\sim Gaussian(Y|[\Sigma^{-1}]_{ij}=\frac{\partial^2 \log p(x|y)}{\partial x_i\partial x_j}$$ when |Y| is large
