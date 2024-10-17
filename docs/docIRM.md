# Trainer IRM 
## Invariant Risk Minimization

Decompose a classification task into feature extraction $\Phi(\cdot)$ and classificaiton layer $w(\cdot)$, then the task loss is

$\ell^{(d)} (w \circ \Phi) = \mathbb{E}_{(X, Y) \sim \mathcal{D}_d}[\ell(w \circ \Phi(X), Y)]$
where we use $\ell$ to denote the cross entropy for a classification task, and $\mathcal{D}_d$ for distribution of domain $d$.

The idea of IRM is to choose classifier $w$ to be in the intersection of optimal classifiers for each domain $$d$$.

$$w \in {\arg\min}_{\bar{w}} \ell^{(d)}(\bar{w} \circ \Phi) \quad \forall d$$

regardless of feature extractor $\Phi(\cdot)$, 
this serves as a constraint on the choice of classifiers $w$.

The feature extractor $\Phi(\cdot)$ then get optimized under this constraint.

Thus IRM forms a bi-level optimization by jointly optimize $\Phi$ and $w$ which is hard to solve, so in practice IRMv1 is used. 

## IRMv1

In DomainLab, we write the loss function as $$\ell(\cdot) + \lambda R(\cdot)$$, which result in the optmization below:

$$\min_{\Phi, w} \sum_{d} \ell^{(d)}(w \circ \Phi) + \lambda \sum_{d} \|\nabla_{w|w=1.0} \ell^{(d)}(w \circ \Phi)\|^2$$

where $\lambda$ is a hyperparameter that controls the trade-off between the empirical risk and the penalty. One interpretation can be the penalty encourages the representation $\Phi$ to be orthogonal to the gradient of the loss (e.g. cross entropy) at $w = 1.0$ across all domains.

In practice, one could simply divide one mini-batch into two subsets, let $i$ and $j$ to index these two subsets, multiply  subset $i$ and subset $j$ forms an unbiased estimation of the L2 norm of gradient.
In detail: the squared gradient norm via inner product between $\nabla_{w|w=1} \ell(w \circ \Phi(X^{(d, i)}), Y^{(d, i)})$ of dimension dim(Grad) with $\nabla_{w|w=1} \ell(w \circ \Phi(X^{(d, j)}), Y^{(d, j)})$ of dimension dim(Grad) For more details, see section 3.2 and Appendix D of : Arjovsky et al., “Invariant Risk Minimization.”

## Examples
```shell
python main_out.py --te_d=0 --task=mnistcolor10 --model=erm --trainer=irm --nname=conv_bn_pool_2

```
