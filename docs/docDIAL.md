# DIAL: Domain Invariant Adversarial Learning

The algorithm introduced in https://arxiv.org/pdf/2104.00322.pdf uses adversarial learning to tackle the task of domain generalization. Therefore, the source domain is the natural dataset, while the target domain is generated using adversarial attack on the source domain.

## generating the adversarial domain

For creating the adversarial domain one aims to find an adversarial image $x'$ to the natural image $x'$ with $||x- x'||$ small, such that the output of a classification network $\phi$ fulfills $||\phi(x) - \phi(x')||$ big. Domainlab archives this goal by starting with $x'_0 = x + \sigma \tilde{x}~$, $\tilde{x} \sim \mathcal{N}(0, 1)$ and using $n$ steps in a gradient descend with step size $\tau$ to maximize $||\phi(x) - \phi(x')||$.

## network structure

The network consists of three parts. First of all a feature extractor extracts the main characteristics of the images. This features are then used as the input to a label classifier and a domain classifier. 
During training the network is optimized to a have low error on the classification task, while ensuring that the internal representation (output of the feature extractor) cannot discriminate between the natural and adversarial domain. This goal can be archived by using a special loss function.


<img src="figs/DIAL_netw.png" width="450"> 

Fig: network structure (https://arxiv.org/pdf/2104.00322.pdf)

## loss function

The loss function of the algorithm is a combination of three terms:

1. standard cross entropy loss between the predicted label probabilities and the actual label ($CE_{nat}$ for the natural domain, $CE_{adv}$ for the adversarial domain)
2. Kullback-Leibler divergence between classifier output on the natural images and their adversarial counterparts ($KL$)
3. standard cross-entropy loss between predicted domain probability and domain label ($D_{nat}$ for the natural domain, $D_{adv}$ for the adversarial domain)

The loss functions are given by:
$$
DIAL_{CE} = CE_{nat} + \lambda ~ CE_{adv} - r(D_{nat} + D_{adv}) \\
DIAL_{KL} = CE_{nat} + \lambda ~ KL - r(D_{nat} + D_{adv})
$$
As the task is to minimize the label classification loss and maximize classification loss for the the adversarial domain, a gradient reversal layer is inserted into the network, right in front of the domain classifier. This layer leaves the input unchanged during forward propagation and reverses the gradient by multiplying it with a negative scalar during the back-propagation. This parameter is initialized to a small value and is gradually increased to $r$.


---

This procedure yields to the following availability of hyperparameter:
- `--dial_steps_perturb`: how many gradient step to go to find an adversarial image ($n$ from "*generating the adversarial domain*" above)
- `--dial_noise_scale`: variance of gaussian noise to inject on pure image ($\sigma$ from "*generating the adversarial domain*" above)
- `--dial_lr`: learning rate to generate adversarial images ($\tau$ in the paper)
- `--dial_epsilon`: pixel wise threshold to perturb images
- `--gamma_reg`: ? ($\epsilon$ in the paper)
- `--lr`: learning rate ($\alpha$ in the paper)
