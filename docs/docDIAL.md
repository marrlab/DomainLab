# DIAL: Domain Invariant Adversarial Learning

The algorithm introduced in https://arxiv.org/pdf/2104.00322.pdf uses adversarial learning to tackle the task of domain generalization. Therefore, the source domain is the natural dataset, while the target domain is generated using adversarial attack on the source domain. 

The network consists of three parts, a feature extractor which produces the input used by a label classifier and a domain classifier. 
During training the network is optimized to have low error on the classification task, while ensuring that the internal representation (output of the feature extractor) cannot discriminate between the natural and adversarial domain. 

### loss function

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
- `--dial_steps_perturb`: how many gradient step to go to find an image as adversarials
- `--dial_noise_scale`: variance of gaussian noise to inject on pure image
- `--dial_lr`: learning rate to generate adversarial images
- `--dial_epsilon`: pixel wise threshold to perturb images
- `--gamma_reg`: ?
- `--lr`: learning rate
