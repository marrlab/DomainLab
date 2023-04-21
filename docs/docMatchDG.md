# MatchDG: Domain Generalization using Causal Matching

This algorithm introduced in https://arxiv.org/pdf/2006.07500.pdf is motivated by causality theory. The authors try to enforce, that a model does classify an image only on the object information included in the image and not on the domain information.

## Motivation: causality theory

The autors of the paper motivate their approach by looking at the data-generation process


## Training

This is done by using two phases during training. 

1. Phase: Training a network $\phi$ on a matching task using a contrastive loss with hyperparameter $\tau$ (`--tau`) to minimize distance between same-class inputs from different domains in comparison to inputs from different classes across domains. After some epoch (`--epos_per_match_update`) the match tensor is updated using the current versions of $\phi$.

$\Rightarrow$ A trained network $\phi^*$. For two images from the same objects in different domains, the distance between the two outputs is small, while for two different objects the distance is large. The trained network is later used to compute matches.

2. Phase: Training a classification network $f = h(\phi(x))$ reusing the same network structure $\phi$ from phase 1. The loss function consists of two parts, one minimizing the classification error and one acting as a regularization. The regularization minimizes the distance of the outputs of $\phi$ for two images which are matches according to $\phi^*$, but arise from different domains. The network is trained from scratch, the trained network $\phi^*$ is only used to compute the matches used in the regularization. 

$\Rightarrow$ Classification network $f$.

---

This procedure yields to the following availability of hyperparameter:
- `--tau`: hyperparameter in the contrastive loss. In the paper this parameter is chosen to be $0.05$ in all experiments. ($\tau$ from above)
- `--epochs_ctr`: number of epochs for minimizing the contrastive loss in phase 1.
- `--epos_per_match_update`: Number of epochs before updating the match tensor.
- `--gamma_reg`: weight for the regularization term in phase 2.
