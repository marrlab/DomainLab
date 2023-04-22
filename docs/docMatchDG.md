# MatchDG: Domain Generalization using Causal Matching

This algorithm introduced in https://arxiv.org/pdf/2006.07500.pdf is motivated by causality theory. The authors try to enforce, that a model does classify an image only on the object information included in the image and not on the domain information.

## Motivation: causality theory

The authors of the paper motivate their approach by looking at the data-generation process. The underlying causal model (SCM) is given in figure 1. In the graphic one starts of from the object $O$ and the domain D. The object is directly influenced by its true label $y_\text{true}$ while label and domain do only correlate with each other. Additionally the object is correlated with the domain conditioned on $y_\text{true}$. The information from the object $O$ and the domain $D$ do together form the image $x$ which shall be classified by the neuronal network. Doing so, the object does contribute to the image by providing high-level causal features $x_C$ that are common to any image of the same object. This features are the key for classifying the object, as there is only subliminal influence of the domain, therefore the prediction $y$ is only depending on this causal features. The second contribution to the image are domain-dependent high-level features of the object $x_A$, which depend on both, the object $O$ and the domain $D$. This domain-dependent features shall not be respected in the classification as there is a high influence of the domain.


<div style="align: center; text-align:center;">
 <img src="figs/matchDG_causality.png" alt="PGM for HDUVA" style="width:200px;"/> 
 <div class="caption">Figure 1: Structural causal model for the data-generating process. Observed variables are shaded; dashed arrows denote correlated nodes. Object may not be observed. (Image source: Figure 2 of Domain Generalization using Causal Matching https://arxiv.org/pdf/2006.07500.pdf) </div>
</div>


## Network

Before defining the network, one needs to define three sets: 
- $\mathcal{X}$: image space with $x \in \mathcal{X}$ 
- $\mathcal{C}$: causal feature space with $x_C \in \mathcal{C}$
- $\mathcal{Y}$: label space with $y \in \mathcal{Y}$ 

For the classification the goal is to classify an object only based on its causal features $x_C$, hence we define a network $h: \mathcal{C} \rightarrow \mathcal{Y}$. Since $x_C$ for an image $x$ is unknown, one needs to learn a representation function $\phi: \mathcal{X} \rightarrow \mathcal{C}$. Together these network form the desired classifier $f = h(\phi(x)) : \mathcal{X} \rightarrow \mathcal{Y}$.

 




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
