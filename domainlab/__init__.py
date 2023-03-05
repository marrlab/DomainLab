"""
globals for the whole package
"""
__docformat__ = "restructuredtext"
import torch

algos = {
    "deepall": "Pool all domains together and train an ERM (empirical risk minimization) model",
    "diva": "DIVA: Domain Invariant Variational Autoencoders, https://arxiv.org/pdf/1905.10427.pdf",
    "hduva": "Hierarchical Variational Auto-Encoding for Unsupervised Domain Generalization: \
    https://arxiv.org/pdf/2101.09436.pdf",
    "dann": "Domain adversarial invariant feature learning",
    "jigsaw": "Domain Generalization by Solving Jigsaw Puzzles, https://arxiv.org/abs/1903.06864",
    "matchdg": "Domain Generalization using Causal Matching, https://arxiv.org/abs/2006.07500"
}

g_inst_component_loss_agg = torch.mean
