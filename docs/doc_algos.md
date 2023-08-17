# Overview of implemented models and algorithms

Note DomainLab decomposes the concept of model from network and algorithm. That is, the model only defines the loss of the neural network, while the architecture of the neural network can be specified independently as a component of the model, and algorithm is model with a specified trainer and observer. Algorithm share the same name as model with a default trainer specified.

## algorithm "deepall"
Pool all domains together and train an ERM (empirical risk minimization) model

## algorithm "matchdg"
Mahajan, Divyat, Shruti Tople, and Amit Sharma. "Domain generalization using causal matching." International Conference on Machine Learning. PMLR, 2021.

## algorithm "diva"
Ilse, Maximilian, et al. "Diva: Domain invariant variational autoencoders." Medical Imaging with Deep Learning. PMLR, 2020.

## algorithm "hduva"
Hierarchical Variational Auto-Encoding for Unsupervised Domain Generalization: https://arxiv.org/pdf/2101.09436.pdf, ICLR 2020 RobustML.

## algorithm "dann": 
Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks." The journal of machine learning research 17.1 (2016): 2096-2030.

## algorithm "jigen": 
Carlucci, Fabio M., et al. "Domain generalization by solving jigsaw puzzles." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019. https://arxiv.org/abs/1903.06864

## algorithm deepall, dann, diva with trainer "dial"
Levi, Matan, Idan Attias, and Aryeh Kontorovich. "Domain invariant adversarial learning.", Transactions on Machine Learning Research, arXiv preprint arXiv:2104.00322 (2021).

## algorithm deepall with trainer "mldg"
Li, Da, et al. "Learning to generalize: Meta-learning for domain generalization." Proceedings of the AAAI conference on artificial intelligence. Vol. 32. No. 1. 2018.

## New algo
Carla und Lisa und Xudong