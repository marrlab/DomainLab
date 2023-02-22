# JiGen

The JiGen method proposed in [https://arxiv.org/abs/2007.01434](https://arxiv.org/abs/1903.06864) by Carlucci et al. extends the understanding of the concept of spatial correlation in the neuronal network by training the network not only on a classification task, but also on solving jigsaw puzzles. 

For the jigsaw puzzles the images are split into $n \times n$ patches and permutated. 

The model is to predict the permutation index which result in the permutated image. 

To be able to solve the classification problem and the jigsaw puzzle in parallel, shuffled and ordered images will first be fed into a convolutional network for feature extraction and will then be given to the image classifier and the jigsaw classifier.

For the training of classificaiton task the network uses a cross-entropy loss for both tasks while weighting the jigsaw classifier by a hyper parameter. Additionally the relative ratio or probability of shuffling the tiles of one instance from training data set is given by another hyper parameter. 

The advantage of this method is that is does not require domain label, as the jigsaw puzzle can be solved despite the missing domain labels. 

