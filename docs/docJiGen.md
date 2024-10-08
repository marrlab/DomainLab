# Model JiGen

<div style="align: center; text-align:center;">
<figure>  
<img src="https://github.com/marrlab/DomainLab/blob/master/docs/figs/jigen.png?raw=true" style="width:300px;"/> 
</figure>
</div>


The JiGen method extends the understanding of the concept of spatial correlation in the
neural network by training the network not only on a classification task, but also on solving jigsaw puzzles.

To create a jigsaw puzzle, an image is split into $n \times n$ patches, which are then permuted.
The goal is training the model to predict the correct permutation, which results in the permuted image.

To solve the classification problem and the jigsaw puzzle in parallel, the permuted and
the original images are first fed into a convolutional network for feature extraction and then given
to two classifiers, one being the image classifier and the other the jigsaw classifier.

For the training of both classification networks, a cross-entropy loss is used. The total loss is then
given by the loss of the image classification task plus the loss of the jigsaw task, whereby the
jigsaw loss is weighted by a hyperparameter.
Another hyperparameter denotes the probability of shuffling the patches of one instance from the training
data set, i.e. the relative ratio.

The advantage of this method is that it does not require domain labels, as the jigsaw puzzle can be
solved despite missing domain labels.

### Model parameters
The following hyperparameters can be specified:
- `nperm`: number of patches in a permutation
- `pperm`: relative ratio, as explained above
- `gamma_reg`: weighting term for the jigsaw classifier

Furthermore, the user can specify a custom grid length via `grid_len`.

_Reference_: Carlucci, Fabio M., et al. "Domain generalization by solving jigsaw puzzles."
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.


## Examples

### model jigen with implemented neural network
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --model=jigen --nname=alexnet --pperm=1 --nperm=100 --grid_len=3
```


### sannity check with jigen tile shuffling
```shell
python main_out.py --te_d=sketch --tpath=examples/tasks/demo_task_path_list_small.py --debug --bs=8 --model=jigen --nname=alexnet --pperm=1 --nperm=100 --grid_len=3 --san_check
```
