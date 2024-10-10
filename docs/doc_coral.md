# Deep CORAL
## Deep CORAL: Correlation Alignment for Deep Domain Adaptation

nonlinear transformation that aligns correlations of
layer activations in deep neural network
https://arxiv.org/pdf/1607.01719

## Examples

```
python main_out.py --te_d 0 --tr_d 3 7 --bs=32 --epos=1 --task=mnistcolor10 --model=erm --nname=conv_bn_pool_2 --trainer=coral
```
