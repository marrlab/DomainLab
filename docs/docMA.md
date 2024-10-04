# Simple Moving Average
For each epoch, convex combine the weights for each layey from Paper: Ensemble of Averages: Improving Model Selection and
Boosting Performance in Domain Generalization, Devansh Arpit, Huan Wang, Yingbo Zhou, Caiming Xiong, Salesforce Research, USA

Example:
```
python main_out.py --te_d 0 1 --tr_d 3 5 --task=mnistcolor10 --epos=2 --bs=2 --model=erm --nname=conv_bn_pool_2 --trainer=ma
``` 
