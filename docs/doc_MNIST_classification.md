# colored MNIST classification

We include in the DomainLab package colored verion of MNIST where the color correspond to domain and digit correspond to semantic concept that we want to classify. 

## colored MNIST dataset
We provide 10 different colored version of MNIST as 10 different domains which which number 0 to 9. The digit and background are colored differently, thus a domain correspond to a 2-color combination. 
An extraction of digit 0 to 9 from domain 0 is shown in Figure 1. 

<div style="align: center; text-align:center;">
digits 0 - 9: <img src="figs/colored_MNIST/singels/digit0.jpg" style="width:30px;"/>
<img src="figs/colored_MNIST/singels/digit1.jpg" style="width:30px;"/>
<img src="figs/colored_MNIST/singels/digit2.jpg" style="width:30px;"/>
<img src="figs/colored_MNIST/singels/digit3.jpg" style="width:30px;"/>
<img src="figs/colored_MNIST/singels/digit4.jpg" style="width:30px;"/>
<img src="figs/colored_MNIST/singels/digit5.jpg" style="width:30px;"/>
<img src="figs/colored_MNIST/singels/digit6.jpg" style="width:30px;"/>
<img src="figs/colored_MNIST/singels/digit7.jpg" style="width:30px;"/>
<img src="figs/colored_MNIST/singels/digit8.jpg" style="width:30px;"/>
<img src="figs/colored_MNIST/singels/digit9.jpg" style="width:30px;"/>
 <div class="caption">Figure 1: digits zero to nine from domain zero of colored MNIST </div>
</div>

<br/>
The available digit-background color combinations together with the domain number are shown in Figure 2 where only different digit 0s are listed. 
Note that domain 0 and domain 3, for the same color, in one domain it appears as foreground and another as background color.  
<br/>

<div style="align: center; text-align:center;">
 domain 0: <img src="figs/colored_MNIST/0/digit0.jpg" style="width:300px;"/><br/>
 domain 1: <img src="figs/colored_MNIST/1/digit0.jpg" style="width:300px;"/><br/>
 domain 2: <img src="figs/colored_MNIST/2/digit0.jpg" style="width:300px;"/><br/>
 domain 3: <img src="figs/colored_MNIST/3/digit0.jpg" style="width:300px;"/><br/>
 domain 4: <img src="figs/colored_MNIST/4/digit0.jpg" style="width:300px;"/><br/>
 domain 5: <img src="figs/colored_MNIST/5/digit0.jpg" style="width:300px;"/><br/>
 domain 6: <img src="figs/colored_MNIST/6/digit0.jpg" style="width:300px;"/><br/>
 domain 7: <img src="figs/colored_MNIST/7/digit0.jpg" style="width:300px;"/><br/>
 domain 8: <img src="figs/colored_MNIST/8/digit0.jpg" style="width:300px;"/><br/>
 domain 9: <img src="figs/colored_MNIST/9/digit0.jpg" style="width:300px;"/>
 <div class="caption">Figure 2: digit zero from domain zero to nine of colored MNIST </div>
</div>
<br/>

## domain generalisation on colored MNIST

A particular hard task for domain generalization is, if only a few training domains are available and the test domain differs a lot from the train domains. Here we use domain 0 and 3 from Figure 2, for testing domain we choose domain 1 and 2 as the colors appearing here are far different from the ones used in training. 

For our test we like to compare diva and deepall, this was done using the following command lines:

### deepall (Emperical Risk Minimization)

```shell
python main_out.py --te_d 1 2 --tr_d 0 3 --task=mnistcolor10 --epos=500 --bs=16 --aname=deepall --nname=conv_bn_pool_2 --san_check --san_num=8 --lr=1e-3 --seed=0
```

### diva 

```shell
python main_out.py --te_d 1 2 --tr_d 0 3 --task=mnistcolor10 --epos=500 --bs=16 --aname=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=1e5 --gamma_d=1e5 --lr=1e-3 --seed=0
```

**Notes**
- `--san_check` and `--san_num=8` are only used to generate the dataset extractions we plotted in figure 1 and 2 to check if the datasets we used for training are correct
- `--epos` was set high enough to end the training using the early stopping criterion.


### Results

For both algorithms the early stop criterion ended the training. Although diva is a more complex method which needs more time for one epoch of training, the total training time of dive was much lower than deepall, due to the fewer epochs. The performance of the trained models on the test domains are summarized in the following table:

| method    | epochs | acc       | precision | recall     | specificity | f1          | auroc     |
| -         | -      | -         | -         | -          | -           | -           | -         |
| deepall   | 9      | 0.798     | 0.858     | 0.800      | 0.978       | 0.797       | 0.832     |
| diva      | 16     | 0.959     | 0.961     | 0.958      | 0.995       | 0.958       | 0.999     |
