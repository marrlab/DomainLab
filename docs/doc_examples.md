# Command line examples

## Colored version of MNIST

### leave one domain out
```shell
python main_out.py --te_d=0 --task=mnistcolor10 --keep_model --model=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=10e5 --gamma_d=1e5
```

### choose train and test
```shell
python main_out.py --te_d 0 1 2 --tr_d 3 7 --task=mnistcolor10 --model=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=7e5 --gamma_d=1e5
```

### make a sanity check for the dataset using 8 instances from each domain and from each class
```shell
python main_out.py --te_d=0 --task=mini_vlcs --debug --bs=2 --model=diva --nname=alexnet --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5 --san_check --san_num=4
```
### sanity check on only 2 train domains and 2 test domain2
```shell
python main_out.py --te_d 0 1 --tr_d 3 5 --task=mnistcolor10 --debug --bs=2 --model=erm --nname=conv_bn_pool_2 --san_check --san_num=4
```

### generation of images
```shell
python main_out.py --te_d=0 --task=mnistcolor10 --keep_model --model=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=10e5 --gamma_d=1e5 --gen
```

### use hduva on color mnist, train on 2 domains
```shell
python main_out.py --tr_d 0 1 2 --te_d 3 --bs=2 --task=mnistcolor10 --model=hduva  --nname=conv_bn_pool_2 --gamma_y=7e5 --nname_encoder_x2topic_h=conv_bn_pool_2 --nname_encoder_sandwich_x2h4zd=conv_bn_pool_2
```

### hduva is domain-unsupervised, so it works also with a single domain
```shell
python main_out.py --tr_d 0  --te_d 3 4 --bs=2 --task=mnistcolor10 --model=hduva --nname=conv_bn_pool_2 --gamma_y=7e5 --nname_encoder_x2topic_h=conv_bn_pool_2 --nname_encoder_sandwich_x2h4zd=conv_bn_pool_2
```


## Larger images:

### model diva with implemented neural network
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --model=diva --nname=alexnet --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5
```

### model diva with custom neural network
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --model=diva --npath=examples/nets/resnet.py --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5
```

### model erm with implemented neural network
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --model=erm --nname=alexnet
```

### model dann with implemented neural network
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --model=dann --nname=alexnet
```

### model jigen with implemented neural network
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --model=jigen --nname=alexnet --pperm=1 --nperm=100 --grid_len=3
```


### sannity check with jigen tile shuffling
```shell
python main_out.py --te_d=sketch --tpath=examples/tasks/demo_task_path_list_small.py --debug --bs=8 --model=jigen --nname=alexnet --pperm=1 --nperm=100 --grid_len=3 --san_check
```

### training implemented neural network with matchdg
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --model=erm --trainer=matchdg --epochs_ctr=3 --epos=6 --nname=alexnet
```

### trainer matchdg with mnist
```shell
 python main_out.py --te_d 0 1 2 --tr_d 3 7 --task=mnistcolor10 --model=erm --trainer=matchdg --nname=conv_bn_pool_2 --epochs_ctr=2 --epos=6
```

### hduva with implemented neural network
```shell
python main_out.py --te_d=caltech --bs=2 --task=mini_vlcs --model=hduva --nname=conv_bn_pool_2 --gamma_y=7e5 --nname_encoder_x2topic_h=conv_bn_pool_2 --nname_encoder_sandwich_x2h4zd=conv_bn_pool_2
```

### hduva use alex net
```shell
python main_out.py --te_d=caltech --bs=2 --task=mini_vlcs --model=hduva --nname=conv_bn_pool_2 --gamma_y=7e5 --nname_encoder_x2topic_h=conv_bn_pool_2 --nname_encoder_sandwich_x2h4zd=alexnet
```


## Custom Neural Network

### model erm with custom neural network
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --model=erm --npath=examples/nets/resnet.py
```

### trainer matchdg with custom neural network
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --bs=2 --model=erm --trainer=matchdg --epochs_ctr=3 --epos=6 --npath=examples/nets/resnet.py
```


### training hduva with matchdg

```shell
python main_out.py --te_d 0 1 2 --tr_d 3 7 --task=mnistcolor10 --bs=2 --model=hduva --trainer=matchdg --epochs_ctr=3 --epos=6 --nname=conv_bn_pool_2 --gamma_y=7e5 --nname_encoder_x2topic_h=conv_bn_pool_2 --nname_encoder_sandwich_x2h4zd=conv_bn_pool_2
```

### hduva use custom net for sandwich encoder
```shell
python main_out.py --te_d=caltech --bs=2 --task=mini_vlcs --model=hduva --nname=conv_bn_pool_2 --gamma_y=7e5 --nname_encoder_x2topic_h=conv_bn_pool_2 --npath_encoder_sandwich_x2h4zd=examples/nets/resnet.py
```

### hduva use custom net for topic encoder
```shell
python main_out.py --te_d=caltech --bs=2 --task=mini_vlcs --model=hduva --nname=conv_bn_pool_2 --gamma_y=7e5 --npath_encoder_x2topic_h=examples/nets/resnet.py --nname_encoder_sandwich_x2h4zd=conv_bn_pool_2
```

### hduva use custom net for classification encoder
```shell
python main_out.py --te_d=caltech --bs=2 --task=mini_vlcs --model=hduva --npath=examples/nets/resnet.py --gamma_y=7e5 --nname_encoder_x2topic_h=conv_bn_pool_2 --nname_encoder_sandwich_x2h4zd=conv_bn_pool_2
```


## User defined tasks

### Folder Task
#### Folder Task with implemented neural network
```shell
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --debug --bs=2 --nname=alexnet --model=diva --nname_dom=alexnet --gamma_y=7e5 --gamma_d=1e5
```

#### Folder Task with externally user defined neural network
```shell
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --debug --bs=2 --npath=examples/nets/resnet.py --model=diva --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5
```

### ImagePath Task
#### ImagePath Task with implemented algorithm
```shell
python main_out.py --te_d=sketch --tpath=examples/tasks/demo_task_path_list_small.py --debug --bs=2 --nname=alexnet --model=diva --nname_dom=alexnet --gamma_y=7e5 --gamma_d=1e5
```

#### ImagePath Task with externally user defined neural network
```shell
python main_out.py --te_d=sketch --tpath=examples/tasks/demo_task_path_list_small.py --debug --bs=2 --npath=examples/nets/resnet.py --model=diva --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5
```

## Custom algorithm defined in external python file
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=3 --apath=examples/models/demo_custom_model.py --model=custom --nname_argna2val my_custom_arg_name --nname_argna2val alexnet
```

```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=3 --apath=examples/models/demo_custom_model.py --model=custom --npath_argna2val my_custom_arg_name --npath_argna2val examples/nets/resnet.py
```

## Adversarial images training
```shell
python main_out.py --te_d=0 --task=mnistcolor10 --keep_model --model=erm --trainer=dial --nname=conv_bn_pool_2
```
### Train DIVA model with DIAL trainer

```shell
python main_out.py --te_d 0 1 2 --tr_d 3 7 --task=mnistcolor10 --model=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=7e5 --gamma_d=1e5 --trainer=dial
```
### Set hyper-parameters for trainer as well
```shell
python main_out.py --te_d 0 1 2 --tr_d 3 7 --task=mnistcolor10 --model=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=7e5 --gamma_d=1e5 --trainer=dial --dial_steps_perturb=1
```

## Meta Learning Domain Generalization
```shell
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --model=erm --trainer=mldg --nname=alexnet
```
