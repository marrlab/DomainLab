# Examples

## Colored version of MNIST

### leave one domain out
```shell
python main_out.py --te_d=0 --task=mnistcolor10 --keep_model --aname=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=10e5 --gamma_d=1e5
```

### choose train and test
```shell
python main_out.py --te_d 0 1 2 --tr_d 3 7 --task=mnistcolor10 --aname=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=7e5 --gamma_d=1e5
```

### make a sanity check for the dataset using 8 instances from each domain and from each class
```shell
python main_out.py --te_d=0 --task=mini_vlcs --debug --bs=2 --aname=diva --nname=alexnet --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5 --san_check --san_num=4
```
### sanity check on only 2 train domains and 2 test domain2
```shell
python main_out.py --te_d 0 1 --tr_d 3 5 --task=mnistcolor10 --debug --bs=2 --aname=deepall --nname=conv_bn_pool_2 --san_check --san_num=4
```

### generation of images
```shell
python main_out.py --te_d=0 --task=mnistcolor10 --keep_model --aname=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=10e5 --gamma_d=1e5 --gen
```

### use hduva on color mnist, train on 2 domains
```shell
python main_out.py --tr_d 0 1 2 --te_d 3 --bs=2 --task=mnistcolor10 --aname=hduva  --nname=conv_bn_pool_2 --gamma_y=7e5 --nname_topic_distrib_img2topic=conv_bn_pool_2 --nname_encoder_sandwich_layer_img2h4zd=conv_bn_pool_2
```

### hduva is domain-unsupervised, so it works also with a single domain
```shell
python main_out.py --tr_d 0  --te_d 3 4 --bs=2 --task=mnistcolor10 --aname=hduva --nname=conv_bn_pool_2 --gamma_y=7e5 --nname_topic_distrib_img2topic=conv_bn_pool_2 --nname_encoder_sandwich_layer_img2h4zd=conv_bn_pool_2
```



