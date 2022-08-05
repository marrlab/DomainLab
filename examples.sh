#!/bin/bash -x -v
set -e  # exit upon first error
# colored version of mnist where each color represent one domain, in total 10 colors
## leave one domain out

start=`date +%s`
python main_out.py --te_d=0 --task=mnistcolor10 --keep_model --aname=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=10e5 --gamma_d=1e5
end=`date +%s`
runtime=$((end-start))
echo $runtime
## choose train and test
python main_out.py --te_d 0 1 2 --tr_d 3 7 --task=mnistcolor10 --aname=diva --nname=conv_bn_pool_2 --nname_dom=conv_bn_pool_2 --gamma_y=7e5 --gamma_d=1e5


# use hduva on color mnist, train on 2 domains
python main_out.py --tr_d 0 1 2 --te_d 3 --bs=2 --task=mnistcolor10 --aname=hduva  --nname=conv_bn_pool_2 --gamma_y=7e5 --nname_topic_distrib_img2topic=conv_bn_pool_2 --nname_encoder_sandwich_layer_img2h4zd=conv_bn_pool_2
# hduva is domain-unsupervised, so it works also with a single domain
python main_out.py --tr_d 0  --te_d 3 4 --bs=2 --task=mnistcolor10 --aname=hduva --nname=conv_bn_pool_2 --gamma_y=7e5 --nname_topic_distrib_img2topic=conv_bn_pool_2 --nname_encoder_sandwich_layer_img2h4zd=conv_bn_pool_2
# hduva on mini VLCS task
python main_out.py --te_d=caltech --bs=2 --task=mini_vlcs --aname=hduva --nname=conv_bn_pool_2 --gamma_y=7e5 --nname_topic_distrib_img2topic=conv_bn_pool_2 --nname_encoder_sandwich_layer_img2h4zd=conv_bn_pool_2
# hduva use alex net
python main_out.py --te_d=caltech --bs=2 --task=mini_vlcs --aname=hduva --nname=conv_bn_pool_2 --gamma_y=7e5 --nname_topic_distrib_img2topic=conv_bn_pool_2 --nname_encoder_sandwich_layer_img2h4zd=alexnet
# hduva use custom net for sandwich encoder
python main_out.py --te_d=caltech --bs=2 --task=mini_vlcs --aname=hduva --nname=conv_bn_pool_2 --gamma_y=7e5 --nname_topic_distrib_img2topic=conv_bn_pool_2 --npath_encoder_sandwich_layer_img2h4zd=examples/nets/resnet.py
# hduva use custom net for topic encoder
python main_out.py --te_d=caltech --bs=2 --task=mini_vlcs --aname=hduva --nname=conv_bn_pool_2 --gamma_y=7e5 --npath_topic_distrib_img2topic=examples/nets/resnet.py --nname_encoder_sandwich_layer_img2h4zd=conv_bn_pool_2
# hduva use custom net for classification encoder
python main_out.py --te_d=caltech --bs=2 --task=mini_vlcs --aname=hduva --npath=examples/nets/resnet.py --gamma_y=7e5 --nname_topic_distrib_img2topic=conv_bn_pool_2 --nname_encoder_sandwich_layer_img2h4zd=conv_bn_pool_2







# Default tasks (mini_vlcs) (224*224 sized image)
## model diva with default neural network
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=diva --nname=alexnet --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5

## model diva with custom neural network
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=diva --npath=examples/nets/resnet.py --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5

## model deepall with default neural network
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --aname=deepall --nname=alexnet
## model deepall with custom neural network
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --aname=deepall --npath=examples/nets/resnet.py
## model matchdg with default neural network
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=matchdg --epochs_ctr=3 --epochs_erm=3 --nname=alexnet
## model matchdg with custom neural network
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=matchdg --epochs_ctr=3 --epochs_erm=3 --npath=examples/nets/resnet.py


###### matchdg learning rate is crutial for not having NAN

# User defined tasks

## Folder Task
### Folder Task with default neural network
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --debug --bs=2 --nname=alexnet --aname=diva --nname_dom=alexnet --gamma_y=7e5 --gamma_d=1e5

### Folder Task with externally user defined neural network
python main_out.py --te_d=caltech --tpath=examples/tasks/task_vlcs.py --debug --bs=2 --npath=examples/nets/resnet.py --aname=diva --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5



## ImagePath Task
### ImagePath Task with default algorithm
python main_out.py --te_d=sketch --tpath=examples/tasks/demo_task_path_list_small.py --debug --bs=2 --nname=alexnet --aname=diva --nname_dom=alexnet --gamma_y=7e5 --gamma_d=1e5

### ImagePath Task with externally user defined neural network
python main_out.py --te_d=sketch --tpath=examples/tasks/demo_task_path_list_small.py --debug --bs=2 --npath=examples/nets/resnet.py --aname=diva --npath_dom=examples/nets/resnet.py --gamma_y=7e5 --gamma_d=1e5



# Custom algorithm defined in external python file
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=3 --apath=examples/algos/demo_custom_algo_builder.py --aname=custom --nname_argna2val my_custom_arg_name --nname_argna2val alexnet

python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=3 --apath=examples/algos/demo_custom_algo_builder.py --aname=custom --npath_argna2val my_custom_arg_name --npath_argna2val examples/nets/resnet.py
