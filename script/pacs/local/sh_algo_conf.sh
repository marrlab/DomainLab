#!/bin/bash
declare -A dict_algo_args
dict_algo_args['diva']='--npath_dom=examples/nets/resnet.py --gamma_y=1e5 --gamma_d=1e5'
dict_algo_args['matchdg']='--epochs_ctr=1 --epochs_erm=1'
dict_algo_args['hduva']='--gamma_y=1e5 --nname_topic_distrib_img2topic=conv_bn_pool_2 --nname_encoder_sandwich_layer_img2h4zd=conv_bn_pool_2'
dict_algo_args['deepall']=''
# echo ${dict_algo_args['diva']}


