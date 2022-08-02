declare -A dict_algo_args # define dictionary
dict_algo_args['diva']='--nname_dom=conv_bn_pool_2 --gamma_y=1e5 --gamma_d=1e5'
dict_algo_args['hduva']='--gamma_y=7e5 --nname_topic_distrib_img2topic=conv_bn_pool_2 --nname_encoder_sandwich_layer_img2h4zd=conv_bn_pool_2'
dict_algo_args['matchdg']='--epochs_ctr=20 --epochs_erm=80'  # FIXME
dict_algo_args['deepall']=''

# echo ${dict_algo_args['diva']}
