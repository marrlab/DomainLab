declare -A dict_algo_args
dict_algo_args['diva']='--npath_dom=examples/nets/resnet.py --gamma_y=1e5 --gamma_d=1e5'
dict_algo_args['matchdg']='--epochs_ctr=1 --epochs_erm=1'
dict_algo_args['hduva']='--gamma_y=1e5 --nname_dom=conv_bn_pool_2'
dict_algo_args['deepall']=''
echo ${dict_algo_args['diva']}


