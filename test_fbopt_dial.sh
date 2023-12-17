export CUDA_VISIBLE_DEVICES="" 
python main_out.py --te_d=caltech --task=mini_vlcs --bs=16  --model=fboptdial --trainer=dial --nname=alexnet --nname_dom=alexnet --gamma_y=1e6 --gamma_d=1e6
