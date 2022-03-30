python main_out.py --te_d=0 --task=mnistcolor10 --keep_model
python main_out.py --te_d 0 1 2 --tr_d 3 4 5 6 7 8 9 --task=mnistcolor10 --keep_model
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --nocu
python main_out.py --te_d=caltech --tpath=libdg/zoo/task_vlcs.py --debug --bs=2 --nocu
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=8 --aname=deepall
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=32 --aname=dann
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=2 --aname=hduva --nocu
python main_out.py --te_d=caltech --task=mini_vlcs --debug --bs=20 --aname=matchdg --epochs_ctr=3 --epochs_erm=3   # learning rate is crutial for having NAN
