import torch

def get_device(flag_no_cu):
    flag_cuda = torch.cuda.is_available() and (not flag_no_cu)
    device = torch.device("cuda" if flag_cuda else "cpu")
    print("")
    print("using device:", str(device))
    print("")
    return device
