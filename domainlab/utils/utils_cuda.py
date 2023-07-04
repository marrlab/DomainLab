"""
choose devices
"""
import torch
from domainlab.utils.logger import Logger


def get_device(args):
    """
    choose devices
    """
    flag_no_cu = args.nocu
    flag_cuda = torch.cuda.is_available() and (not flag_no_cu)
    if args.device is None:
        device = torch.device("cuda" if flag_cuda else "cpu")
    else:
        device = torch.device("cuda:"+args.device if flag_cuda else "cpu")
    logger = Logger.get_logger()
    logger.info("")
    logger.info(f"using device: {str(device)}")
    logger.info("")
    return device
