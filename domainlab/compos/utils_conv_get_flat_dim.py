import torch


def get_flat_dim(module, i_channel, i_h, i_w, batchsize=5):
    """flat the convolution layer output and get the flat dimension for fully
    connected network
    :param module:
    :param i_channel:
    :param i_h:
    :param i_w:
    :param batchsize:
    """
    img = torch.randn(i_channel, i_h, i_w)
    img3 = img.repeat(batchsize, 1, 1, 1)  # create batchsize repitition
    conv_output = module(img3)
    if len(conv_output.shape) == 2:
        flat_dim = conv_output.shape[1]
    else:
        flat_dim = conv_output.shape[1] * conv_output.shape[2] * conv_output.shape[3]
    return flat_dim
