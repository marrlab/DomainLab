import torch.nn as nn


class NetTorchVisionBase(nn.Module):
    """
    fetch model from torchvision
    """
    def __init__(self, flag_pretrain):
        super().__init__()
        self.net_torchvision = None
        self.fetch_net(flag_pretrain)

    def fetch_net(self, flag_pretrain):
        raise NotImplementedError

    def forward(self, tensor):
        """
        delegate forward operation to self.net_torchvision
        """
        out = self.net_torchvision(tensor)
        return out

    def show(self):
        """
        print out which layer will be optimized
        """
        for name, param in self.net_torchvision.named_parameters():
            if param.requires_grad:
                print("layers that will be optimized: \t", name)
