import torch.nn as nn
from torchvision import models as torchvisionmodels

from domainlab.compos.nn_zoo.nn import LayerId
from domainlab.compos.nn_zoo.nn_torchvision import NetTorchVisionBase
from domainlab.utils.logger import Logger


class AlexNetBase(NetTorchVisionBase):
    """
    .. code-block:: python

       AlexNet(
       (features): Sequential(
           (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
           (1): ReLU(inplace=True)
           (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
           (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
           (4): ReLU(inplace=True)
           (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
           (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
           (7): ReLU(inplace=True)
           (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
           (9): ReLU(inplace=True)
           (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
           (11): ReLU(inplace=True)
           (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
       )
       (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
       (classifier): Sequential(
           (0): Dropout(p=0.5, inplace=False)
           (1): Linear(in_features=9216, out_features=4096, bias=True)
           (2): ReLU(inplace=True)
           (3): Dropout(p=0.5, inplace=False)
           (4): Linear(in_features=4096, out_features=4096, bias=True)
           (5): ReLU(inplace=True)
           (6): Linear(in_features=4096, out_features=7, bias=True)
       )
       )
    """
    def fetch_net(self, flag_pretrain):
        self.net_torchvision = torchvisionmodels.alexnet(
            pretrained=flag_pretrain)


class Alex4DeepAll(AlexNetBase):
    """
    change the last layer output of AlexNet to the dimension of the
    """
    def __init__(self, flag_pretrain, dim_y):
        super().__init__(flag_pretrain)
        if self.net_torchvision.classifier[6].out_features != dim_y:
            logger = Logger.get_logger()
            logger.info(f"original alex net out dim "
                        f"{self.net_torchvision.classifier[6].out_features}")
            num_ftrs = self.net_torchvision.classifier[6].in_features
            self.net_torchvision.classifier[6] = nn.Linear(num_ftrs, dim_y)
            logger.info(f"re-initialized to {dim_y}")


class AlexNetNoLastLayer(AlexNetBase):
    """
    Change the last layer of AlexNet with identity layer,
    the classifier from VAE can then have the same layer depth as deep_all
    model so it is fair for comparison
    """
    def __init__(self, flag_pretrain):
        super().__init__(flag_pretrain)
        self.net_torchvision.classifier[6] = LayerId()
