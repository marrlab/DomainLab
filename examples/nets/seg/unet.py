
from .unet_parts import *


class UNET_Feature_Extractor(nn.Module):
    def __init__(self, net):
        self.net = net

    def forward(self, x):
        return self.net.go_down(x)
        

class UNet(nn.Module):
    def __init__(self, n_classes, n_channels=3, bilinear=False): # 3: RGB channels
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.x4 = None

    def go_down(self, x):
        self.x1 = self.inc(x)
        self.x2 = self.down1(self.x1)
        self.x3 = self.down2(self.x2)
        self.x4 = self.down3(self.x3)
        self.x5 = self.down4(self.x4)

    def forward(self, x):
        x5 = self.go_down(x)
        x = self.up1(x5, self.x4)
        x = self.up2(x, self.x3)
        x = self.up3(x, self.x2)
        x = self.up4(x, self.x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

# CHANGEME: user is required to implement the following function
# with **exact** signature to return a neural network architecture for
# classification of dim_y number of classes if remove_last_layer=False
# or return the same neural network without the last layer if
# remove_last_layer=False.
def build_feat_extract_net(dim_y, remove_last_layer):
    """
    This function is compulsory to return a neural network feature extractor.
    :param dim_y: number of classes to be classify can be None
    if remove_last_layer = True
    :param remove_last_layer: for resnet for example, whether
    remove the last layer or not.
    """

    small_unet = UNet(dim_y)
    unet_feature = UNET_Feature_Extractor(small_unet)

    return small_unet, unet_feature
