import torch.nn as nn


class LayerId(nn.Module):
    """
    used to delete layers
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x:
        """
        return x


class DenseNet(nn.Module):
    """
    (input)-dropout-linear-relu-dropout-linear-relu(output)
    """
    def __init__(self, input_flat_size, out_hidden_size=1024, last_feat_dim=4096, p_dropout=0.5):
        """
        :param input_flat_size:
        :param out_hidden_size:
        :param last_feat_dim:
        """
        super().__init__()
        self.h_layers = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(input_flat_size, last_feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(last_feat_dim, out_hidden_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, tensor_x):
        """
        :param x:
        """
        flat = tensor_x.view(tensor_x.shape[0], -1)
        out = self.h_layers(flat)
        return out
