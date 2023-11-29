"""
make an experiment
"""
from torch import nn
from torchvision import models as torchvisionmodels
from torchvision.models import ResNet50_Weights
from torchvision import transforms
import os
from domainlab.mk_exp import mk_exp
from domainlab.dsets.dset_mnist_color_solo_default import DsetMNISTColorSoloDefault
from domainlab.tasks.task_dset import mk_task_dset
from domainlab.models.model_deep_all import mk_deepall
from domainlab.tasks.utils_task import ImSize
from domainlab.tasks.task_folder_mk import mk_task_folder
from domainlab.arg_parser import mk_parser_main
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
class FCLayer(nn.Module):
    def __init__(self, model, model_type, num_classes):
        super(FCLayer, self).__init__()
        self.model = model
        if model_type == 'vit_b_16':
            num_ftrs = 768
        elif model_type == 'vit_b_32':
            num_ftrs = 768
        elif model_type == 'vit_l_16':
            num_ftrs = 1024
        elif model_type == 'vit_l_32':
            num_ftrs = 1024
        else:
            print('Invilade model type!!!')
        self.fc = nn.Linear(num_ftrs, num_classes)
    def forward(self, x):
        x = self.model(x)['getitem_5']
        out = self.fc(x)
        return out

class PTVIT(nn.Module):
    def __init__(self, model_type, num_cls=15):
        super(PTVIT, self).__init__()
        if model_type == 'vit_b_16':
            self.model = vit_b_16(pretrained=True)
        elif model_type == 'vit_b_32':
            self.model = vit_b_32(pretrained=True)
        elif model_type == 'vit_l_16':
            self.model = vit_l_16(pretrained=True)
        elif model_type == 'vit_l_32':
            self.model = vit_l_32(pretrained=True)
        else:
            print('Invilade model type!!!')
        # freeze all the network except the final layer
        # for param in self.model.parameters():
        #  	param.requires_grad = False
        # These two lines not needed but, you would use them to work out which node you want
        nodes, eval_nodes = get_graph_node_names(self.model)
        #print('model nodes:', nodes)
        self.features_encoder = create_feature_extractor(self.model, return_nodes=['encoder'])
        #print('features_encoder:', self.features_encoder)
        self.features_vit_flatten = create_feature_extractor(self.model, return_nodes=['getitem_5'])
        self.features_fc = create_feature_extractor(self.model, return_nodes=['heads'])
        self.model_final = FCLayer(self.features_vit_flatten, model_type, num_cls)

    def forward(self, input):
        #out1 = self.features_vit_flatten(input)
        #print('out1:', out1['getitem_5'].shape)
        out = self.model_final(input)
        return out
def test_mk_exp():
    """
    test mk experiment API
    """
    abs_path2domainlab = "~/codes/DomainLab"
    model_type = 'vit_b_16'
    model = PTVIT(model_type)
    # specify domain generalization task
    task = mk_task_folder(extensions={"caltech": "jpg", "sun":
                                      "jpg", "labelme": "jpg"},
                          list_str_y=["chair", "car"],
                          dict_domain_folder_name2class={
                              "caltech": {"auto": "car", "stuhl": "chair"},
                              "sun": {"vehicle": "car", "sofa": "chair"},
                              "labelme": {"drive": "car", "sit": "chair"}
                          },
                          dict_domain_img_trans={
                              "caltech": transforms.Compose(
                                  [transforms.Resize((256, 256)),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ]),
                              "sun": transforms.Compose(
                                  [transforms.Resize((256, 256)),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ]),
                              "labelme": transforms.Compose(
                                  [transforms.Resize((256, 256)),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ]),
                          },
                          img_trans_te=transforms.Compose(
                              [transforms.Resize((256, 256)),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                          isize=ImSize(3, 224, 224),
                          dict_domain2imgroot={
                              "caltech": os.path.join(
                                  abs_path2domainlab,
                                  "data/vlcs_mini/caltech/"),
                              "sun": os.path.join(
                                  abs_path2domainlab,
                                  "data/vlcs_mini/sun/"),
                              "labelme": os.path.join(
                                  abs_path2domainlab,
                                  "data/vlcs_mini/labelme/")},
                          taskna="e_mini_vlcs")
    # specify backbone to use
    # backbone = torchvisionmodels.resnet.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    backbone = PTVIT(model_type='vit_b_16', num_cls=task.dim_y)
    # num_final_in = backbone.fc.in_features
    # backbone.fc = nn.Linear(num_final_in, dim_y)
    # specify model to use
    model = mk_deepall()(backbone)
    # make trainer for model
    exp = mk_exp(task, model, trainer="mldg", test_domain="caltech", batchsize=2)
    exp.execute(num_epochs=2)
if __name__ == '__main__':
    test_mk_exp()
