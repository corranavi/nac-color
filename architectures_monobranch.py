import torch
import os
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F

from typing import Dict, OrderedDict

#NEW
import lightning.pytorch as pl

number_of_input_channels = 5

class NACColorizedMONOmodel(pl.LightningModule):
    """
    Full model: colorizer + feature extractors + classifiers
    """

    def __init__(self, FC, dropout, backbone="ResNet50", colorize=False, freeze_backbone = True, ckpt_filepath = "", evaluation=False):
        super(NACColorizedMONOmodel, self).__init__()
        self.colorize = colorize

        if self.colorize:
            self.colorizer = ColorizationModule() # MRIColorizer(type="pixelshuffle") 

        self.multiparametric = MonobranchMRIModel(FC, dropout, backbone=backbone, freeze_backbone = freeze_backbone) 


    def forward(self, x: torch.Tensor) -> tuple[torch.tensor, Dict]:
        #x_DWI_1, x_T2_1, x_DCEpeak_1, x_DCE3TP_1 = x[0], x[2], x[4], x[6] 
        #x_DWI_2, x_T2_2, x_DCEpeak_2, x_DCE3TP_2 = x[1], x[3], x[5], x[7]

        # Colorizzazione
        x_DWI_1 = x[0][:,0:1, :, :]
        x_DWI_2 = x[1][:,0:1, :, :]
        x_T2_1 = x[2][:,0:1, :, :]
        x_T2_2 = x[3][:,0:1, :, :]
        x_DCEpeak_1 = x[4][:,0:1, :, :]
        x_DCEpeak_2 = x[5][:,0:1, :, :]
        x_DCE_t0_1 = x[6][:,0:1, :, :] #t0
        x_DCE_t2_1 = x[6][:,2:3, :, :] #t2
        x_DCE_t0_2 = x[7][:,0:1, :, :] #t0
        x_DCE_t2_2 = x[7][:,2:3, :, :] #t2

        # ----- devo sovrapporle 
        x_PRE = torch.cat((x_DWI_1, x_T2_1, x_DCEpeak_1, x_DCE_t0_1, x_DCE_t2_1), dim=1)
        x_POST = torch.cat((x_DWI_2, x_T2_2, x_DCEpeak_2, x_DCE_t0_2, x_DCE_t2_2), dim=1)

        x_PRE_colorized = self.colorizer(x_PRE)
        x_POST_colorized = self.colorizer(x_POST)

        # li concateno affinchè siano nello shape che si aspetta il multiparametric
        colorized_x = torch.cat((x_PRE_colorized.unsqueeze(0), x_POST_colorized.unsqueeze(0)))
        
        # Passo X al modello base classificatore
        probs = self.multiparametric(colorized_x)
        return colorized_x,probs
    
    def _load_pretrained_model(self, ckpt_filepath):
        checkpoint = torch.load(ckpt_filepath)
        colorizer_state_dict = {k.replace('model.colorizer.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.colorizer.')}
        multiparametric_state_dict = {k.replace('model.multiparametric.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.multiparametric.')}
        self.colorizer.load_state_dict(colorizer_state_dict)
        self.multiparametric.load_state_dict(multiparametric_state_dict)
        print(f"Loaded weights!")

    def _reset_classifiers_layers(self):
        # Reset of last layers for pCR Classifier
        pCR_classifier_reset_done = True
        print(f"\nResetting pCR classifier parameters: ")
        for n,modulo in enumerate(self.multiparametric.classifier._modules):
            print(f"{n} - {modulo}")
            if modulo in ["fc1", "fc2"]:    #dropout e activation function non hanno parametri da resettare
                try:
                    self.multiparametric.classifier._modules[modulo].reset_parameters()
                    print(f"\tParametri del modulo {modulo} resettati!")
                except Exception as e:
                    print(f"{e}")
                    pCR_classifier_reset_done = False

        return pCR_classifier_reset_done

# Feature Extractor and Classifier ----------------------------------------------------------------------------------------------------
class MonobranchMRIModel(pl.LightningModule): #(nn.Module):

    def __init__(self, FC, dropout, backbone="ResNet50", freeze_backbone = False):
        """
        Each branch is a ResNet50, pre-trained on ImageNet. The head is substituted by a FC (4096,2).
        """
        super(MonobranchMRIModel, self).__init__()
        self.name = "MultiparametricMRIModel"
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        self.branchHyperMRI = BranchModel(self.backbone, "hyperMRI", self.freeze_backbone)

        #Multiparametric classifier
        self.concatenated_features_dimension = self.branchHyperMRI.extracted_features_dim 
       
        self.classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(p=dropout)),
            ('fc1', nn.Linear(self.concatenated_features_dimension, FC)), 
            ('act1', nn.ReLU()),
            ('fc2',nn.Linear(FC, 2)), 
            #nn.Softmax(dim = 1)
        ])
        )

    def forward(self, x: torch.Tensor) -> Dict:
        #   (DWI_pre, DWI_post, T2_pre, T2_post,DCEpeak_pre, DCEpeak_post, DCE3TP_pre, DCE3TP_post)  # TODO - include the control on the selected branches for the experiments
        x_PRE = x[0]
        x_POST = x[1]

        MRI_PRE_and_POST_features = self.branchHyperMRI(x_PRE, x_POST)
        print(f"\n### Shape of features concatenated: {MRI_PRE_and_POST_features.shape}")
        pCR_logits = self.classifier(MRI_PRE_and_POST_features)

        logits = {
            "pCR" : pCR_logits,   
        }

        return logits


class BranchModel(pl.LightningModule): #(nn.Module):

    def __init__(self, backbone, branch_name, freeze_backbone: bool = False):
        """
        By default each branch is a ResNet50, pre-trained on ImageNet. The head is substituted by a FC (4096,2).
        """
        super(BranchModel, self).__init__()

        self.branch_name = branch_name
        self.freeze_backbone = freeze_backbone

        # Features extractor
        if backbone == "ResNet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        #conv_layer_params = [param for name, param in self.model.named_parameters() if 'conv' in name]
        
        # Classifier
        self.extracted_features_dim = 2 * self.model.fc.in_features
        
        # Freezing 
        if self.freeze_backbone:
           self.model.requires_grad_(False)  # tutti i parametri del modello diventano non trainabili
           #self.model.fc.requires_grad_(True) #TODO - se è MONOBRANCH non ha il classificatore
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.model.conv1(x1)
        x1 = self.model.bn1(x1)
        x1 = self.model.relu(x1)
        x1 = self.model.maxpool(x1)
        x1 = self.model.layer1(x1)
        x1 = self.model.layer2(x1)
        x1 = self.model.layer3(x1)
        x1 = self.model.layer4(x1)
        x1 = self.model.avgpool(x1) 
        x1 = x1.view(int(x1.size()[0]),-1)
        print(f"Shape of extracted features by branch {self.branch_name}, PRE nac: {x1.shape}")

        x2 = self.model.conv1(x2)
        x2 = self.model.bn1(x2)
        x2 = self.model.relu(x2)
        x2 = self.model.maxpool(x2)
        x2 = self.model.layer1(x2)
        x2 = self.model.layer2(x2)
        x2 = self.model.layer3(x2)
        x2 = self.model.layer4(x2)
        x2 = self.model.avgpool(x2) 
        x2 = x2.view(int(x2.size()[0]),-1)
        print(f"Shape of extracted features by branch {self.branch_name}, POST nac: {x2.shape}")

        conc_features = torch.cat((x1, x2), dim=1)
        print(f"Shape of concatenated features (branch {self.branch_name}, pre + post NAC): {conc_features.shape}\n")
        
        return conc_features
    
# Colorizer ------------------------------------------------------------------------------------------

def bn_weight_init(m):
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def _make_res_layers(nl, ni, kernel=3, stride=1, padding=1):
    layers = []
    for i in range(nl):
        layers.append(ResBlock(ni, kernel=kernel, stride=stride, padding=padding))

    return nn.Sequential(*layers)

def conv_layer(in_layer, out_layer, kernel=3, stride=1, padding=1, instanceNorm=False):
    """ Per costruire i conv layer di pixel shuffle """
    return nn.Sequential(
        nn.Conv2d(in_layer, out_layer, kernel_size=kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_layer) if not instanceNorm else nn.InstanceNorm2d(out_layer),
        nn.LeakyReLU(inplace=True)
    )

def icnr(x, scale=4, init=nn.init.kaiming_normal_):
    """ ICNR init of `x`, with `scale` and `init` function.

        Checkerboard artifact free sub-pixel convolution: https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)

class ResBlock(nn.Module):
    def __init__(self, ni, nf=None, kernel=3, stride=1, padding=1):
        super().__init__()
        if nf is None:
            nf = ni
        self.conv1 = conv_layer(ni, nf, kernel=kernel, stride=stride, padding=padding)
        self.conv2 = conv_layer(nf, nf, kernel=kernel, stride=stride, padding=padding)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class ColorizationModule(pl.LightningModule):
    
    def __init__(self, type=None, name="unique_colorizer"):
        super(ColorizationModule, self).__init__()
        self.colorization_name = name

        # if type == "colorU":
        #     self.model = ColorU()
        # elif type == "deconv":
        #     self.model = Deconv()
        # elif type == "pixelshuffle":
        self.model = PixelShuffle()

    def forward(self,x):
        return self.model(x)
    
class BaseDECO(pl.LightningModule):
    def __init__(self, out=224, init=None):
        super().__init__()
        self.out_s = out
        self.init = init
    
    def init_weights(self):
        if self.init == None:
            pass
        elif self.init == 1:
            self.apply(bn_weight_init)

class PixelShuffle(BaseDECO):
    """
        Modello PixelShuffle, che è quello che lavora sulla risoluzione dei checkboard artifacts
    """

    def __init__(self, out=224, init=1, scale=4, lrelu=False):
        super().__init__(out, init)
        self.conv1 = nn.Conv2d(number_of_input_channels, 64, kernel_size=7, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks = _make_res_layers(8,64)
        self.pixel_shuffle = PixelShuffle_ICNR(ni=64, nf=3, scale=scale, lrelu=lrelu)
        self.init_weights()

    def forward(self, xb):
        """
        @:param xb : Tensor "x batch"
          Batch of input images

        @:return tensor
          A batch of output images
        """
        _xb = self.maxpool(self.act1(self.bn1(self.conv1(xb))))
        _xb = self.resblocks(_xb)

        return self.pixel_shuffle(_xb)

class PixelShuffle_ICNR(pl.LightningModule):
    """ Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init,
        and `weight_norm`.

        "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts":
        https://arxiv.org/abs/1806.02658
    """

    def __init__(self, ni: int, nf: int = None, scale: int = 4, icnr_init=True, blur_k=2, blur_s=1,
                 blur_pad=(1, 0, 1, 0), lrelu=True):
        super().__init__()
        nf = ni if nf is None else nf
        self.conv = conv_layer(ni, nf * (scale ** 2), kernel=1, padding=0, stride=1) if lrelu else nn.Sequential(
            nn.Conv2d(64, 3 * (scale ** 2), 1, 1, 0), nn.BatchNorm2d(3 * (scale ** 2)))
        if icnr_init:
            icnr(self.conv[0].weight, scale=scale)
        self.act = nn.LeakyReLU(inplace=False) if lrelu else nn.Hardtanh(-10000, 10000)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        self.pad = nn.ReplicationPad2d(blur_pad)
        self.blur = nn.AvgPool2d(blur_k, stride=blur_s)

    def forward(self, x):
        x = self.shuf(self.act(self.conv(x)))
        return self.blur(self.pad(x))