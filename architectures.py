import torch
import os
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F

from typing import Dict

#NEW
import lightning.pytorch as pl


class MultiParametricMRIModel(pl.LightningModule): #(nn.Module):

    def __init__(self, FC, dropout, backbone="ResNet50"):
        """
        Each branch is a ResNet50, pre-trained on ImageNet. The head is substituted by a FC (4096,2).
        """
        super(MultiParametricMRIModel, self).__init__()
        self.name = "MultiparametricMRIModel"
        self.backbone = backbone
        self.branchDWI = BranchModel(self.backbone, "DWI")
        self.branchT2 = BranchModel(self.backbone, "T2")
        self.branchDCEpeak = BranchModel(self.backbone, "DCE_peak")
        self.branchDCE3TP = BranchModel(self.backbone, "DCE_3TP")

        #Multiparametric classifier
        features_dimension = 4*self.branchDWI.extracted_features_dim   # TODO - it is better to parametrize the 4 with the number of branches, i.e. len(args.branches)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(features_dimension, FC), 
            nn.ReLU(),
            nn.Linear(FC, 2), 
            #nn.Softmax(dim = 1)
        )

    def forward(self, x: torch.Tensor) -> Dict:
        #   (DWI_pre, DWI_post, T2_pre, T2_post,DCEpeak_pre, DCEpeak_post, DCE3TP_pre, DCE3TP_post)  # TODO - include the control on the selected branches for the experiments
        x_DWI_1, x_T2_1, x_DCEpeak_1, x_DCE3TP_1 = x[0], x[2], x[4], x[6] 
        x_DWI_2, x_T2_2, x_DCEpeak_2, x_DCE3TP_2 = x[1], x[3], x[5], x[7]

        DWI_features, DWI_probs = self.branchDWI(x_DWI_1, x_DWI_2)
        T2_features, T2_probs = self.branchT2(x_T2_1, x_T2_2)
        DCEpeak_features, DCEpeak_probs = self.branchDCEpeak(x_DCEpeak_1, x_DCEpeak_2)
        DCE3TP_features, DCE3TP_probs = self.branchDCE3TP(x_DCE3TP_1, x_DCE3TP_2)

        concatenated_features = torch.cat((DWI_features, T2_features, DCEpeak_features, DCE3TP_features), dim=1)
        print(f"\n### Shape ALL features concatenated: {concatenated_features.shape}")
        final_probs = self.classifier(concatenated_features)

        probs = {
            "pCR" : final_probs,   
            "DWI_probs" : DWI_probs,
            "T2_probs" : T2_probs,
            "DCEpeak_probs" : DCEpeak_probs,
            "DCE3TP_probs" : DCE3TP_probs
        }

        return probs


class BranchModel(pl.LightningModule): #(nn.Module):

    def __init__(self, backbone, branch_name):
        """
        By default each branch is a ResNet50, pre-trained on ImageNet. The head is substituted by a FC (4096,2).
        """
        super(BranchModel, self).__init__()

        self.branch_name = branch_name
        if backbone == "ResNet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        #conv_layer_params = [param for name, param in self.model.named_parameters() if 'conv' in name]
        
        self.extracted_features_dim = 2 * self.model.fc.in_features
        self.model.fc = nn.Sequential(
                nn.Linear(self.extracted_features_dim,2),
                #nn.Softmax(dim=1)                     
            )
        
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
        probs = self.model.fc(conc_features)
        
        return conc_features, probs
    
