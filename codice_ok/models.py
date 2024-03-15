import torch
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F

from typing import Dict

#TODO: set the proper input dimensions 

class MultiParametricMRIModel(nn.Module):

    def __init__(self):
        """
        Each branch is a ResNet50, pre-trained on ImageNet. The head is substituted by a FC (4096,2) followed
        by a Softmax activation function. The original model seems to output logits instead of soft probs.
        #TODO: Analyze the loss function to understand the output of the individual branches.
        """
        super(MultiParametricMRIModel, self).__init__()

        self.branchDWI = BranchModel()
        self.branchT2 = BranchModel()
        self.branchDCEpeak = BranchModel()
        self.branchDCE3TP = BranchModel()

        #Multimodality classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(16384, 128), #dimensione = 4*4096 = 16384
            nn.ReLU(),
            nn.Linear(128, 2), 
            nn.Softmax(dim = 1)
        )

    def forward(self, x: torch.Tensor) -> Dict:
        #   (DWI_pre, DWI_post, T2_pre, T2_post,DCEpeak_pre, DCEpeak_post, DCE3TP_pre, DCE3TP_post)
        x_DWI_1, x_T2_1, x_DCEpeak_1, x_DCE3TP_1 = x[0].unsqueeze(0), x[2].unsqueeze(0), x[4].unsqueeze(0), x[6].unsqueeze(0)
        x_DWI_2, x_T2_2, x_DCEpeak_2, x_DCE3TP_2 = x[1].unsqueeze(0), x[3].unsqueeze(0), x[5].unsqueeze(0), x[7].unsqueeze(0)

        DWI_features, DWI_probs = self.branchDWI(x_DWI_1, x_DWI_2)
        T2_features, T2_probs = self.branchT2(x_T2_1, x_T2_2)
        DCEpeak_features, DCEpeak_probs = self.branchDCEpeak(x_DCEpeak_1, x_DCEpeak_2)
        DCE3TP_features, DCE3TP_probs = self.branchDCE3TP(x_DCE3TP_1, x_DCE3TP_2)

        concatenated_features = torch.cat((DWI_features, T2_features, DCEpeak_features, DCE3TP_features), dim=1)
        print(f"\n### Shape ALL features concatenated: {concatenated_features.shape}")
        final_probs = self.classifier(concatenated_features)
        print(f"FINAL probs: {final_probs}\n")

        probs = {
            "final_probs" : final_probs,
            "DWI_probs" : DWI_probs,
            "T2_probs" : T2_probs,
            "DCEpeak_probs" : DCEpeak_probs,
            "DCE3TP_probs" : DCE3TP_probs
        }

        return probs
    

class BranchModel(nn.Module):

    def __init__(self):
        """
        Each branch is a ResNet50, pre-trained on ImageNet. The head is substituted by a FC (4096,2) followed
        by a Softmax activation function. The original model seems to output logits instead of soft probs.
        #TODO: Analyze the loss function to understand the output of the individual branches.
        """
        super(BranchModel, self).__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Sequential(
                nn.Linear(4096,2)
                #nn.Softmax(dim=1),       #TODO da rimuovere se voglio solo i logits                        
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
        print(f"Extracted features of single modality, PRE nac: {x1.shape}")

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
        print(f"Extracted features of single modality, POST nac: {x2.shape}")

        conc_features = torch.cat((x1, x2), dim=1)
        print(f"Shape features single modality concatenated (pre + post NAC): {conc_features.shape}\n")
        probs = self.model.fc(conc_features)
        print(f"probs: {probs}\n")
        
        return conc_features, probs