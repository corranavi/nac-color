import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

#TODO: set the proper input dimensions 

class MultiParametricMRIModel(nn.Module):
    def __init__(self):
        super(MultiParametricMRIModel, self).__init__()

        #DWI
        self.branchDWI = models.resnet50(weights=models.ResNet50_Weights.DEFAULT, )
        self.branchDWI.fc = nn.Linear(2048, 2) #probabilmente qua va raddoppiato l'input perchè la prediction è fatta su pre e post concatenati

        #T2
        self.branchT2 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.branchT2.fc = nn.Linear(2048, 2) #idem come sopra (4096, 2)

        #DCE peak
        self.branchDCEpeak = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.branchDCEpeak.fc = nn.Linear(2048, 2) #idem come sopra

        #DCE 3TP
        self.branchDCE3TP = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.branchDCE3TP.fc = nn.Linear(2048, 2) #idem come sopra

        #Multimodality classifier
        self.classifier = nn.Sequential(
            nn.Dropout(float=0.5),
            nn.Linear(shape_features_concatenate, 2), #dimensione = 4*4096 = 28672
            nn.ReLU(),
            nn.Linear(shape_features_concatenate, 2), #28672
            nn.Softmax()
        )

    def forward(self, x):
        #qua devo capire come passare x. Nelle due linee successive ipotizzo che x sia un tensore che 
        #contenga una slice per ciascuna delle 8 modalità: 
        #   (DWI_pre, DWI_post, T2_pre, T2_post,DCEpeak_pre, DCEpeak_post, DCE3TP_pre, DCE3TP_post)
        x_DWI_1, x_T2_1, x_DCEpeak_1, x_DCE3TP_1 = x[0], x[2], x[4], x[6]
        x_DWI_2, x_T2_2, x_DCEpeak_2, x_DCE3TP_2 = x[1], x[3], x[5], x[7]

        features_DWI_1 = self.process_image_with_branch(self.branchDWI, x_DWI_1)
        features_DWI_2 = self.process_image_with_branch(self.branchDWI, x_DWI_2)
        features_DWI = torch.cat(features_DWI_1, features_DWI_2)
        logits_DWI = self.branchDWI.fc(features_DWI)


        features_T2_1 = self.process_image_with_branch(self.branchT2, x_T2_1)
        features_T2_2 = self.process_image_with_branch(self.branchT2, x_T2_2)
        features_T2 = torch.cat(features_T2_1, features_T2_2)
        logits_T2 = self.branchT2.fc(features_T2)

        features_DCEpeak_1 = self.process_image_with_branch(self.branchDCEpeak, x_DCEpeak_1)
        features_DCEpeak_2 = self.process_image_with_branch(self.branchDCEpeak, x_DCEpeak_2)
        features_DCEpeak = torch.cat(features_DCEpeak_1, features_DCEpeak_2)
        logits_DCEpeak = self.branchDCEpeak.fc(features_DCEpeak)

        features_DCE3TP_1 = self.process_image_with_branch(self.branchDCE3TP, x_DCE3TP_1)
        features_DCE3TP_2 = self.process_image_with_branch(self.branchDCE3TP, x_DCE3TP_2)
        features_DCE3TP = torch.cat(features_DCE3TP_1, features_DCE3TP_2)
        logits_DCE3TP = self.branchDCE3TP.fc(features_DCE3TP)

        concatenated_features = torch.cat(features_DWI, features_T2, features_DCEpeak, features_DCE3TP)
        final_logits = self.classifier(concatenated_features)

        #handle modality DWI
        # x_DWI = self.branchDWI.conv1(x_DWI)
        # x_DWI = self.branchDWI.bn1(x_DWI)
        # x_DWI = self.branchDWI.relu(x_DWI)
        # x_DWI = self.branchDWI.maxpool(x_DWI)
        # x_DWI = self.branchDWI.layer1(x_DWI)
        # x_DWI = self.branchDWI.layer2(x_DWI)
        # x_DWI = self.branchDWI.layer3(x_DWI)
        # x_DWI = self.branchDWI.layer4(x_DWI)
        # x_DWI = self.branchDWI.avgpool(x_DWI) #these are the features extracted by branch DWI
        # features_DWI = x_DWI.view(int(x_DWI.size()[0]),-1)
        # #la prediction è fatta sulla concatenazione di pre_nac + post_nac
        # features_list.extend((features_DWI)) #c'è poi da aggiungere nache quello post-nac, ora è come se stessi considerando solo il pre
        # logits_DWI = self.branchDWI.fc(x_DWI)

        # #handle modality T2
        # x_T2 = self.branchT2.conv1(x_T2)
        # x_T2 = self.branchT2.bn1(x_T2)
        # x_T2 = self.branchT2.relu(x_T2)
        # x_T2 = self.branchT2.maxpool(x_T2)
        # x_T2 = self.branchT2.layer1(x_T2)
        # x_T2 = self.branchT2.layer2(x_T2)
        # x_T2 = self.branchT2.layer3(x_T2)
        # x_T2 = self.branchT2.layer4(x_T2)
        # x_T2 = self.branchT2.avgpool(x_T2) #these are the features extracted by branch T2   
        # features_T2 = x_T2.view(int(x_T2.size()[0]),-1)
        # #la prediction è fatta sulla concatenazione di pre_nac + post_nac
        # features_list.extend((features_T2))
        # logits_T2 = self.branchT2.fc(x_T2)

        # #handle modality DCE_peak
        # x_DCEpeak = self.branchDCEpeak.conv1(x_DCEpeak)
        # x_DCEpeak = self.branchDCEpeak.bn1(x_DCEpeak)
        # x_DCEpeak = self.branchDCEpeak.relu(x_DCEpeak)
        # x_DCEpeak = self.branchDCEpeak.maxpool(x_DCEpeak)
        # x_DCEpeak = self.branchDCEpeak.layer1(x_DCEpeak)
        # x_DCEpeak = self.branchDCEpeak.layer2(x_DCEpeak)
        # x_DCEpeak = self.branchDCEpeak.layer3(x_DCEpeak)
        # x_DCEpeak = self.branchDCEpeak.layer4(x_DCEpeak)
        # x_DCEpeak = self.branchDCEpeak.avgpool(x_DCEpeak) #these are the features extracted by branch DCE peak
        # features_DCEpeak = x_DCEpeak.view(int(x_DCEpeak.size()[0]),-1)
        # #la prediction è fatta sulla concatenazione di pre_nac + post_nac
        # features_list.extend((features_DCEpeak))
        # logits_DCEpeak = self.branchDCEpeak.fc(x_DCEpeak)

        # #handle modality DCE3TP
        # x_DCE3TP = self.branchDCE3TP.conv1(x_DCE3TP)
        # x_DCE3TP = self.branchDCE3TP.bn1(x_DCE3TP)
        # x_DCE3TP = self.branchDCE3TP.relu(x_DCE3TP)
        # x_DCE3TP = self.branchDCE3TP.maxpool(x_DCE3TP)
        # x_DCE3TP = self.branchDCE3TP.layer1(x_DCE3TP)
        # x_DCE3TP = self.branchDCE3TP.layer2(x_DCE3TP)
        # x_DCE3TP = self.branchDCE3TP.layer3(x_DCE3TP)
        # x_DCE3TP = self.branchDCE3TP.layer4(x_DCE3TP)
        # x_DCE3TP = self.branchDCE3TP.avgpool(x_DCE3TP) #these are the features extracted by branch DCE 3TP
        # features_DCE3TP = x_DCE3TP.view(int(x_DCE3TP.size()[0]),-1)
        # #la prediction è fatta sulla concatenazione di pre_nac + post_nac
        # features_list.extend((features_DCE3TP))
        # logits_DCE3TP = self.branchDCE3TP.fc(x_DCE3TP)

        return final_logits, logits_DWI, logits_T2, logits_DCEpeak, logits_DCE3TP
    
    def process_image_with_branch(self, model, x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x) 
        extracted_features = x.view(int(x.size()[0]),-1)
        return extracted_features