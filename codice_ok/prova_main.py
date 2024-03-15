import numpy as np
import torch
from torch.utils.data import DataLoader

#custom classes and methods
from dataset_final import MRIDataset
from dataset_utils import retrieve_folders_list, Kfold_split
from utils.dataset_utils import get_transformations

#models
from models import MultiParametricMRIModel

#train-related functions
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve, pairwise_distances

BATCH_SIZE = 2
max_epochs = 1
LR = 0.0001
momentum = 0.9

folders_list = retrieve_folders_list("C:\\Users\\c.navilli\\Desktop\\Prova\\dataset_mini")
datasets = Kfold_split(folders_list, 1) #per semplicità usiamo un solo fold
transformations = get_transformations()

train_dataset = MRIDataset(datasets[0][0], transformations) 
train_loader = DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE,   
                        shuffle = True  
                        )
val_dataset = MRIDataset(datasets[0][1], transform=None)
val_loader = DataLoader(dataset = val_dataset,
                        shuffle = False
                        )

model = MultiParametricMRIModel()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=momentum)

print("SCOPE - EVALUATING TRANSFORMATIONS")
for epoch in range(max_epochs):
    print(f"Epoch {epoch}")
    model.eval()
    #model.train()
    for step_index, (patients,patients_sequences, labels) in enumerate(train_loader):  
        print(f"Step: {step_index}")
        print(f"\nAll patients of the batch #{step_index}: {patients}")
        print(f"Shape of the sequences tensor (of the batch): {patients_sequences.shape}")
        print(f"Shape of the Labels tensor (of the batch): {labels.shape}")
        print()

        modalities_first = torch.transpose(patients_sequences, 2,1)
        print(modalities_first.shape)
        print(patients_sequences.shape)
        for slice_num, slice_series in enumerate(modalities_first[0]):
            print(f"{slice_num+1} - Il tensore con una slice per ciascuna modalità ha dimensioni: {slice_series.shape}")
            probs = model(slice_series)
            for k,v in probs.items():
                print(f"\tBranch: {k} - predicted: {v.argmax().item()}")
            print(f"\n\tGround truth: {labels[0][slice_num]} - {labels[0][slice_num][0].argmax().item()}\n")

            #loss = F.binary_cross_entropy(probs, class_labels.view(probs.shape))

            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
        
        #train_accuracy = compute_accuracy(model, train_loader)
        #val_accuracy = compute_accuracy(model, val_loader)