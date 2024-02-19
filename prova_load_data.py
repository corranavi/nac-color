#Questo è script per provare la costruzione del dataset e del dataloader.

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import MRIDataset
from dataset_utils import retrieve_folders_list, Kfold_split

def try_dataloader(loader):
    for index, (patient,patient_sequences) in enumerate(loader):  
            print(f"\nPatient: {patient} - ",end="")
            print(f"Tensor list: {patient_sequences.shape}")
            for modality in range(patient_sequences.shape[1]):
                    print(f"\n\tModality {modality}: {patient_sequences[0][modality].shape}")
                    print("\tSIMULATE: process each image of the sequence.")
                    for slice in range(patient_sequences.shape[2]):
                         print(f"\t--- Image {slice}: {patient_sequences[0][modality][slice].shape}")
    print("")

if __name__=="__main__":

    #[num_patients x num_modalities x num_images x num_channels x height_pixels x weight_pixels]

    #1. Istantiate the dataset(s)
    folders_list = retrieve_folders_list("C:\\Users\\c.navilli\\Desktop\\Prova\\dataset_mini")
    datasets = Kfold_split(folders_list, 3)[0]
    print(f"Lista delle kfold:\n{len(datasets)}")

    train_dataset = MRIDataset(datasets[0]) 
    train_loader = DataLoader(dataset=train_dataset,
                             batch_size=1,   #1 perchè si tengono fissi i weights per lo stesso paziente, ma cambiano da paz a paz
                             shuffle = True  
                             )
    val_dataset = MRIDataset(datasets[1])
    val_loader = DataLoader(dataset = val_dataset,
                            shuffle = False
                            )
    
    #2. Simulate the training
    max_epochs = 1

    for epoch in range(max_epochs):
        print(f"Epoch {epoch}")
        # for index, (patient,patient_sequences) in enumerate(train_loader):  
        #     print(f"\nPatient: {patient} - ",end="")
        #     print(f"Tensor list: {patient_sequences.shape}")
        #     for modality in range(patient_sequences.shape[1]):
        #             print(f"\n\tModality {modality}: {patient_sequences[0][modality].shape}")
        #             print("\tSIMULATE: process each image of the sequence.")
        #             for slice in range(patient_sequences.shape[2]):
        #                  print(f"\t--- Image {slice}: {patient_sequences[0][modality][slice].shape}")
        # print("")
        print("#"*10, " LOADING TRAIN DATASET ", "#"*10)
        try_dataloader(train_loader)

        print("\n","§"*10, " LOADING VAL DATASET ", "§"*10)
        try_dataloader(val_loader)
        
    #TO-DO: 
        # implementare lo split train - test con il k-fold: stratified? !Fatto
        # implementare la scelta dei canali (al momento non è parametrizzato, prende tutti e 4 i canali)
        # implementare la scelta di MRI1 +/- MRI2 (al momento non è parametrizzato, concatena MRI2 a MRI1 sempre)
        # capire il training step -> gestione del multi-modello

