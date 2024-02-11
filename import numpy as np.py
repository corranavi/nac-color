#script di prova

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import MRIDataset

if __name__=="__main__":

    #1. Istanzia il Dataset (per ora, tutto intero)
    train_dataset = MRIDataset() 
    train_loader = DataLoader(dataset=train_dataset,
                             batch_size=1,   #devo mettere 1 così se la __get_item__ 
                             shuffle = True  #Se a False, l'ordine dei batch è sempre lo stesso
                             )
    
    #2. Training
    max_epochs = 5
    count_results = {}

    for epoch in range(max_epochs):
        
        print(f"Epoch {epoch}")
        for index, (names,labels) in enumerate(train_loader):  
            name_list_4_the_batch=[]       
            for name, label in zip(names,labels):
                name_list_4_the_batch.append(name)
                print(f"\t{name} : {label}")

            name_list_4_the_batch.sort()

            if len(name_list_4_the_batch)>1:
                (name1,name2)=name_list_4_the_batch[0],name_list_4_the_batch[1]
            else:
                (name1,name2)=name_list_4_the_batch[0],''

            if (name1,name2) in count_results.keys():
                count_results[(name1,name2)]+=1
            else:
                count_results[(name1,name2)]=1    
            print("")

    for k,v in count_results.items():
        print(k," - ",v)
    print(f"Total number of pairs : {np.sum([x for x in count_results.values()])}")

    tenzore = torch.tensor([[[[[1,2,3],
                              [1,2,3],
                              [1,2,3]],],]]) #mi aspetto shape [1,3,3]
    print(tenzore.shape)

    # for i in range(0,len(lista),2k+1):
    #   for j in range(i, 2k+1+i):
    
    #Ok, quello qua sotto funziona, per prendere tutte le slice della sottosequenza
    k=2
    lettere = ['a','b','c','d','e','f','g','h','i','j']
    risultato = []
    for i in range(0,len(lettere), 2*k+1):
        sottolista=[]
        for j in range(i,i+2*k+1):
            sottolista.append(lettere[j])
        risultato.append(sottolista)
        print(sottolista)
    print(risultato)

    final_features_list = [ ]
    positions=[(0,1),(2,3),(4,5),(6,7)]
    for couple in positions:
        pre_nac_list = subsequence[couple[0]]
        post_nac_list = subsequence[couple[1]]

        #prendi i primi 2k+1 da pre_nac e poi concatena i successivi 2k+1 da post_nac
        pre_nac_shaded_list = []
        for i in range(0,len(pre_nac_list), 2*k+1):
            individual_patient=[]
            for j in range(i,i+2*k+1):
                individual_patient.append(pre_nac_list[j])
            pre_nac_shaded_list.append(individual_patient)
            print(individual_patient)
        print(pre_nac_shaded_list)

        post_nac_shaded_list = []
        for i in range(0,len(post_nac_list), 2*k+1):
            individual_patient=[]
            for j in range(i,i+2*k+1):
                individual_patient.append(post_nac_list[j])
            post_nac_shaded_list.append(individual_patient)
            print(individual_patient)
        print(post_nac_shaded_list)

        if len(pre_nac_shaded_list)!=len(post_nac_shaded_list):
            print("Le due liste non hanno la stessa lunghezza")
            exit()
        
        total_modality_list = []
        for patient_idx in range(len(post_nac_shaded_list)):
            patient_pre_and_post = pre_nac_shaded_list.extend(post_nac_shaded_list)
            total_modality_list.append(patient_pre_and_post)
        
        final_features_list.append(total_modality_list)

    print("La lista con tutte le features ha {len(final_features_list)} elementi. ")
    for i in range(len(final_features_list)):
        print(f"La modalità numero {i+1} ha {len(final_features_list[i])} sequenze, ovvero pazienti:")
        for j in range(len(final_features_list[i])):
            print(f"La sequenza {j+1} del paziente {i+1} ha {len(final_features_list[i][j])} slices.")
        print("")
    print("")
