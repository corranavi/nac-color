import torch
from torchvision.transforms import v2 as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

tensore_di_prova = torch.randn((2,8,5,3,4,4))
#2 pazienti
#   8 modalità
#       9 slice per modalità
#           3x4x4 pixel per ciascuna slice

#prendi un paziente alla volta. prendi una slice per volta da ciascun canale, concatenale per applicare la
#trasformazione, poi smembra il nuovo tensore e rimetti nei singoli canali

def printa_dimensioni_tensore_di_prova(tensore_di_prova):
    for patient_index in range(tensore_di_prova.shape[0]):
        patient_tensor = tensore_di_prova[patient_index]
        print(f"Patient number {patient_index+1} : {patient_tensor.shape}")

        for modality_index in range(patient_tensor.shape[0]):
            modality_tensor = patient_tensor[modality_index]
            print(f"Modality number {modality_index+1} : {modality_tensor.shape}")

            for slice_index in range(modality_tensor.shape[0]):
                slice_tensor = modality_tensor[slice_index]
                print(f"Slice number {slice_index+1} : {slice_tensor.shape}")
            
            print()
        
        print()

#def trasponi_trasforma_riporta_dimensioni(tensore_di_prova):


# for patient_index in range(tensore_di_prova.shape[0]):
#     patient_tensor = tensore_di_prova[patient_index]
#     print(f"Patient number {patient_index+1} : {patient_tensor.shape}")

#     transposed = torch.transpose(patient_tensor, 0,1)

#     for slice_index in range(transposed.shape[0]):
#         slice_tensor = transposed[slice_index]
#         print(f"slice number {slice_index+1} : {slice_tensor.shape}")

#         single_slice_all_modalities_tensor = torch.Tensor()

#         for modality_index in range(slice_tensor.shape[0]):
#             modality_tensor = slice_tensor[modality_index]
#             print(f"Modality number {modality_index+1} : {modality_tensor.shape}")

#             single_slice_all_modalities_tensor = torch.cat((single_slice_all_modalities_tensor, modality_tensor.unsqueeze(0)))
        
#         print(f"Questa lista mi aspetto che abbia 8 tensori: {single_slice_all_modalities_tensor.shape}")

#         #single_slice_all_modalities_tensor = transform(single_slice_all_modalities_tensor)

transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
])


num_slices = 9
num_modalities = 8

tensorino = torch.randn(num_modalities,num_slices,3,4,4)

single_slice_all_modalities = [[] for _ in range(num_slices)]
print(single_slice_all_modalities)

for modalita in range(tensorino.shape[0]):
    print(f"Modalità numero {modalita}")
    for i in range(tensorino[modalita].shape[0]):
        print(f"Slice numero {i} - ",end="")
        print(f"Dimensioni: {tensorino[modalita][i].shape}")

        single_slice_all_modalities[i].append(tensorino[modalita][i])
    
    print()

print(f"Mi aspetto ci siano tanti elementi quante le slice, ovvero {num_slices}: {len(single_slice_all_modalities)}")
print(f"Mi aspetto ci siano tanti elementi quante le modalita, ovvero {num_modalities}: {len(single_slice_all_modalities[0])}")
print(single_slice_all_modalities[0][0].shape)

single_slice_all_modalities[0][0] = transforms(single_slice_all_modalities[0][0])

print(f"LEN {len(single_slice_all_modalities)}")
print(f"LEN {len(single_slice_all_modalities[0])}")

#Ora devo rimettere assieme la struttura delle sequenze ---------------------------------------------
tensore_ricostruito = []
for modalita in range(num_modalities):
    modalita_ricostruita = []
    for index_slice in range(num_slices):
        slice = single_slice_all_modalities[modalita][index_slice]
        print(f"\t{slice.shape}")
        modalita_ricostruita.append(slice)
        print(len(modalita_ricostruita))
    tensore_ricostruito.append(modalita_ricostruita)

print(f"Tensore ricostruito deve avere {num_modalities} modalita: {len(tensore_ricostruito)}")
print(f"Ciascuna modalità deve avere {num_slices} slices: {len(tensore_ricostruito[0])}")

tensore_ricostruito = torch.Tensor(tensore_ricostruito)

slice1_MRI1 = np.clip(tensorino[0][0], 0, 1)
slice1_MRI2 = np.clip(tensorino[1][0], 0, 1)
slice2_MRI1 = np.clip(tensorino[0][1], 0, 1)
slice2_MRI2 = np.clip(tensorino[1][1], 0, 1)
slice3_MRI1 = np.clip(tensorino[0][2], 0, 1) # questo non dovrebbe essere trasformato

slice1_MRI1_tr = np.clip(tensore_ricostruito[0][0], 0, 1)
slice1_MRI2_tr = np.clip(tensore_ricostruito[1][0], 0, 1)
slice2_MRI1_tr = np.clip(tensore_ricostruito[0][1], 0, 1)
slice2_MRI2_tr = np.clip(tensore_ricostruito[1][1], 0, 1)
slice3_MRI1_tr = np.clip(tensore_ricostruito[0][2], 0, 1) #questo mi aspetto non sia cambiato

print(f"STAMPAAA {tensore_ricostruito.shape}")
def printa_immagine(immagine1, immagine2):
    # plt.imshow(immagine1.view(immagine1.shape[1], immagine1.shape[2], immagine1.shape[0]))
    # plt.show()
    plt.imshow(immagine2.view(immagine2.shape[1], immagine2.shape[2], immagine2.shape[0]))
    plt.show()

#prima
printa_immagine(slice1_MRI1, slice1_MRI1_tr)

#seconda
printa_immagine(slice1_MRI2, slice1_MRI2_tr)

#terza
printa_immagine(slice2_MRI1, slice2_MRI1_tr)

#quarta
printa_immagine(slice2_MRI2, slice2_MRI2_tr)

#quinta - NON CAMBIA
printa_immagine(slice3_MRI1, slice3_MRI1_tr)