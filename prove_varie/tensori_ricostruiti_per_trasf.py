import torch

"""
Ok, questo metodo funziona bene. Devo solo applicare la trasformazione nel punto giusto,
dopodichè mi ritorna il tensore con le giuste dimensioni per passare al modello.
"""


num_modalities = 4
num_slices = 5

DWI_tensor_pre = torch.Tensor([[1.0],[1.1],[1.2],[1.3],[1.4],[1.5],[1.6],[1.7],[1.8]])
DWI_tensor_post = torch.Tensor([[10],[11],[12],[13],[14],[15],[16],[17],[18]])

T2_tensor_pre = torch.Tensor([[2.0],[2.1],[2.2],[2.3],[2.4],[2.5],[2.6],[2.7],[2.8]])
T2_tensor_post = torch.Tensor([[20],[21],[22],[23],[24],[25],[26],[27],[28]])

# tensore_del_paziente = torch.cat((DWI_tensor_pre.unsqueeze(0), 
#                                 DWI_tensor_post.unsqueeze(0), 
#                                 T2_tensor_pre.unsqueeze(0), 
#                                 T2_tensor_post.unsqueeze(0)))

tensore_del_paziente = torch.randn((num_modalities, num_slices, 3, 4,4)) #names=["modality","slice","c","h","w"])

print(tensore_del_paziente.shape)

print("DWI PRE")
print(DWI_tensor_pre.shape)
print("\nDWI POST")
print(DWI_tensor_post.shape)

print("\nMODALITA PARAMETRIZZATA")
tensore_riassemblato = torch.Tensor()
for slice in range(num_slices):
    tensore_slice = torch.Tensor()
    for modality in range(num_modalities):
        tensore_slice = torch.cat((tensore_slice, tensore_del_paziente[modality][slice].unsqueeze(0)))
    print(f"Slice numero {slice+1}!")
    print("")

    #TODO applica la trasformazione al tensore_slice

    #Riassembla il tensore
    print(f"Misura del nuovo tensore post-trasf: {tensore_slice.shape}")

    modalita_separate = torch.Tensor()
    for i in range(num_modalities):
        modalita_i = tensore_slice[i].unsqueeze(0)#.unsqueeze(0) #ok
        modalita_separate = torch.cat((modalita_separate, modalita_i)) #ok
    
    #print(f"Modalità riassemblate: {modalita_separate}")
    print(f"Shape del riassemblata: {modalita_separate.shape}")

    tensore_riassemblato = torch.cat((tensore_riassemblato, modalita_separate.unsqueeze(0)))

print("\Tensore originale:")
print("Shape: ", tensore_del_paziente.shape)
#print(tensore_del_paziente)

riassemblato_misure_corrette = torch.transpose(tensore_riassemblato, 0,1)

print("\nTensore riassemblato: ")
print("Shape: ", riassemblato_misure_corrette.shape)
#print(riassemblato_misure_corrette)

#OK!
# print(tensore_del_paziente[0][2])
# print(riassemblato_misure_corrette[0][2])

def apply_same_transformation_all_channels(original_tensor: torch.Tensor, transform : object = None):
    """
    Take as input a tensor referring to a single patient (all modalities, pre and post nac, all slices)
    and process slice by slice so that the same transformation is applied accross the different channels
    (for example vertical flip).
    Args:
        original_tensor: the tensor referring to a single patient
        transform: the torchvision transformation to be applied
    Returns:
        reconstructed_tensor: a tensor with the same dimension as the original, but with transformations applied.
    """
    num_modalities = original_tensor.shape[0]
    num_slices = original_tensor.shape[1]

    reconstructed_tensor = torch.Tensor()

    for slice in range(num_slices):
        slice_tensor = torch.Tensor()
        for modality in range(num_modalities):
            slice_tensor = torch.cat((slice_tensor, original_tensor[modality][slice].unsqueeze(0)))
        print(f"Slice numero {slice+1}!")
        print("")

        #apply transformation
        slice_tensor = transform(slice_tensor)

        #Riassembla il tensore
        print(f"Shape slice_tensor after transformation: {tensore_slice.shape}")

        divided_modalities = torch.Tensor()
        for i in range(num_modalities):
            modality_i = tensore_slice[i].unsqueeze(0)#.unsqueeze(0) #ok
            divided_modalities = torch.cat((divided_modalities, modality_i)) #ok
        
        #print(f"Modalità riassemblate: {modalita_separate}")
        print(f"Shape after divided modalities: {divided_modalities.shape}")
        reconstructed_tensor = torch.cat((reconstructed_tensor, divided_modalities.unsqueeze(0)))

    print("\Original tensor has shape: ", original_tensor.shape)
    reconstructed_tensor = torch.transpose(reconstructed_tensor, 0,1)
    print("\nTransformed tensor has shape: ", reconstructed_tensor.shape)
    assert original_tensor.shape == reconstructed_tensor.shape

    return reconstructed_tensor



