import torch
from torchvision.transforms import v2 as T

def get_transformations(choice: int = 1) -> "torchvision.transforms":
    if choice == 1:
        transforms = T.Compose([
            T.RandomResizedCrop(size=(224, 224), antialias=True),
            T.RandomHorizontalFlip( p = 0.5),
            T.ColorJitter( brightness = (0.5 , 3)),
            T.RandomAffine(
                degrees = 0,
                #scale =(0, 0.3),
                shear = 0.3
            )
        ])
    else:
        transforms = None
    return transforms

def normalize_slice_channelwise(slice: torch.Tensor)-> torch.Tensor:
    std,mean = torch.std_mean(slice, dim=(1,2))
    print(f"Mean: {mean} | STD: {std}")
    normalization = T.Compose([
        T.Normalize(mean=mean, std=std)
    ])
    normalized_slice = normalization(slice)
    std_n,mean_n = torch.std_mean(normalized_slice, dim=(1,2))
    print(f"Mean: {mean_n} | STD: {std_n}")
    return normalized_slice

def apply_same_transformation_all_modalities(original_tensor: torch.Tensor, transform : object = None):
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
        print(f"Shape slice_tensor after transformation: {slice_tensor.shape}")

        divided_modalities = torch.Tensor()
        for i in range(num_modalities):
            modality_i = slice_tensor[i].unsqueeze(0)#.unsqueeze(0) #ok
            divided_modalities = torch.cat((divided_modalities, modality_i)) #ok
        
        #print(f"Modalità riassemblate: {modalita_separate}")
        print(f"Shape after divided modalities: {divided_modalities.shape}")
        reconstructed_tensor = torch.cat((reconstructed_tensor, divided_modalities.unsqueeze(0)))

    print("\Original tensor has shape: ", original_tensor.shape)
    reconstructed_tensor = torch.transpose(reconstructed_tensor, 0,1)
    print("\nTransformed tensor has shape: ", reconstructed_tensor.shape)
    assert original_tensor.shape == reconstructed_tensor.shape

    return reconstructed_tensor



if __name__ == "__main__":

    #random_tensor = torch.randint(low=0, high=256, size = (3,4,4)).float() #.unsqueeze(0)
    random_tensor = torch.Tensor([[[1., 2., 3.],
                                 [1., 2., 3.],
                                 [1., 2., 3.]],
                                 [[1., 2., 3.],
                                 [1., 2., 3.],
                                 [1., 2., 3.]],
                                 [[1., 2., 3.],
                                 [1., 2., 3.],
                                 [1., 2., 3.]]])
    print(random_tensor.shape)
    print(random_tensor)
    # try:
    normalized = normalize_slice_channelwise(random_tensor)

    transformations = get_transformations()

    transformed = transformations(normalized)
    print(transformed)

    # except:
    #     print("Qualcosa è andato storto")