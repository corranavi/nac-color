import torch
import logging
import numpy as np
from torchvision.transforms import v2 as T

# datagen_train = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
#                                            samplewise_center=True,
#                                            samplewise_std_normalization=True,
#                                            vertical_flip=True,
#                                            zoom_range=0.3,
#                                            brightness_range=[0.5, 3],
#                                            shear_range=0.3)

def get_transformations(choice: int = 1) -> "torchvision.transforms":
    if choice == 1:
        transforms = T.Compose([
            T.RandomResizedCrop(size=(224, 224), scale=(1,1), antialias=True),
            T.RandomHorizontalFlip( p = 0.5),
            T.ColorJitter( brightness = (0.5 , 3)),
            T.RandomAffine(
                degrees = 0,
                scale =(0.7, 1.3),
                shear = 0.3
            )
        ])
    else:
        transforms = None
    return transforms

def normalize_slice_channelwise(slice: torch.Tensor)-> torch.Tensor:
    """
    Apply normalization to a single 3-channels image.
    """
    std,mean = torch.std_mean(slice, dim=(-2,-1))
    #print(f"Mean: {mean} | STD: {std}")
    normalization = T.Compose([
        T.Normalize(mean=mean, std=std)
    ])
    normalized_slice = normalization(slice)
    std_n,mean_n = torch.std_mean(normalized_slice, dim=(-2,-1))
    print(f"Post normalization:       Mean: {mean_n} | STD: {std_n}")
    return normalized_slice

def normalize_slices(t1: torch.Tensor)-> torch.Tensor:
    """
    This method apply normalization to the whole test dataset.
    """
    X = torch.Tensor()
    #t1.shape = [BATCH, MODALITIES, 3, 224,224]
    for modality in range(t1.shape[0]):
      slice_normalized = normalize_slice_channelwise(t1[modality])
      X = torch.cat((X, slice_normalized.unsqueeze(0)))
    return X

# def normalize_slices(slices_tensor: torch.Tensor)-> torch.Tensor:
#     """
#     This method apply normalization to a tensor of slices.
#     """
#     normalized_slices_tensor = torch.Tensor()
#     for modality in range(slices_tensor.shape[0]):
#         print(f"Slices modality shape: {slices_tensor[modality].shape}")
#         print(slices_tensor[modality])
#         slice_normalized = normalize_slice_channelwise(slices_tensor[modality])
#         normalized_slices_tensor = torch.cat((normalized_slices_tensor, slice_normalized.unsqueeze(0)))
#     return normalized_slices_tensor

def get_patient_level(Y_test: torch.Tensor, Y_prob: torch.Tensor, slices: int):
    """
    
    """
    logging.info(f"Samples: {len(Y_test)}")
    logging.info(f"Slices: {slices*2+1}")
    patient_num = len(Y_test) / (slices * 2 + 1)
    logging.info(f"Patient num: {patient_num}")
    Y_test_split = []
    Y_prob_split = []

    split = np.array_split(Y_test, patient_num)

    for slice in split:
        all_equal = all(pCR == slice.argmax(axis=-1)[0] for pCR in slice.argmax(axis=-1))   #tutte le labels dello stesso paziente devono essere uguali

        if all_equal:
            Y_test_split.append(slice[0])
        else:
            print("Error in slices division per patient!")
            return

    split = np.array_split(Y_prob, patient_num)
    split = np.mean(split, axis=1)             #il voto di maggioranza è eseguito così: si fa la media delle probabilità per 0 e la media delle probabilità per 1, invece che passare già dalle predicted labels

    for slice in split:
        Y_prob_split.append(slice)

    return torch.Tensor(np.array(Y_test_split)), torch.Tensor(np.array(Y_prob_split))