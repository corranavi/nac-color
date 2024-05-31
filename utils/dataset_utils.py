import torch
import logging
import numpy as np
from torchvision.transforms import v2 as T

def get_transformations():
    transforms = T.Compose([
            T.RandomResizedCrop(size=(224, 224), scale=(1,1), antialias=True), #interpolation=InterpolationMode.NEAREST,
            T.RandomVerticalFlip( p = 0.5),
            T.ColorJitter( brightness = (0.5 , 2)),
            T.RandomAffine(
                degrees = 0,
                scale =(0.7, 1.3),
                shear = 0.3
            ),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) --> normalization on ImageNet dataset.
        ])
    return transforms

def get_val_transformations():
    transforms = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) --> normalization on ImageNet dataset.
        ])
    return transforms


def normalize_slice_channelwise(slice: torch.Tensor)-> torch.Tensor:
    """
    Apply normalization to a single 3-channels image.
    """
    std,mean = torch.std_mean(slice, dim=(-2,-1))
    #logging.info(f"Mean: {mean} | STD: {std}")
    normalization = T.Compose([
        T.Normalize(mean=mean, std=std)
    ])
    normalized_slice = normalization(slice)
    #std_n,mean_n = torch.std_mean(normalized_slice, dim=(-2,-1))
    #logging.info(f"Post normalization:       Mean: {mean_n} | STD: {std_n}")
    return normalized_slice

def normalize_slices(t1: torch.Tensor)-> torch.Tensor:
    """
    This method apply slice-wise normalization to the input batch.
    Expected tensor input shape: [BATCH_SIZE, NUM_MODALITIES, C, H, W]
    """
    X = torch.Tensor()
    for modality in range(t1.shape[0]):
      slice_normalized = normalize_slice_channelwise(t1[modality])
      X = torch.cat((X, slice_normalized.unsqueeze(0)))
    return X