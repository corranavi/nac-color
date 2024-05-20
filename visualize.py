"""
Questo file serve per scopi di visualizzazione delle trasformazioni applicate
"""
import torch
import torchvision
from torchvision.transforms import v2 as T
from dataset_lib import MRIDataset
from utils.train_utils import retrieve_folders_list, Kfold_split
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

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

if __name__ == "__main__":
    index_to_print = [9,22,148]
    folders_list = retrieve_folders_list("dataset/NAC_Input")
    datasets_list = Kfold_split(folders_list, 1)
    training_list = datasets_list[0]
    validation_list = datasets_list[1]
    slices = 3
    transformations = get_transformations()
    dataset = MRIDataset(datasets_list[0], slices)

    time_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for index in index_to_print:

        images = dataset.__getitem__(index)
        transformed_images = transformations(images)
        comparison_images = torch.cat((images.unsqueeze(0), transformed_images.unsqueeze(0)))

        grid = torchvision.utils.make_grid(comparison_images.view(-1,3,224,224))
        ndarr = grid.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(10,10))
        plt.imshow(ndarr)
        plt.axis('off')  # Turn off axis
        plt.savefig(f'DATA_VIZ/{time_run}_{index}.png', bbox_inches='tight', pad_inches=0)
        plt.close()