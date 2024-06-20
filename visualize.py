"""
Questo file serve per scopi di visualizzazione delle trasformazioni applicate
"""
import torch
import torchvision
from torchvision.transforms import v2 as T
from dataset_lib import MRIDataset
from utils.train_utils_old import retrieve_folders_list, Kfold_split
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import logging
logging.basicConfig(level=logging.INFO)

torch.manual_seed(42)
np.random.seed(42)

# def get_transformations(choice: int = 1) -> "torchvision.transforms":
#     if choice == 1:
#         transforms = T.Compose([
#             T.RandomResizedCrop(size=(224, 224), scale=(1,1), antialias=True),
#             T.RandomHorizontalFlip( p = 0.5),
#             T.ColorJitter( brightness = (0.5 , 2)), #brightness troppo alta considerando la compressione bit -> provare a riddure max value a 2
#             T.RandomAffine(
#                 degrees = 0,
#                 scale =(0.7, 1.3),
#                 shear = 0.3
#             ),
#             #T.ToTensor()
#         ])
#     else:
#         transforms = None
#     return transforms              #per visualizzare post norm devi riscalare tra min max e poi x 255

def save_images(images, output_name):
    grid = torchvision.utils.make_grid(images.view(-1,3,224,224))
    ndarr = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(10,10))
    plt.imshow(ndarr)
    plt.axis('off')  # Turn off axis
    plt.savefig(output_name, bbox_inches='tight', pad_inches=0)
    plt.close()

# def clip_into_0_255(original_tensor):
#     """
#     This method converts original images such that values are clipped between values 0 and 255 like RGM images.
#     """
#     x = original_tensor.clone()
#     y = x*255/x.max()
#     z = torch.round(y)
#     return z

if __name__ == "__main__":
    index_to_print = [9,22,27] #[9,22,148]
    time_run = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    slices = 3
    full_path_to_dataset = "C:\\Users\\c.navilli\\Desktop\\Prova\\dataset_mini"
    preprocess_type="percentile"                        #choices for preprocess_type = min_max, norm, 10bit, 12bit, 16bit, norm_and_scale. Defaults to norm without scale
    
    if preprocess_type in ["16bit"]:
        multiply_255=True
    else:
        multiply_255=False

    folders_list = retrieve_folders_list(full_path_to_dataset)
    datasets_list = Kfold_split(folders_list, 1)
    
    dataset = MRIDataset(datasets_list[0][0], slices, preprocess_type=preprocess_type) 

    transformations = get_transformations() 

    for index in index_to_print:
        print(index)
        image_file_name = f'2605_2_image_scale_{preprocess_type}_{index}.png'
        images, _ = dataset.__getitem__(index)
        if multiply_255:
            images = images *255
            image_file_name = image_file_name.split(".")[0]+"_255"+".png"
        
        images_for_visualization = images #/(2**16-1)*255 #---> perchè così saranno compresi tra 0 e 1, come si aspetta la visualizzazione per dati di tipo float
        # if index==9:
        #     print(images)
        transformed_images = transformations(images)
        print(f"Min pre transformation: {images.min()} - Max pre transformation: {images.max()}")
        print(f"Min post transformation: {transformed_images.min()} - Max post transformation: {transformed_images.max()}")

        comparison_images = torch.cat((images_for_visualization.unsqueeze(0), transformed_images.unsqueeze(0)))

        save_images(comparison_images, image_file_name)

        
