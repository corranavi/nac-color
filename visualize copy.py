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

torch.manual_seed(42)
np.random.seed(42)

percentiles = [1,99]
image = np.random.randint(low=0, high=65536, size=(5, 5), dtype=np.uint16)
print(image)
plt.imshow(image)
pmin, pmax = np.percentile(image, percentiles)
print(f"pmin {pmin}\npmax {pmax}")
image = (image - pmin) / (pmax - pmin)
image[image<0] = 0
image[image>1] = 1
image_int8 = (image * 255).astype(np.uint8)
print(image_int8)
plt.imshow(image_int8)
