"""
Questo file serve per scopi di visualizzazione delle trasformazioni applicate
"""
import torch
import os
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.transforms import v2 as T
from dataset_lib import MRIDataset
from utils.train_utils import retrieve_folders_list, Kfold_split
from utils.dataset_utils import get_val_transformations
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from model import NACLitModel

import logging
import wandb
logging.basicConfig(level=logging.INFO)

torch.manual_seed(42)
np.random.seed(42)

class ImageVisualizer:

    def __init__(self, architecture:str, fold:int=1, stage:int=1):
        self.architecture = architecture
        self.fold = fold
        self.exp = "colorization" if stage == 1 else "all"
        print(f"Model used will be {self.architecture} - experiment {self.exp}, with colorization pattern from fold {self.fold}.")
        self._set_model_from_ckpt()

        # empty attributes at beginning
        self.all_labels=None
        self.predicted_classes=None
        self.normalized_original_images = None
        self.unnormalized_orginal_images = None
        self.colorized_images = None
        self.wrong_pred_idxs = None
        self.right_pred_idxs = None
    
    def set_dataset(self, folders_list):
        transformations = get_val_transformations()
        self.input_dataset = MRIDataset(folders_list, 3, transform=transformations, preprocess_type="12bit")
        self.visualize_dataset = MRIDataset(folders_list, 3, transform=None, preprocess_type="10bit") #10bit è quella che fa vedere meglio il segnale senza alterare (come invece fa il minmax)

    def _set_model_from_ckpt(self):
        #
        root = "./checkpoints"
        ckpt_path = os.path.join(root, self.architecture,f"Fold_{str(self.fold)}",f"ckpt_{self.exp}_12bit.ckpt")
        #print(ckpt_path)
        try:
            self.model = NACLitModel.load_from_checkpoint(checkpoint_path=ckpt_path, architecture=self.architecture, exp_name="evaluation", colorize=True)
        except Exception as e:
            print(f"Something went wrong when loading pre-trained model: {e}")
    
    def colorize_and_predict_whole_dataset(self):
        """ Passa al modello tutto il dataset, restituisce immagini originali normalizzate (non utili alla viz), non normalizzate, colorizzate e le predizioni """
        self.model.eval()
        all_images= torch.Tensor()
        all_unnorm_images= torch.Tensor()
        all_labels= torch.Tensor()

        for index in range(0, len(self.input_dataset)):
            images, label = self.input_dataset.__getitem__(index)
            unnormalized_images, _ = self.visualize_dataset.__getitem__(index)
            all_images = torch.cat((all_images, images.unsqueeze(0)))
            all_unnorm_images = torch.cat((all_unnorm_images, unnormalized_images.unsqueeze(0)))
            all_labels = torch.cat((all_labels, label))

        self.all_labels = all_labels.argmax(dim=1)

        with torch.no_grad():
            all_images = all_images.permute(dims=(1,0, *range(2, all_images.dim())))
            colorized, preds = self.model(all_images)
            predicted_class = preds['pCR'].argmax(dim=1)

        self.normalized_original_images = all_images
        self.unnormalized_orginal_images = all_unnorm_images.permute(dims=(0,1,2,4,3))
        self.colorized_images = colorized.permute(dims=(1,0,2,4,3))
        print(f"Line 81 {self.colorized_images.shape}")
        self.predicted_classes = predicted_class 
        self._compare_predictions_and_labels()

    def _compare_predictions_and_labels(self):
        """Setta due nuovi attribute: liste con indici di predizioni corrette e sbagliate """
        self.wrong_pred_idxs = np.where(self.all_labels != self.predicted_classes)[0]
        self.right_pred_idxs = np.where(self.all_labels == self.predicted_classes)[0]

    def get_wrong_and_correct_predictions_index(self):
        """Ritorna lista con indici degli ERRORI e indici di quelle giuste"""
        return self.wrong_pred_idxs, self.right_pred_idxs

    def get_original_and_colorized_slice(self, slice_idx):
        print(self.unnormalized_orginal_images.shape)
        original = self.unnormalized_orginal_images[slice_idx]
        colorized = self.colorized_images[slice_idx]
        return original, colorized 
    
    def get_predicted_classes(self):
        return self.predicted_classes
    
    def get_labels(self):
        return self.all_labels
    

def save_images(images, output_name):
    grid = torchvision.utils.make_grid(images.view(-1,3,224,224))
    ndarr = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(10,10))
    plt.imshow(ndarr)
    plt.axis('off')  # Turn off axis
    plt.savefig(output_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_model_from_ckpt(architecture, fold:int=1, stage:int=1):
    root = "./checkpoints"
    exp = "colorization" if stage == 1 else "all"
    ckpt_path = os.path.join(root,architecture,f"Fold_{str(fold)}",f"ckpt_{exp}_12bit.ckpt")
    print(ckpt_path)
    litmodel = NACLitModel.load_from_checkpoint(checkpoint_path=ckpt_path, architecture=architecture, exp_name="evaluation", colorize=True)
    return litmodel
    #litmodel.eval()

def colorize_and_predict_whole_dataset(model, dataset):
    """ Passa al modello tutto il dataset, restituisce immagini originali normalizzate (non utili alla viz), non normalizzate, colorizzate e le predizioni """
    model.eval()

    all_images= torch.Tensor()
    all_unnorm_images= torch.Tensor()
    all_labels= torch.Tensor()

    for index in range(0, len(dataset)):
        images, label = dataset.__getitem__(index)
        dataset_unnormalized= dataset ###
        unnormalized_images, _ = dataset_unnormalized.__getitem__(index)
        all_images = torch.cat((all_images, images.unsqueeze(0)))
        all_unnorm_images = torch.cat((all_unnorm_images, unnormalized_images.unsqueeze(0)))
        all_labels = torch.cat((all_labels, label))

    all_labels = all_labels.argmax(dim=1)

    with torch.no_grad():
        all_images = all_images.permute(dims=(1,0, *range(2, all_images.dim())))
        colorized, preds = model(all_images)
        predicted_class = preds['pCR'].argmax(dim=1)

    return all_images, all_unnorm_images, colorized, predicted_class

def colorize_and_predict_slices_list(model, dataset, list_of_index: list[int]):
    for idx in list_of_index:
        images, unnormalized_images, colorized, predicted_class = colorize_and_predict_single_slice(model, dataset, idx)
        #concatenale e dalle in input

def print_slices_list(original_images, colorized, list_of_index):
    """Data una lista di indici stampa gli originali e le colorizzate"""
    pass


def colorize_and_predict_single_slice(model, dataset, index):
    """ Passa al modello un'unica slice (con tutte le modalità), restituisce immagini originali normalizzate (non utili alla viz), non normalizzate, colorizzate e le predizioni """
    images, label = dataset.__getitem__(index)

    dataset_unnormalized= dataset ###
    unnormalized_images, _ = dataset_unnormalized.__getitem__(index)
    images = images.unsqueeze(0)
    label = label.argmax(dim=1)
    model.eval()
    with torch.no_grad():
        images = images.permute(dims=(1,0, *range(2, images.dim())))
        colorized, preds = model(images)
        predicted_class = preds['pCR'].argmax(dim=1)
    return images, unnormalized_images, colorized, predicted_class

def split_pre_and_post_NAC_for_visualize(one_slice_all_modalities, architecture="monobranch"):
    """ """
    if architecture=="monobranch":
        DWI_pre = one_slice_all_modalities[0]
        T2_pre = one_slice_all_modalities[2]
        DCEt0_pre = one_slice_all_modalities[6][0:1,:,:].repeat(3,1,1)
        DCEt1_pre = one_slice_all_modalities[6][1:2,:,:].repeat(3,1,1)
        DCEt2_pre = one_slice_all_modalities[6][2:3,:,:].repeat(3,1,1)

        DWI_post = one_slice_all_modalities[1]
        T2_post = one_slice_all_modalities[3]
        DCEt0_post = one_slice_all_modalities[7][0:1,:,:].repeat(3,1,1)
        DCEt1_post = one_slice_all_modalities[7][1:2,:,:].repeat(3,1,1)
        DCEt2_post = one_slice_all_modalities[7][2:3,:,:].repeat(3,1,1)

        pre = torch.cat((DWI_pre.unsqueeze(0),T2_pre.unsqueeze(0),DCEt0_pre.unsqueeze(0),DCEt1_pre.unsqueeze(0),DCEt2_pre.unsqueeze(0)))
        post = torch.cat((DWI_post.unsqueeze(0),T2_post.unsqueeze(0),DCEt0_post.unsqueeze(0),DCEt1_post.unsqueeze(0),DCEt2_post.unsqueeze(0)))

    else: #multi
        DWI_pre = one_slice_all_modalities[0]
        T2_pre = one_slice_all_modalities[2]
        DCEpeak_pre = one_slice_all_modalities[4]
        DCE3TP_pre = one_slice_all_modalities[6]

        DWI_post = one_slice_all_modalities[1]
        T2_post = one_slice_all_modalities[3]
        DCEpeak_post = one_slice_all_modalities[5]
        DCE3TP_post = one_slice_all_modalities[7]

        pre = torch.cat((DWI_pre.unsqueeze(0),T2_pre.unsqueeze(0),DCEpeak_pre.unsqueeze(0),DCE3TP_pre.unsqueeze(0)))
        post = torch.cat((DWI_post.unsqueeze(0),T2_post.unsqueeze(0),DCEpeak_post.unsqueeze(0),DCE3TP_post.unsqueeze(0)))

    return pre, post

def plot_grid(images_tensor, caption="", architecture="monobranch"):
    """ Per plottare la griglia con le 5 scan per la slice (singola) passata in input - solo per il monobranch"""
 
    grid_img = torchvision.utils.make_grid(images_tensor, )#nrow=5)

    # Convert the grid image to numpy for plotting
    np_img = grid_img.numpy()

    plt.figure(figsize=(15, 15))  # Adjust the figsize as needed
    labels=["DWI", "T2", "DCE t0", "DCE t1", "DCE t2"] if architecture=="monobranch" else ["DWI", "T2", "DCE peak", "DCE 3TP"]
    for i in range(len(labels)):
        # Calculate the position to place the label
        row = i // len(labels)
        col = i % len(labels)
        plt.text(col * 224 + 10, row * 224 + 10, labels[i], color='red', fontsize=8, weight='bold')
    # Plot the grid
    
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # Convert from (C, H, W) to (H, W, C)
    for i in range(len(labels)):
        # Calculate the position to place the label
        row = i // 5
        col = i % 5
        plt.text(col * 224 + 10, row * 224 + 10, labels[i], color='red', fontsize=8, weight='bold')
    plt.axis('off')  # Turn off the axis
    plt.suptitle(caption)
    plt.show()

def plot_single_image(image_tensor, caption=""):
    np_img = image_tensor.numpy()
    plt.figure(figsize=(15, 15))  # Adjust the figsize as needed
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # Convert from (C, H, W) to (H, W, C)
    plt.axis('off')  # Turn off the axis
    plt.suptitle(caption)
    plt.show()

if __name__ == "__main__":
    full_path_to_dataset = "C:\\Users\\c.navilli\\Desktop\\Prova\\dataset_mini"
    print(f"Choose Architecture ['monobranch', 'multibranch']")
    ARCHITECTURE = input()
    print(f"Choose stage: [1,2]")
    STAGE = eval(input())
    print(f"Choosse fold: [1, 2, 3, 4]")
    fold_idx = eval(input())
    print(f"Print single original images? [0: no, 1:yes]")
    print_single_image_original = eval(input())
    print(f"Print single colorized images? [0: no, 1:yes]")
    print_single_image_colorized = eval(input())

    folders_list = retrieve_folders_list(full_path_to_dataset)
    datasets_list = Kfold_split(folders_list, 1)
    transformations = get_val_transformations()

    image_visualizer = ImageVisualizer(architecture=ARCHITECTURE,fold=fold_idx, stage=STAGE)
    image_visualizer.set_dataset(datasets_list[0][0])
    image_visualizer.colorize_and_predict_whole_dataset()

    # Check sugli errori ----------------------------------------------------------------
    wrong_idxs, correct_idxs = image_visualizer.get_wrong_and_correct_predictions_index()
    true_labels = image_visualizer.get_labels()
    wrong_labels = true_labels[wrong_idxs]
    correct_labels = true_labels[correct_idxs]
    print(f"Indici delle slice predette correttamente: {correct_idxs}")
    print(f"Indici delle slice predette erroneamente: {wrong_idxs}")
    print(f"Labels predette correttamente: {correct_labels}")
    print(f"Labels non predette correttamente: {wrong_labels}")

    # Print di una slice alla volta in loop  -----------------------------------------------------------------
    idx_list = [10]
    if ARCHITECTURE=="monobranch":
        for slice_idx in idx_list:
            original_image, colorized_images = image_visualizer.get_original_and_colorized_slice(slice_idx= slice_idx) #la 10 ok
            print(original_image.shape)
            pre_nac, post_nac = split_pre_and_post_NAC_for_visualize(original_image)
            pre_nac_colorized, post_nac_colorized = colorized_images[0], colorized_images[1]

            pre_grid = torchvision.utils.make_grid(pre_nac.view(-1,3,224,224), padding=2, pad_value=1)
            print(f"La grid ha shape {pre_grid.shape}")
            plot_grid(pre_grid, "Pre-NAC original images")
            print(pre_nac_colorized.shape)
            plot_single_image(pre_nac_colorized, "Pre-NAC global colored image")

            post_grid = torchvision.utils.make_grid(post_nac.view(-1,3,224,224), padding=2, pad_value=1)
            plot_grid(post_grid, "Pre-NAC original images")
            plot_single_image(post_nac_colorized, "Pre-NAC global colored image")
    else:
        for slice_idx in idx_list:
            # Test sul multibranch
            original_image, colorized_images = image_visualizer.get_original_and_colorized_slice(slice_idx= slice_idx) #ho due tensori da 8 immagini ciascuna
            pre_nac, post_nac = split_pre_and_post_NAC_for_visualize(original_image, "multibranch")
            colorized_pre_nac, colorized_post_nac = split_pre_and_post_NAC_for_visualize(colorized_images, "multibranch")
            
            if print_single_image_original:
                for img in range(pre_nac.shape[0]):
                    plot_single_image(pre_nac[img], "")
                for img in range(post_nac.shape[0]):
                    plot_single_image(post_nac[img], "")
            if print_single_image_colorized:
                for img in range(colorized_pre_nac.shape[0]):
                    plot_single_image(colorized_pre_nac[img], "")
                for img in range(colorized_post_nac.shape[0]):
                    plot_single_image(colorized_post_nac[img], "")
            # Pre-Nac
            plot_grid(pre_nac, "Grayscale pre-NAC scans, Multibranch architecture", architecture="multibranch")
            plot_grid(colorized_pre_nac, "Colorized pre-NAC scans, Multibranch architecture", architecture="multibranch")
            # Post-Nac
            plot_grid(post_nac, "Grayscale post-NAC scans, Multibranch architecture", architecture="multibranch")
            plot_grid(colorized_post_nac, "Colorized post-NAC scans, Multibranch architecture", architecture="multibranch")
