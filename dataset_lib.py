import os
import ast
import SimpleITK as sitk
import numpy as np
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch 

from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
import lightning.pytorch as pl

from utils.dataset_utils import get_transformations_COPY, get_val_transformations, normalize_slice_channelwise, normalize_slices

import logging

branches_list = ["DWI","T2","DCE_peak","DCE_3TP"]
scans = ["DWI","T2","DCE"]

class MRIDataset(Dataset):

    def __init__(self, folders_list=None, slices = 3, transform = None, preprocess_type = ""):
        
        self.extra_slices_num = slices
        self.branches_list = branches_list
        self.scans = scans
        self.transform = transform
        self.preprocess_type = preprocess_type

        print("1. DEFINING SUBSEQUENCES")
        features, labels, scan_list, extra_slices_per_side, patient_list = self._define_subsequences(folders_list)
        
        print("\n2. DEFINING INPUTS")
        X,Y = self._define_input(patient_list, features, labels)
        X = X.permute(dims=(1,0, *range(2, X.dim()))) # [SLICES, MODALITIES, C, H, W]
        
        self.scan_list = scan_list
        self.slices = X
        self.labels = Y.float()

        #print(self.class_weights)
        
    def __getitem__(self, index) -> (torch.Tensor):#, torch.Tensor):
        """
        Return the processed slice corresponding to the input index.
        Params:
            index(int): index of the slice to be returned
        Returns:
            img(torch.Tensor): a tensor containing a 3D tensor for each branch of the model.
            label(torch.Tensor): a tensor containing a 1D tensor for each branch of the model.
        """

        images = self.slices[index]
        labels = self.labels[index]

        if index==0 or index==2:
            print(images)

        if self.transform is not None:
            images = self.transform(images) 
            print("Normalize post transformation")
            images = normalize_slices(images)
            
        return images, labels
    
    def __len__(self):
        """
        Return the length of the dataset, i.e. the number of slices.
        """
        return self.slices.shape[0]
    
    def _retrieve_folders_list(self, root_dir:str) -> "list[str]":
        """
        Scan the root dir and retrieve all the individual folders containing the DICOM images.
        Params:
            root_dir(str): the path of the root directory.
        Returns:
            train_folders(list[str]): list with the paths of the folders containing a sequence each.
        """
        train_folders=[]
        for root, dirs, files in os.walk(root_dir):
            dirs.sort()
            for folder in dirs:
                if folder[:3] != 'NAC' and folder != 'MRI_1' and folder != 'MRI_2':  #excluding all the repos that are not the final ones
                    train_folders.append(os.path.join(root,folder))
        return train_folders

    def _preprocess_img(self, img):
        """
        This method handles 16bit images and convert them into a proper range for torchvision transform, tailored on 8bit images.
        """
        preprocess = self.preprocess_type

        slice = sitk.GetArrayFromImage(img)
        #slice.astype(np.float32)
        
        print(f"ORIGINAL VALUES FOR SLICE --- MIN: {slice.min()} - MAX: {slice.max()}")
        if preprocess=="min_max":
            slice = slice/slice.max()
        elif preprocess=="16bit":
            slice = slice/(2**16-1)
        elif preprocess=="10bit":
            slice = slice/(2**10-1)
            slice[slice>1.]=1.
        elif preprocess=="12bit":
            slice = slice/(2**12-1)
            slice[slice>1.]=1.
        elif preprocess=="norm_and_scale":
            slice = (slice-slice.mean())/slice.std()
            slice = (slice - slice.min())/(slice.max()-slice.min())
        elif preprocess=="norm":
            slice = (slice-slice.mean())/slice.std()
        elif preprocess=="to_int16":  #malissimo ---> tutte saturate
            slice = slice.astype(np.int16)
        elif preprocess=="to_uint8":  #malissimo ---> tutte saturate
            slice = slice.astype(np.float32) * 255. / slice.max()
            slice = slice.astype(np.uint8)
        else:
            pass #questo ritornerà un uint16, che non è supportato da pytorch
        print(f"\tPOST VALUES FOR SLICE --- MIN: {slice.min()} - MAX: {slice.max()}\n")
        
        print(f"Type: {slice.dtype}")
        return  slice
    
    def _define_subsequences(self, folders: "list[str]"):
        """
            sequences è la lista di folders ottenute prima 

        define_subsequences first reads the ground_truth (pCR result: 0 or 1) from a .txt file then operates for each sequences:
            - .dcm and .IMA images are read
            - from the dicom metadata, information like Patient, MRI (1 or 2), sequence type are retrieved
            - the metadata SeriesDescription is used to locate the index slice
            - the metadata InstanceNumber is used to properly select extra slices according to the "--slices" param and the index slice
        labels are added once for each patient (each sequence from the same patient has the same result) and the images (read with simpleITK) are appended to the corresponding sub_sequence array
        """

        sequence_count = 0 
        index_count = 0 
        image_count = 0

        new_patient = ""

        features = []
        labels = []
        scan_list = []
        paz_list = []
        DWI_1, DWI_2, T2_1, T2_2, T1_1, T1_2 = ([] for i in range(6)) 
        #sub_sequences = [DWI_1, DWI_2, T2_1, T2_2, T1_1, T1_2] #lista di liste

        sub_sequences_dict = {"DWI_1": DWI_1, "DWI_2": DWI_2, "T2_1": T2_1, "T2_2": T2_2, "T1_1": T1_1, "T1_2": T1_2}

        #labels_file = "/home/cnavilli/tesi_nac/labels/pCR.txt"
        labels_file = "labels/pCR.txt"

        try:
            with open(labels_file, "r") as file:
                contents = file.read()
                dict = ast.literal_eval(contents)
        except Exception as error:
            print("Cannot find file in labels/ folder, please put the file pCR.txt inside \n")
            if not os.path.exists("labels"):
                os.mkdir("labels")
            exit()
        
        for sequence in folders:
            sequence_count += 1       
            extra_slices_per_side = self.extra_slices_num
            patient_flag = False      
            dicom_files = []          
            images = []               
            slices = []               
 
            for r, d, f in os.walk(sequence):
                f.sort()
                for file in f:
                    if '.dcm' in file or '.IMA' in file:
                        dicom_files.append(os.path.abspath(os.path.join(r,file)))
                        
            reader = sitk.ImageFileReader()
            reader.SetFileName(dicom_files[0])
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()

            description = reader.GetMetaData("0008|103e").strip()
            name_num = description.split('_')[0] 
            if name_num != new_patient:    
                new_patient = name_num
                patient_flag = True
            mri_string = description.split('_')[1]      
            scan_string = description.split('_')[2]    

            if scan_string in self.scans:
                for dicom_file in dicom_files:
                    images.append(sitk.ReadImage(dicom_file)) 
                slices_num = len(images)
                max_slices = int(((slices_num-1)/2))
                if extra_slices_per_side > max_slices:
                    extra_slices_per_side = max_slices   

                max_instance = - np.inf
                min_instance = np.inf
                for img in images:  
                    img_instance = int(img.GetMetaData('0020|0013'))
                    if img_instance > max_instance:
                        max_instance = img_instance
                    if img_instance < min_instance:
                        min_instance = img_instance

                for img in images:
                    if img.GetMetaData('0008|103e') == "IndexSlice":  
                        index_count += 1
                        image_count += 1
                        index_instance = int(img.GetMetaData('0020|0013'))

                        upper_bound = min(max_instance, (index_instance + extra_slices_per_side))
                        lower_bound = max(min_instance, (index_instance - extra_slices_per_side))

                        if (index_instance + extra_slices_per_side) > max_instance:
                            diff_bound = extra_slices_per_side - (max_instance - index_instance)
                            lower_bound = max(min_instance, (index_instance - extra_slices_per_side - diff_bound))

                        if (index_instance - extra_slices_per_side) < 0:
                            diff_bound = extra_slices_per_side - (index_instance)
                            upper_bound = min(max_instance, (index_instance + extra_slices_per_side + diff_bound))

                #print(f"Per questa sequenza i bound sono: LB {lower_bound} - INDEX {index_instance} - UB {upper_bound}") 
                for img in images:
                    img_instance = int(img.GetMetaData('0020|0013'))
                    if img_instance >= index_instance and img_instance <= upper_bound:
                        image_count += 1
                        if scan_string != "DCE":   
                            #float_slice = sitk.GetArrayFromImage(img)
                            slices.append(self._preprocess_img(img))
                        else:
                            slices.append(img)

                    if img_instance < index_instance and img_instance >= lower_bound:
                        image_count += 1
                        if scan_string != "DCE":
                            #float_slice = sitk.GetArrayFromImage(img)
                            slices.append(self._preprocess_img(img))
                        else:
                            slices.append(img)

            name_string = "NAC_" + name_num
            if name_string in dict:     
                label = list(str(dict[name_string]))  
            
            if patient_flag:         
                for i in range(len(slices)):
                    labels.append(label)            #la stessa label è ripetuta per il numero di fette considerate
                    scan_list.append(name_string)   #nella lista delle scansioni aggiungo "NAC_1" ecc ecc
                    paz_list.append(name_num)  
            
            if "DWI" in self.branches_list:  
                if scan_string == "DWI":       
                    if mri_string == "MRI1":   
                        DWI_1.extend(slices)   
                    else:                      
                        DWI_2.extend(slices)
                                                                    
            if "T2" in self.branches_list: 
                if scan_string == "T2":
                    if mri_string == "MRI1":
                        T2_1.extend(slices)
                    else:
                        T2_2.extend(slices)

            if "DCE_peak" in self.branches_list or "DCE_3TP" in self.branches_list:
                if scan_string == "DCE":
                    if mri_string == "MRI1":
                        T1_1.extend(slices)
                    else:
                        T1_2.extend(slices)

        lista_dei_tagli = []
        for name,sub_sequence in sub_sequences_dict.items():  
            lista_dei_tagli.append(name)
            features.append(sub_sequence)

        sub_sequence_div = (10/len(self.branches_list))/2 
        #print("Fold:", fold+1)              
        print("Total DICOM Sequences:", round(sequence_count/sub_sequence_div))
        print("Total DICOM Index Slices:", round(index_count/sub_sequence_div))
        print("Total DICOM Selected Slices:", round(image_count/sub_sequence_div), "\n")

        # Adding the class_weight to account for the class unbalance
        class_weights_dict = self._class_balance(np.array(labels))
        print(f"Class weight dictionary: {class_weights_dict}")
        self.class_weights_tensor = torch.tensor([class_weights_dict[key] for key in sorted(class_weights_dict.keys())], dtype=torch.float32)   
            
        labels = np.array(labels, dtype=int)
        labels = torch.from_numpy(labels).to(torch.float32) 
        print(f"Labels dtype: {labels.dtype}")
        print(f"Channels: {lista_dei_tagli}")
        print(f"Scan list: {scan_list}")
        print(f"Extra slice considered: {extra_slices_per_side} - total subsequence lenght: {extra_slices_per_side*2+1}")

        return features, labels, scan_list, extra_slices_per_side, paz_list           

    def _class_balance(self, labels_array: np.array):
        """
        Count the frequences of the two classes.
        Parameters:
        - labels_array(np.array): array with the labels
        Returns:
        - dictionary with the count of the occurrencies for each label
        """
        counter_0 = 0
        counter_1 = 0
        k = self.extra_slices_num
        for i in range(labels_array.shape[0]):
            if labels_array[i][0]=='0':            
                counter_0+=1
            else:
                counter_1+=1
        return {'0':counter_0/(k*2+1), '1':counter_1/(k*2+1)}
    
    def _define_DCE(self, T1_1, T1_2):
        new_patient = ""
        DCE_dict = {}
        branches_list =self.branches_list 

        features = []
        labels = []

        DCE_peak1, DCE_peak2, DCE_3TP1, DCE_3TP2 = ([] for i in range(4))
        sub_sequences = [DCE_peak1, DCE_peak2, DCE_3TP1, DCE_3TP2]

        k = self.extra_slices_num
        slices = (k*2 + 1) 

        for img in T1_1:
            patient = img.GetMetaData("0020|000d").strip()
            if patient != new_patient:
                new_patient = patient

                new_time = ""
                time_array = []

                DCE_dict[patient] = {}

            time = img.GetMetaData("0008|0031").strip()
            if time != new_time:
                new_time = time
                time_array.append(time)
                DCE_dict[patient] = time_array

        for patient in DCE_dict:
            patient_time = list(DCE_dict[patient])
            patient_time = np.array(patient_time)
            patient_time = np.sort(patient_time)

            DCE_pre = patient_time[0]
            DCE_peak = patient_time[1]
            DCE_post = patient_time[2]

            pre_array = []
            peak_array = []
            post_array = []
            DCE_count = 0

            for img in T1_1:
                patient_name = img.GetMetaData("0020|000d").strip()
                time = img.GetMetaData("0008|0031").strip()

                if "DCE_peak" in self.branches_list:
                    if patient_name == patient and time == DCE_peak:
                        DCE_peak1.append(self._preprocess_img(img))

                if "DCE_3TP" in self.branches_list:
                    if patient_name == patient:
                        if time == DCE_pre:
                            pre_array.append(self._preprocess_img(img))
                            DCE_count += 1
                        elif time == DCE_peak:
                            peak_array.append(self._preprocess_img(img))
                            DCE_count += 1
                        else:
                            post_array.append(self._preprocess_img(img))
                            DCE_count += 1
                        if DCE_count == slices*3:
                            for i in range(slices):
                                image_3TP = np.vstack((pre_array[i], peak_array[i], post_array[i]))
                                DCE_3TP1.append(image_3TP)

        for img in T1_2:
            patient = img.GetMetaData("0020|000d").strip()
            if patient != new_patient:
                new_patient = patient

                new_time = ""
                time_array = []

                DCE_dict[patient] = {}

            time = img.GetMetaData("0008|0031").strip()
            if time != new_time:
                new_time = time
                time_array.append(time)
                DCE_dict[patient] = time_array

        for patient in DCE_dict:
            patient_time = list(DCE_dict[patient])
            patient_time = np.array(patient_time)
            patient_time = np.sort(patient_time)

            DCE_pre = patient_time[0]
            DCE_peak = patient_time[1]
            DCE_post = patient_time[2]

            pre_array = []
            peak_array = []
            post_array = []
            DCE_count = 0

            for img in T1_2:
                patient_name = img.GetMetaData("0020|000d").strip()
                time = img.GetMetaData("0008|0031").strip()

                if "DCE_peak" in self.branches_list:
                    if patient_name == patient and time == DCE_peak:
                        DCE_peak2.append(self._preprocess_img(img))

                if "DCE_3TP" in self.branches_list:
                    if patient_name == patient:
                        if time == DCE_pre:
                            pre_array.append(self._preprocess_img(img))
                            DCE_count += 1
                        elif time == DCE_peak:
                            peak_array.append(self._preprocess_img(img))
                            DCE_count += 1
                        else:
                            post_array.append(self._preprocess_img(img))
                            DCE_count += 1
                        if DCE_count == slices*3:
                            for i in range(slices):
                                image_3TP = np.vstack((pre_array[i], peak_array[i], post_array[i]))
                                DCE_3TP2.append(image_3TP)

        for sub_sequence in sub_sequences:
            features.append(np.array(sub_sequence))
        return features

    def _define_input(self, patient_list, features, labels):
        branches_list = self.branches_list 
        if "DCE_peak" or "DCE_3TP" in branches_list:
            DCE_features = self._define_DCE(features[-2], features[-1])
            features = features[:len(features)-2]
            for DCE_feature in DCE_features:
                features.append(DCE_feature)
        print(f"CI SONO {len(features)} MODALITA''''''")

        X = torch.Tensor()
        modalities = {0: 'DWI pre-NAC',
                      1: 'DWI post-NAC',
                      2: 'T2 pre-NAC',
                      3: 'T2 pre-NAC',
                      4: 'DCE_peak pre-NAC',
                      5: 'DCE_peak pre-NAC',
                      6: 'DCE_3TP pre-NAC',
                      7: 'DCE_3TP pre-NAC'}
        
        for i,feature in enumerate(features): 
            print(f"{modalities[i]}:") #TODO da rimuovere
            feature = torch.tensor(feature)
            if feature.any():
                X_sub = feature
                if feature.shape[1] == 1:
                    X_sub = torch.repeat_interleave(X_sub,repeats = 3, dim=1)
                X_sub = X_sub.to(torch.float32).unsqueeze(0) 
                X = torch.cat([X, X_sub], dim = 0)
                print(f"Shape check - line 459 in define_input: {X.shape}")

        Y = F.one_hot(labels.to(torch.int64), 2) #config.NB_CLASSES)

        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        return X, Y
    

    
class MRIDataModule(pl.LightningDataModule):

    def __init__(self, training_folders, validation_folders, slices, batch_size, num_workers = 0):
        super().__init__()
        #self.training_folders = training_folders
        #self.validation_folders = validation_folders
        #self.slices = slices
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transformations = get_transformations_COPY()
        self.val_transformations = get_val_transformations()

        self.train_dataset = MRIDataset(training_folders, slices, transform = self.val_transformations)
        self.class_weights = self.train_dataset.class_weights_tensor    #class weight is computed only over the training dataset, and is used in the computation of the training loss ONLY
        self.validation_dataset = MRIDataset(validation_folders, slices, transform=self.val_transformations)
        self.test_dataset = self.validation_dataset

    # def setup(self, stage):
    #     self.train_dataset = MRIDataset(self.training_folders, self.slices, transform = self.transformations)
    #     self.class_weights = self.train_dataset.class_weights_tensor    #class weight is computed only over the training dataset, and is used in the computation of the training loss ONLY
    #     self.validation_dataset = MRIDataset(self.validation_folders, self.slices)
    #     self.test_dataset = self.validation_dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
