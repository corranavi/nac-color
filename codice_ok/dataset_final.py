import os
import ast
import SimpleITK as sitk
import numpy as np
import torch 

from torch.nn import functional as F
from torch.utils.data import Dataset

from utils.dataset_utils import normalize_slice_channelwise, apply_same_transformation_all_modalities

import logging

class MRIDataset(Dataset):

    def __init__(self, folders_list=None, transform = None):
        self.transform = transform

        print("1. DEFINING SUBSEQUENCES")
        features, labels, scan_list, extra_slices_per_side, patient_list = self._define_subsequences(folders_list)
        
        print("\n2. DEFINING INPUTS")
        X,Y = self._define_input(patient_list, features, labels)

        print("\n3. REARRANGE INPUTS")
        X,Y = self._new_rearrange_feature_list(X,Y, extra_slices_per_side)
        #Y = self._rearrange_labels_list(Y, extra_slices_per_side)
        
        self.patients = self._rearrange_patients_list(patient_list, extra_slices_per_side)
        self.patient_sequences = X
        self.labels = Y
        
    def __getitem__(self, index) -> (torch.Tensor):#, torch.Tensor):
        """
        Return the processed slice corresponding to the input index.
        Params:
            index(int): index of the slice to be returned
        Returns:
            img(torch.Tensor): a tensor containing a 3D tensor for each branch of the model.
            label(torch.Tensor): a tensor containing a 1D tensor for each branch of the model.
        """

        patient_sequences = self.patient_sequences[index]
        patient = self.patients[index]
        patient_labels = self.labels[index]

        if self.transform is not None:
            patient_sequences = apply_same_transformation_all_modalities(patient_sequences, self.transform)
            
        return patient, patient_sequences, patient_labels
    
    def __len__(self):
        """
        Return the length of the dataset, i.e. the number of slices.
        """
        return self.patient_sequences.shape[0]
    
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

        sub_sequences_dict = {"DWI_1":DWI_1,"DWI_2": DWI_2, "T2_1": T2_1,"T2_2": T2_2,"T1_1": T1_1,"T1_2": T1_2}

        try:
            with open("labels/pCR.txt", "r") as file:
                contents = file.read()
                dict = ast.literal_eval(contents)
        except Exception as error:
            print("Cannot find file in labels/ folder, please put the file pCR.txt inside \n")
            if not os.path.exists("labels"):
                os.mkdir("labels")
            exit()
        
        for sequence in folders:
            sequence_count += 1       
            extra_slices_per_side = 4 #config.SLICES --
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

            if scan_string in ["DWI","T2","DCE"]: # in realtà questi tagli sono contenuti in config.scans:
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

                print(f"Per questa sequenza i bound sono: LB {lower_bound} - INDEX {index_instance} - UB {upper_bound}")    
                for img in images:
                    img_instance = int(img.GetMetaData('0020|0013'))
                    if img_instance >= index_instance and img_instance <= upper_bound:
                        image_count += 1
                        if scan_string != "DCE":   
                            float_slice = sitk.GetArrayFromImage(img)
                            slices.append(float_slice.astype(np.float32))
                        else:
                            slices.append(img)

                    if img_instance < index_instance and img_instance >= lower_bound:
                        image_count += 1
                        if scan_string != "DCE":
                            float_slice = sitk.GetArrayFromImage(img)
                            slices.append(float_slice.astype(np.float32))
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

            branches_list =["DWI","T2","DCE_peak","DCE_3TP"]
            
            if "DWI" in branches_list: #config.branches_list:  
                if scan_string == "DWI":       
                    if mri_string == "MRI1":   
                        DWI_1.extend(slices)   
                    else:                      
                        DWI_2.extend(slices)
                                                                    
            if "T2" in branches_list: #config.branches_list:
                if scan_string == "T2":
                    if mri_string == "MRI1":
                        T2_1.extend(slices)
                    else:
                        T2_2.extend(slices)

            if "DCE_peak" in branches_list or "DCE_3TP" in branches_list:
                if scan_string == "DCE":
                    if mri_string == "MRI1":
                        T1_1.extend(slices)
                    else:
                        T1_2.extend(slices)

        lista_dei_tagli = []
        for name,sub_sequence in sub_sequences_dict.items():  
            lista_dei_tagli.append(name)
            features.append(sub_sequence)

        sub_sequence_div = (10/len(branches_list))/2   #config.branches_list
        #print("Fold:", fold+1)              
        print("Total DICOM Sequences:", round(sequence_count/sub_sequence_div))
        print("Total DICOM Index Slices:", round(index_count/sub_sequence_div))
        print("Total DICOM Selected Slices:", round(image_count/sub_sequence_div), "\n")
                  
        print(self._class_balance(np.array(labels)))    
            
        labels = np.array(labels, dtype=int)
        labels = torch.from_numpy(labels).to(torch.float32) 
        print(labels.dtype)
        print(lista_dei_tagli,labels, scan_list, extra_slices_per_side)
        return features, labels, scan_list, extra_slices_per_side, paz_list           #scan_list = NAC_1, NAC_1,....per ogni foto
    
    def _normalize_slice(self, slice: torch.Tensor)->torch.Tensor:
        # TODO: Implementare il flusso per rimuovere da ogni slice la mean e std per canale
        #mean = calcola_la_media
        #std = calcola_la_std
        #slice = T.Normalize(mean = mean, std = std)(slice)

        return slice

    def _class_balance(self, labels_array: np.array)->dict:
        """
        Count the frequences of the two classes.
        Parameters:
        - labels_array(np.array): array with the labels
        Returns:
        - dictionary with the count of the occurrencies for each label
        """
        counter_0 = 0
        counter_1 = 0
        k = 4
        for i in range(labels_array.shape[0]):
            if labels_array[i][0]=='0':
                counter_0+=1
            else:
                counter_1+=1
        return {'0':counter_0/(k*2+1), '1':counter_1/(k*2+1)}  #k*2+1 tiene conto di quante slice ho per ciascun paziente. 

    def _define_DCE(self, T1_1, T1_2):
        new_patient = ""
        DCE_dict = {}
        branches_list =["DWI","T2","DCE_peak","DCE_3TP"] #da togliere poi

        features = []
        labels = []

        k = 4
        DCE_peak1, DCE_peak2, DCE_3TP1, DCE_3TP2 = ([] for i in range(4))
        sub_sequences = [DCE_peak1, DCE_peak2, DCE_3TP1, DCE_3TP2]
        slices = (k*2 + 1) #config.SLICES

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

                if "DCE_peak" in branches_list:
                    if patient_name == patient and time == DCE_peak:
                        DCE_peak1.append(sitk.GetArrayFromImage(img).astype(np.float32))

                if "DCE_3TP" in branches_list:
                    if patient_name == patient:
                        if time == DCE_pre:
                            pre_array.append(sitk.GetArrayFromImage(img).astype(np.float32))
                            DCE_count += 1
                        elif time == DCE_peak:
                            peak_array.append(sitk.GetArrayFromImage(img).astype(np.float32))
                            DCE_count += 1
                        else:
                            post_array.append(sitk.GetArrayFromImage(img).astype(np.float32))
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

                if "DCE_peak" in branches_list:
                    if patient_name == patient and time == DCE_peak:
                        DCE_peak2.append(sitk.GetArrayFromImage(img).astype(np.float32))

                if "DCE_3TP" in branches_list:
                    if patient_name == patient:
                        if time == DCE_pre:
                            pre_array.append(sitk.GetArrayFromImage(img).astype(np.float32))
                            DCE_count += 1
                        elif time == DCE_peak:
                            peak_array.append(sitk.GetArrayFromImage(img).astype(np.float32))
                            DCE_count += 1
                        else:
                            post_array.append(sitk.GetArrayFromImage(img).astype(np.float32))
                            DCE_count += 1
                        if DCE_count == slices*3:
                            for i in range(slices):
                                image_3TP = np.vstack((pre_array[i], peak_array[i], post_array[i]))
                                DCE_3TP2.append(image_3TP)

        for sub_sequence in sub_sequences:
            features.append(np.array(sub_sequence))
        return features

    def _define_input(self, patient_list, features, labels):
        branches_list =["DWI","T2","DCE_peak","DCE_3TP"] #new
        if "DCE_peak" or "DCE_3TP" in branches_list: #config.branches_list:
            DCE_features = self._define_DCE(features[-2], features[-1])
            features = features[:len(features)-2]
            for DCE_feature in DCE_features:
                features.append(DCE_feature)

        X = torch.Tensor()
        for feature in features: 
            feature = torch.tensor(feature)
            if feature.any():
                X_sub = feature
                if feature.shape[1] == 1:
                    X_sub = torch.repeat_interleave(X_sub,repeats = 3, dim=1)
                X_sub = X_sub.to(torch.float32).unsqueeze(0) 

                print(f"SHAPE {X_sub.shape}")
                #NEW normalizzazione -----------------------------------------
                X_sub_normalized = torch.Tensor()
                for slice in range(X_sub.shape[1]):
                    slice = normalize_slice_channelwise(X_sub[0][slice])
                    X_sub_normalized = torch.cat([X_sub_normalized, slice.unsqueeze(0)])
                
                X_sub_normalized = X_sub_normalized.unsqueeze(0)
                #X_sub = normalize_slice_channelwise(X_sub)
                #-------------------------------------------------------------

                print(f"X_sub ha dimensioni: {X_sub_normalized.shape}")
                X = torch.cat([X, X_sub_normalized], dim = 0)
                #X.append(X_sub)          #indice X[tipo_risonanza][numero_di_immagine][channels][height][weight]
        Y = F.one_hot(labels.to(torch.int64), 2) #config.NB_CLASSES)
        print(Y)
        print(f"Y shape: {Y.shape}")
        print(f"X shape: {X.shape}")
        return X, Y

    # def _rearrange_feature_list(self, features: "list[list[torch.Tensor]]", k:int)->torch.Tensor:
    #     """
    #     Given the previously extracted list of features, rearrange them so that the output is 
    #     a Torch Tensor whose first dimension is the one of the patients.
    #     Arguments:
    #         features (list[list[torch.Tensor]]): the list of processed images.
    #         k (int): the number of additional slices per side.
    #     Returns:
    #         full_stakced (torch.Tensor): a tensor of dimension [#patients x #modalities x #images x #channels x #height_pixels x #weight_pixels]
    #     """
    #     final_features_list = [ ]
    #     positions=[(0,1),(2,3),(4,5),(6,7)]
    #     for couple in positions:
    #         print(f"Pair: {couple}")
    #         pre_nac_list = features[couple[0]]
    #         post_nac_list = features[couple[1]]

    #         print(f"Lunghezza della pre_nac_list: {len(pre_nac_list)}")
    #         print(f"Lunghezza della post_nac_list: {len(post_nac_list)}")

    #         #prendi i primi 2k+1 da pre_nac e poi concatena i successivi 2k+1 da post_nac
    #         pre_nac_shaded_list = []
    #         for i in range(0,len(pre_nac_list), 2*k+1):
    #             individual_patient=[]
    #             for j in range(i,i+2*k+1):
    #                 individual_patient.append(pre_nac_list[j])
    #                 print(f"LEN INDIVIDUAL PATIENT: {len(individual_patient)}")
    #             pre_nac_shaded_list.append(individual_patient)
    #         print(f"PRE NAC shaded list ha lunghezza: {len(pre_nac_shaded_list)}")

    #         post_nac_shaded_list = []
    #         for i in range(0,len(post_nac_list), 2*k+1):
    #             individual_patient=[]
    #             for j in range(i,i+2*k+1):
    #                 individual_patient.append(post_nac_list[j])
    #             post_nac_shaded_list.append(individual_patient)
    #         print(len(post_nac_shaded_list))

    #         if len(pre_nac_shaded_list)!=len(post_nac_shaded_list):
    #             print("Le due liste non hanno la stessa lunghezza")
    #             exit()
            
    #         modality_joined = []

    #         print(f"Line 519 - {len(pre_nac_shaded_list)}")
    #         print(f"Line 520 - {len(post_nac_shaded_list)}")
    #         print(f"Il tipo è: {type(pre_nac_shaded_list)}")

    #         for patient_idx in range(len(post_nac_shaded_list)):
    #             print(f"Paziente: NAC_{patient_idx+1}")
    #             pre_nac_shaded_list[patient_idx].extend(post_nac_shaded_list[patient_idx])
    #             print(f"Lunghezza della lista di NAC_{patient_idx+1} post extend: {len(pre_nac_shaded_list[patient_idx])}")
    #             print("")
    #             modality_joined.append(pre_nac_shaded_list[patient_idx])

    #         final_features_list.append(modality_joined)

    #     print("Analisi delle dimensioni sulla final_features_list:")
    #     print(f"Mi aspetto siano 4: {len(final_features_list)}")

    #     print(f"La lista con tutte le features ha {len(final_features_list)} elementi. ")
    #     for i in range(len(final_features_list)):
    #         print(f"La modalità numero {i+1} ha {len(final_features_list[i])} liste, ovvero pazienti:")
    #         for j in range(len(final_features_list[i])):
    #             print(f"La sequenza {i+1} del paziente {j+1} ha {len(final_features_list[i][j])} slices.")
    #         print("")
    #     print("")

    #     patient_stacked_list = []    
    #     for patient_idx in range(len(final_features_list)):
    #         list_modalities = []
    #         for modality_idx in range(len(final_features_list[patient_idx])):
    #             sequence_tensor_list = final_features_list[patient_idx][modality_idx]
    #             list_modalities.append(torch.stack(sequence_tensor_list, dim=0))
    #         patient_stacked_list.append(torch.stack(list_modalities, dim=0))
    #     full_stacked = torch.stack(patient_stacked_list, dim=0)
    #     full_stacked = full_stacked.transpose(0,1)
    #     return full_stacked

    def _new_rearrange_feature_list(self, X: "list[list[torch.Tensor]]", Y,  k:int)->torch.Tensor:
        """
        Given the previously extracted list of features, rearrange them so that the output is 
        a Torch Tensor whose first dimension is the one of the patients.
        Arguments:
            features (list[list[torch.Tensor]]): the list of processed images.
            k (int): the number of additional slices per side.
        Returns:
            full_stakced (torch.Tensor): a tensor of dimension [#patients x #modalities x #images x #channels x #height_pixels x #weight_pixels]
        """
        print(f"La features list ha dimensioni {X.shape}")
        final_features_list = [ ]
        positions=[(0,1),(2,3),(4,5),(6,7)]
        for modality_num, couple in enumerate(positions):
            print(f"Pair: {couple}")
            pre_nac_list = X[couple[0]]
            post_nac_list = X[couple[1]]

            print(f"Lunghezza della pre_nac_list: {len(pre_nac_list)}")
            print(f"Lunghezza della post_nac_list: {len(post_nac_list)}")

            # Devo splittare la lista con tutte le sequenze per paziente
            pre_nac_shaded_list = []
            for i in range(0,len(pre_nac_list), 2*k+1):
                print(f"Processing {len(pre_nac_shaded_list)+1}° patient of the list.")
                subsequence_for_patient=[]
                for j in range(i,i+2*k+1):
                    subsequence_for_patient.append(pre_nac_list[j])
                print(f"Numero di slice per modalità {modality_num + 1}, PRE nac: {len(subsequence_for_patient)}")
                pre_nac_shaded_list.append(subsequence_for_patient)
            print(f"PRE NAC shaded list ha lunghezza: {len(pre_nac_shaded_list)}\n")

            post_nac_shaded_list = []
            for i in range(0,len(post_nac_list), 2*k+1):
                print(f"Processing {len(post_nac_shaded_list)+1}° patient of the list.")
                subsequence_for_patient=[]
                for j in range(i,i+2*k+1):
                    subsequence_for_patient.append(post_nac_list[j])
                print(f"Numero di slice per modalità {modality_num + 1}, POST nac: {len(subsequence_for_patient)}")
                post_nac_shaded_list.append(subsequence_for_patient)
            print(f"POST NAC shaded list ha lunghezza: {len(post_nac_shaded_list)}")

            if len(pre_nac_shaded_list)!=len(post_nac_shaded_list):
                print("Le due liste non hanno la stessa lunghezza")
                exit()

            print(f"Il tipo è: {type(pre_nac_shaded_list)}\n")

            final_features_list.append(pre_nac_shaded_list)
            final_features_list.append(post_nac_shaded_list)

        print("Analisi delle dimensioni sulla final_features_list:")
        print(f"Mi aspetto siano 8: {len(final_features_list)}")

        print(f"La lista con tutte le features ha {len(final_features_list)} elementi. ")
        for i in range(len(final_features_list)):
            print(f"La modalità numero {i+1} ha {len(final_features_list[i])} liste, ovvero pazienti:")
            for j in range(len(final_features_list[i])):
                print(f"La sequenza {i+1} del paziente {j+1} ha {len(final_features_list[i][j])} slices.")
            print("")

        patient_stacked_list = []    
        for patient_idx in range(len(final_features_list)):
            list_modalities = []
            for modality_idx in range(len(final_features_list[patient_idx])):
                sequence_tensor_list = final_features_list[patient_idx][modality_idx]
                list_modalities.append(torch.stack(sequence_tensor_list, dim=0))
            patient_stacked_list.append(torch.stack(list_modalities, dim=0))
        full_stacked = torch.stack(patient_stacked_list, dim=0)
        full_stacked = full_stacked.transpose(0,1)
        print(f"Il nuovo tensore X ha dimensioni: {full_stacked.shape}\n")

        # Sharding delle Y 
        Y_shaded_list = torch.Tensor()
        for i in range(0,Y.shape[0], 2*k+1):
            print(f"Processing {Y_shaded_list.shape[0] +1}° patient of the list.")
            subsequence_for_patient = torch.Tensor()
            for j in range(i,i+2*k+1):
                subsequence_for_patient = torch.cat((subsequence_for_patient,Y[j].unsqueeze(0)), dim = 0)
            print(f"Numero di labels per paziente: {subsequence_for_patient.shape[0]}")
            Y_shaded_list = torch.cat((Y_shaded_list, subsequence_for_patient.unsqueeze(0)), dim = 0)
        print(f"Il nuovo tensore Y ha dimensioni: {Y_shaded_list.shape}\n")

        return full_stacked, Y_shaded_list
    
    def _rearrange_patients_list(self, patient_list, k):
        shortened_patients_list = []
        for i in range(0,len(patient_list), 2*k+1):
            shortened_patients_list.append(patient_list[i])
        return shortened_patients_list