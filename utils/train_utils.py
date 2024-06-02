import os
import ast
import SimpleITK as sitk
import numpy as np
import torch 
import pandas as pd
from datetime import datetime

Kfold_val = [[10, 13, 18, 22, 28, 31, 32, 37, 4], #5, 29 missing - 31 double
              [1, 12, 14, 16, 19, 26, 33, 35, 9],
              [15, 17, 2, 20, 24, 27, 3, 30, 7],
              [11, 21, 23, 25, 31, 34, 36, 6, 8]]

# Kfold_val = [[10, 2, 11],   #solo per debugging
#              [3,11],
#              [2],
#              [1]]

def retrieve_folders_list(root_dir:str) -> "list[str]":
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

def Kfold_split(folders, k):
    """
    Splits the folders at patient level, in order to have train and validation dataset.
    Arguments:
        folders (list[str]): list of folder paths.
        k (int): number of kfolds.
    Returns:
        Kfold_list (list[list[str]]): a list containing two lists, train and validation lists. 
    """
    old_patient = ""
    patient_sequences= []
    patient_list = []
    Kfold_list = []

    for sequence in folders:
        patient_flag = False
        dicom_files = []

        for r, d, f in os.walk(sequence):
            f.sort()
            for file in f:
                if '.dcm' in file or '.IMA' in file:
                    dicom_files.append(os.path.abspath(os.path.join(r,file)))
                    break

        reader = sitk.ImageFileReader()
        reader.SetFileName(dicom_files[0])
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        description = reader.GetMetaData("0008|103e").strip()

        name_num = description.split('_')[0]

        if name_num == old_patient:
            patient_sequences.append(sequence)
        else:
            old_patient = name_num
            patient_flag = True
            patient_sequences= []
            patient_list.append(patient_sequences)
            patient_sequences.append(sequence)

    for fold in range(k):
        train_list = []
        val_list = []
        fold_list = []

        nac_train_name = []
        nac_val_name = []

        for patient in patient_list:
            dicom_files = []

            for r, d, f in os.walk(patient[0]):
                f.sort()
                for file in f:
                    if '.dcm' in file or '.IMA' in file:
                        dicom_files.append(os.path.abspath(os.path.join(r,file)))
                        break

            reader.SetFileName(dicom_files[0])
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()

            description = reader.GetMetaData("0008|103e").strip()

            name_num = int(description.split('_')[0])

            if name_num in Kfold_val[fold]: #qua decide quali pazienti finiscono nel train e quali nel val
                val_list.extend(patient)
                nac_val_name.append(str(name_num))
            else:
                train_list.extend(patient)
                nac_train_name.append(str(name_num))
                
        print(f"Fold number {fold+1}:")
        print(f"Training patients: {set(nac_train_name)}")
        print(f"Validation patients: {set(nac_val_name)}\n")

        fold_list.append(train_list)
        fold_list.append(val_list)
        Kfold_list.append(fold_list)

    return Kfold_list

def print_results(roc_slice, roc_patient):
    print("Print dei results")
    
    log_list = ['auc', 'accuracy', 'sensitivity', 'specificity', 'f1_score']
    for key in roc_slice[0].keys():   #roc_slice è una lista di dizionari, con metriche calcolate a livello SLICE
        
        

        #------------------------------------------------------------------------------------- metriche livello paziente
        std_array_patient = []
        if key == 'pCR':
            for d in roc_patient:
                std_array_patient.append(d[key][0])
        mean_value = sum(d[key] for d in roc_patient) / len(roc_patient)
        std_value = np.std(std_array_patient)
        for i, log in enumerate(log_list):
            if log == 'auc' and key == 'pCR':
                #Faceva plot speciale su neptune
                pass
            str_value = '{0}_patient-level {1} mean = {2}'.format(log, key, mean_value[i])
            print(str_value)
            if log == 'auc' and key == 'pCR':
                str_std_value = '{0}_patient-level {1} std = {2}'.format(log, key, std_value)
                print(str_std_value)
        #-------------------------------------------------------------------------------------- metriche livello slice
        std_array_slice = []
        if key == 'pCR':
            for d in roc_slice:
                std_array_slice.append(d[key][0])
        mean_value = sum(d[key] for d in roc_slice) / len(roc_slice)
        std_value = np.std(std_array_slice)
        for i, log in enumerate(log_list):
            if log == 'auc' and key == 'pCR':
                #BOHFaceva plot speciale su neptune
                pass
            str_value = '{0}_slice-level {1} mean = {2}'.format(log, key, mean_value[i])
            print(str_value)
            if log == 'auc' and key == 'pCR':
                str_std_value = '{0}_slice-level {1} std = {2}'.format(log, key, std_value)
                print(str_std_value)
    print("Fine dell' esperimento.")

def export_result_as_df(slice_dictionaries, patient_dictionaries, args):
    
    lists_of_dictionaries = [slice_dictionaries, patient_dictionaries]
    levels = ["SLICE", "PATIENT"]
    branches_list = list(slice_dictionaries[0].keys()) #["pCR", "DWI_probs", "T2_probs", "DCEpeak_probs", "DCE3TP_probs"]

    output_list = []

    for i,level in enumerate(levels):
        dataframe_dict = {}
        for k in branches_list:
            dataframe_dict[k]={}

        for branch in branches_list:
            auc_values = np.array([])
            acc_values = np.array([])
            spe_values = np.array([])
            se_values = np.array([])
            f1_values = np.array([])
            for fold_dict in lists_of_dictionaries[i]:
                auc_values = np.append(auc_values, fold_dict[branch][0])
                acc_values= np.append(acc_values, fold_dict[branch][1])
                se_values= np.append(se_values, fold_dict[branch][2])
                spe_values= np.append(spe_values, fold_dict[branch][3])
                f1_values= np.append(f1_values, fold_dict[branch][4])

            auc_mean = np.mean(auc_values)
            auc_std = np.std(auc_values)
            acc_mean = np.mean(acc_values)
            acc_std = np.std(acc_values)
            spe_mean = np.mean(spe_values)
            spe_std = np.std(spe_values)
            se_mean = np.mean(se_values)
            se_std = np.std(se_values)
            f1_mean = np.mean(f1_values)
            f1_std = np.std(f1_values)
            
            dataframe_dict[branch]["auc_values"] = auc_values
            dataframe_dict[branch]["auc_mean"] = auc_mean
            dataframe_dict[branch]["auc_std"] = auc_std

            dataframe_dict[branch]["acc_values"] = acc_values
            dataframe_dict[branch]["acc_mean"] = acc_mean
            dataframe_dict[branch]["acc_std"] = acc_std

            dataframe_dict[branch]["se_values"] = se_values
            dataframe_dict[branch]["se_mean"] = se_mean
            dataframe_dict[branch]["se_std"] = se_std

            dataframe_dict[branch]["spe_values"] = spe_values
            dataframe_dict[branch]["spe_mean"] = spe_mean
            dataframe_dict[branch]["spe_std"] = spe_std

            dataframe_dict[branch]["f1_values"] = f1_values
            dataframe_dict[branch]["f1_mean"] = f1_mean
            dataframe_dict[branch]["f1_std"] = f1_std

        df = pd.DataFrame(dataframe_dict)
        df_t = df.transpose()
        output_list.append(df_t)
    
    df1 = output_list[0]
    df2 = output_list[1]
    
    df3 = pd.DataFrame({
        "Preprocess": [args.preprocess],
        "Epochs": [args.epochs],
        "LR": [args.learning_rate]
    }).transpose()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with pd.ExcelWriter(f"analisi_dei_risultati_aggregati_{timestamp}.xlsx") as writer:
        df1.to_excel(writer, sheet_name=f'{levels[0]}_level')
        df2.to_excel(writer, sheet_name=f'{levels[1]}_level')
        df3.to_excel(writer, sheet_name=f'Iperparametri')

