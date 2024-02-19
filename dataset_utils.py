import os
import ast
import SimpleITK as sitk
import numpy as np
import torch 

# Kfold_val = [[10, 13, 18, 22, 28, 31, 32, 37, 4], #5, 29 missing - 31 double
#              [1, 12, 14, 16, 19, 26, 33, 35, 9],
#              [15, 17, 2, 20, 24, 27, 3, 30, 7],
#              [11, 21, 23, 25, 31, 34, 36, 6, 8]]

Kfold_val = [[2],  
             [3],
             [2],
             [1]]

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
            else:
                train_list.extend(patient)

        fold_list.append(train_list)
        fold_list.append(val_list)
        Kfold_list.append(fold_list)

    return Kfold_list