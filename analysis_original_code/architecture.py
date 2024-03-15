import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

import SimpleITK as sitk
import argparse
import os
import tempfile
from datetime import datetime
import ast
import numpy as np
import pandas as pd
import random
import copy

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.imagenet_utils import decode_predictions

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve, pairwise_distances, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.impute import KNNImputer
from sklearn.utils import class_weight
from sklearn.linear_model import LinearRegression
import skimage.io
import skimage.segmentation

# Test if GPU detected
assert tf.config.list_physical_devices('GPU')
assert tf.test.is_built_with_cuda()

# DEFAULT
"""
    Default initializations:
        The accepted scans are: DWI, T2, T1 and DCE
        Arguments to be passed through command line are are defined:
            epochs, batch_size, learning_rate, different input_path
            --branches allows to choose which ResNet50 to add, to process specific sub-sequences, which must be compatible with the accepted scans.
            --slices allows to select extra slices in addition to the index one, with a number that will be equal "on the left" and "on the right" of the index slice, used as midpoint
"""
scans = ["DWI", "T2", "DCE"]
accepted_branches = ["DWI", "T2", "DCE_peak", "DCE_3TP"]
NB_CLASSES = 2
Kfold_val = [[10, 13, 18, 22, 28, 31, 32, 37, 4],
             [1, 12, 14, 16, 19, 26, 33, 35, 9],
             [15, 17, 2, 20, 24, 27, 3, 30, 7],
             [11, 21, 23, 25, 31, 34, 36, 6, 8]]


parser = argparse.ArgumentParser(description='Deep Learning on NAC Data')
parser.add_argument('--branches', type=str, action='append', required=False,
                    help='Choose which branches of the architecture to use: DWI, T2, DCE_peak, DCE_3TP')
parser.add_argument('--slices', type=int, default=3, required=False,
                    help='How many more slice to add to the dataset besides the index one')
parser.add_argument('--load_weights', type=int, default=0, required=False,
                    help='Whether to load weights from trained model (to be placed in the folder "weights/") (0=no, 1=yes)')
parser.add_argument('--features', type=int, default=0, required=False,
                    help='Whether to use hand-crafted features as input (0=no, 1=yes)')
parser.add_argument('--multi', type=int, default=0, required=False,
                    help='Whether to use add extra task for feature prediction (0=no, 1=yes)')
parser.add_argument('--feature_multi', type=str, default='MRI_size_rl', required=False,
                    help='Choose which features select for the extra task: MRI_MTT, MRI_ADC_min, MRI_ADC_max, MRI_size_ap, MRI_size_rl, MRI_size_cc')
parser.add_argument('--secondary_weight', type=float, default=0.2, required=False,
                    help='Weight used to scale loss terms from secondary tasks')
parser.add_argument('--class_weight', type=int, default=1, required=False,
                    help='Whether to use class weight during training (0=no, 1=yes)')
parser.add_argument('--folds', type=int, default=4, required=False, help='Number of k-folds for CV')
parser.add_argument('--early_stop', type=int, default=0, required=False,
                    help='Whether to use an early stopping policy (0=no, 1=yes)')
parser.add_argument('--epochs', type=int, default=15, required=False, help='Number of epochs')
parser.add_argument('--batch', type=int, default=12, required=False, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001, required=False, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, required=False, help='Momentum value')
parser.add_argument('--dropout', type=float, default=0.5, required=False, help='Dropout rate')
parser.add_argument('--l2_reg', type=float, default=0.0001, required=False, help='L2 regularization alpha value')
parser.add_argument('--fc', type=int, default=128, required=False, help='Size of last FC layer')
parser.add_argument('--seed', type=int, default=42, required=False, help='Seed for reproducible results')
parser.add_argument('--loss_function', type=str, default='binary_crossentropy', required=False, help='Loss function')
parser.add_argument('--lime_top', type=int, default=4, required=False, help='Number of relevant lime superpixels')
parser.add_argument('--lime_pert', type=int, default=100, required=False, help='Number of lime perturbations')
parser.add_argument('--input_path', type=str, default='NAC_Input', required=False, help='Dataset input path')
parser.add_argument('--neptune_token', type=str, default="", help="NEPTUNE API TOKEN")
parser.add_argument('--name', type=str, default="", help="Folder name")
args = parser.parse_args()
print(vars(args), "\n")

# Configure neptune.ai
project_name = "GRAINS/NacResponse"
neptune_token = args.neptune_token
use_neptune = len(neptune_token) != 0

folder_name = args.name
branches_list = args.branches
SLICES = args.slices
load_weights = args.load_weights
FEATURES = args.features
MULTI = args.multi
feature_multi = args.feature_multi
secondary_tasks_weight = args.secondary_weight
WEIGHT = args.class_weight
EPOCHS = args.epochs
FOLDS = args.folds
early_stop = args.early_stop
NB_EPOCH = args.epochs
BATCH_SIZE = args.batch
shuffle = True
dropout = args.dropout
l2_reg = args.l2_reg
FC = args.fc
SEED = args.seed
momentum = args.momentum
LR = args.learning_rate
LIME = args.lime_top
PERT = args.lime_pert
path = args.input_path
loss_function = args.loss_function

# SEED FIX
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if not branches_list:
    branches_list = ["DCE_peak"]
else:
    for branch in branches_list:
        if branch not in accepted_branches:
            print("Invalid branch {0} selected!\nValid branches: DWI, T2, DCE_peak, DCE_3TP".format(branch))
            exit()

if loss_function not in ["binary_crossentropy", "categorical_crossentropy", "focal_loss"]:
    print("Invalid loss {0} selected!\nValid loss function: binary_crossentropy, focal_loss".format(loss_function))
    exit()

if FOLDS not in range(1, 5):
    print("Invalid number folds {0} selected!\nMaximum number for Kfold is 4".format(FOLDS))
    exit()

PARAMS = {'batch_size': BATCH_SIZE,
          'folds': FOLDS,
          'n_epochs': NB_EPOCH,
          'shuffle': shuffle,
          'dropout': dropout,
          'l2_regularization': l2_reg,
          'FC_size': FC,
          'seed': SEED,
          'load_weights': load_weights,
          'loss': loss_function,
          'learning_rate': LR,
          'optimizer': 'SGD',
          'momentum': momentum,
          'slices': SLICES,
          'features_input': FEATURES,
          'multiple_task': MULTI,
          'multiple_features': feature_multi,
          'secondary_weight': secondary_tasks_weight,
          'class_weight': WEIGHT,
          'early_stop': early_stop,
          'lime_top': LIME,
          'lime_pert': PERT
          }

# UTILITIES
"""
    scan_folders simply selects every scan folder properly stored in the input folder.
    The structure of the input folder is:
        -NAC_X folder for patient number X
        -Each NAC folder contains two folders: MRI_1 and MRI_2
        -Each MRI folder contains the five slected sub_sequences: DWI, T2, DCE (pre, peak and post)
    The test patients are scanned from the "test" folder inside the input one.
"""


def scan_folders():
    train_folders = []

    # List of all sequences (excluding NAC e MRI_1/2 folders)
    for r, d, f in os.walk(path):
        d.sort()
        for folder in d:
            if folder[:3] != 'NAC' and folder != 'MRI_1' and folder != 'MRI_2':
                train_folders.append(os.path.join(r, folder))
    return train_folders


"""
    Perform KFold Cross Validation, returning an array with all the K splits
"""


def Kfold_split(folders, k):
    old_patient = ""
    patient_sequences = []
    patient_list = []
    Kfold_list = []

    for sequence in folders:
        patient_flag = False
        dicom_files = []

        for r, d, f in os.walk(sequence):
            f.sort()
            for file in f:
                if '.dcm' in file or '.IMA' in file:
                    dicom_files.append(os.path.abspath(os.path.join(r, file)))
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
            patient_sequences = []
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
                        dicom_files.append(os.path.abspath(os.path.join(r, file)))
                        break

            reader.SetFileName(dicom_files[0])
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()

            description = reader.GetMetaData("0008|103e").strip()

            name_num = int(description.split('_')[0])

            if name_num in Kfold_val[fold]:
                val_list.extend(patient)
            else:
                train_list.extend(patient)

        fold_list.append(train_list)
        fold_list.append(val_list)
        Kfold_list.append(fold_list)

    return Kfold_list


"""
    define_subsequences first read the ground_truth (pCR result: 0 or 1) from a .txt file then operates for each sequences:
        - .dcm and .IMA images are read
        - from the dicom metadata, information like Patient, MRI (1 or 2), sequence type are retrieved
        - the metadata SeriesDescription is used to locate the index slice
        - the metadata InstanceNumber is used to properly select extra slices according to the "--slices" param and the index slice
    labels are added once for each patient (each sequence from the same patient has the same result) and the images (read with simpleITK) are appended to the corresponding sub_sequence array
"""


def define_subsequences(folders, fold):
    sequence_count = 0
    index_count = 0
    image_count = 0

    new_patient = ""

    features = []
    labels = []
    scan_list = []
    DWI_1, DWI_2, T2_1, T2_2, T1_1, T1_2 = ([] for i in range(6))
    sub_sequences = [DWI_1, DWI_2, T2_1, T2_2, T1_1, T1_2]

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
        selected_slices = SLICES
        patient_flag = False
        dicom_files = []
        images = []
        slices = []

        for r, d, f in os.walk(sequence):
            f.sort()
            for file in f:
                if '.dcm' in file or '.IMA' in file:
                    dicom_files.append(os.path.abspath(os.path.join(r, file)))

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

        if scan_string in scans:
            for dicom_file in dicom_files:
                images.append(sitk.ReadImage(dicom_file))

            slices_num = len(images)
            max_slices = int(((slices_num - 1) / 2))
            if selected_slices > max_slices:
                selected_slices = max_slices

            max_instance = 0
            min_instance = 650

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
                    upper_bound = min(max_instance, (index_instance + selected_slices))
                    lower_bound = max(min_instance, (index_instance - selected_slices))

                    if (index_instance + selected_slices) > max_instance:
                        diff_bound = selected_slices - (max_instance - index_instance)
                        lower_bound = max(min_instance, (index_instance - selected_slices - diff_bound))

                    if (index_instance - selected_slices) < 0:
                        diff_bound = selected_slices - (index_instance)
                        upper_bound = min(max_instance, (index_instance + selected_slices + diff_bound))

                    if scan_string != "DCE":
                        slices.append(sitk.GetArrayFromImage(img))
                    else:
                        slices.append(img)

            for img in images:
                img_instance = int(img.GetMetaData('0020|0013'))

                if index_instance < img_instance <= upper_bound:
                    image_count += 1
                    if scan_string != "DCE":
                        slices.append(sitk.GetArrayFromImage(img))
                    else:
                        slices.append(img)
                if index_instance > img_instance >= lower_bound:
                    image_count += 1
                    if scan_string != "DCE":
                        slices.append(sitk.GetArrayFromImage(img))
                    else:
                        slices.append(img)

        name_string = "NAC_" + name_num
        if name_string in dict:
            label = list(str(dict[name_string]))
        if patient_flag:
            for i in range(len(slices)):
                labels.append(label)
                scan_list.append(name_string)

        if "DWI" in branches_list:
            if scan_string == "DWI":
                if mri_string == "MRI1":
                    DWI_1.extend(slices)
                else:
                    DWI_2.extend(slices)

        if "T2" in branches_list:
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

    for sub_sequence in sub_sequences:
        features.append(sub_sequence)

    sub_sequence_div = (10 / len(branches_list)) / 2
    print("Fold:", fold + 1)
    print("Total DICOM Sequences:", round(sequence_count / sub_sequence_div))
    print("Total DICOM Index Slices:", round(index_count / sub_sequence_div))
    print("Total DICOM Selected Slices:", round(image_count / sub_sequence_div), "\n")
    return features, np.array(labels), scan_list, selected_slices


"""
    define_input operates the last steps in loading input images, obtaining the proper DCE sequences and returning the correct numpy arrays in terms of type and channel
"""


def define_input(features, labels):
    if "DCE_peak" or "DCE_3TP" in branches_list:
        DCE_features = define_DCE(features[-2], features[-1])
        features = features[:len(features) - 2]
        for DCE_feature in DCE_features:
            features.append(DCE_feature)

    X = []
    for feature in features:
        feature = np.array(feature)
        if feature.any():
            X_sub = np.transpose(feature, (0, 3, 2, 1))
            if feature.shape[1] == 1:
                X_sub = np.repeat(X_sub, 3, -1)
            X_sub = X_sub.astype('float64')
            X.append(X_sub)
    Y = to_categorical(labels, NB_CLASSES)
    return X, Y


"""define_DCE processes the loaded T1 slices, returning the two DCE subsequences: the one using only the peak scan 
and the one with the three time points, each occupying one RGB channel of the final image """


def define_DCE(T1_1, T1_2):
    new_patient = ""
    DCE_dict = {}

    features = []
    labels = []
    DCE_peak1, DCE_peak2, DCE_3TP1, DCE_3TP2 = ([] for i in range(4))
    sub_sequences = [DCE_peak1, DCE_peak2, DCE_3TP1, DCE_3TP2]
    slices = (SLICES * 2 + 1)

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
        patient_time = np.array(patient_time).astype(np.float)
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
            time = float(img.GetMetaData("0008|0031").strip())

            if "DCE_peak" in branches_list:
                if patient_name == patient and time == DCE_peak:
                    DCE_peak1.append(sitk.GetArrayFromImage(img))

            if "DCE_3TP" in branches_list:
                if patient_name == patient:
                    if time == DCE_pre:
                        pre_array.append(sitk.GetArrayFromImage(img))
                        DCE_count += 1
                    elif time == DCE_peak:
                        peak_array.append(sitk.GetArrayFromImage(img))
                        DCE_count += 1
                    else:
                        post_array.append(sitk.GetArrayFromImage(img))
                        DCE_count += 1
                    if DCE_count == slices * 3:
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
        patient_time = np.array(patient_time).astype(np.float)
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
            time = float(img.GetMetaData("0008|0031").strip())

            if "DCE_peak" in branches_list:
                if patient_name == patient and time == DCE_peak:
                    DCE_peak2.append(sitk.GetArrayFromImage(img))

            if "DCE_3TP" in branches_list:
                if patient_name == patient:
                    if time == DCE_pre:
                        pre_array.append(sitk.GetArrayFromImage(img))
                        DCE_count += 1
                    elif time == DCE_peak:
                        peak_array.append(sitk.GetArrayFromImage(img))
                        DCE_count += 1
                    else:
                        post_array.append(sitk.GetArrayFromImage(img))
                        DCE_count += 1
                    if DCE_count == slices * 3:
                        for i in range(slices):
                            image_3TP = np.vstack((pre_array[i], peak_array[i], post_array[i]))
                            DCE_3TP2.append(image_3TP)

    for sub_sequence in sub_sequences:
        features.append(np.array(sub_sequence))
    return features


"""
    define_model returns the keras functional model with branches added based on the specified parameter
"""


def define_model():
    input_list = []
    output_list = []
    res_list = []

    res_net = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3), weights="imagenet",
                                             pooling="avg")

    """for layer in res_net.layers:
        if 'conv2' in layer.name or 'conv3' in layer.name or 'conv4' in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True"""

    # Regularization for ResNet50
    regularizer = tf.keras.regularizers.l2(l2_reg)
    if l2_reg != 0:
        if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
            print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
            exit()

        for layer in res_net.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        model_json = res_net.to_json()
        tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
        res_net.save_weights(tmp_weights_path)
        res_net = tf.keras.models.model_from_json(model_json)
        res_net.load_weights(tmp_weights_path, by_name=True)
    # print(res_net.losses)

    if "DWI" in branches_list:
        DWI1_input = tf.keras.Input(shape=(224, 224, 3), name="DWI_1")
        DWI2_input = tf.keras.Input(shape=(224, 224, 3), name="DWI_2")
        input_list.extend((DWI1_input, DWI2_input))

        DWI_res = tf.keras.Model(inputs=res_net.input, outputs=res_net.output, name='resnet50_DWI')
        DWI1_res = DWI_res(DWI1_input)
        DWI2_res = DWI_res(DWI2_input)
        res_list.extend((DWI1_res, DWI2_res))

        conc_features = tf.keras.layers.concatenate([DWI1_res, DWI2_res], name='conc_DWI')
        DWI_pred = tf.keras.layers.Dense(2, activation='softmax', name="DWI")(conc_features)
        output_list.append(DWI_pred)
        print("DWI branch added")

    if "T2" in branches_list:
        T21_input = tf.keras.Input(shape=(224, 224, 3), name="T2_1")
        T22_input = tf.keras.Input(shape=(224, 224, 3), name="T2_2")
        input_list.extend((T21_input, T22_input))

        T2_res = tf.keras.Model(inputs=res_net.input, outputs=res_net.output, name='resnet50_T2')
        T21_res = T2_res(T21_input)
        T22_res = T2_res(T22_input)
        res_list.extend((T21_res, T22_res))

        conc_features = tf.keras.layers.concatenate([T21_res, T22_res], name='conc_T2')
        T2_pred = tf.keras.layers.Dense(2, activation='softmax', name="T2")(conc_features)
        output_list.append(T2_pred)
        print("T2 branch added")

    if "DCE_peak" in branches_list:
        DCEpeak1_input = tf.keras.Input(shape=(224, 224, 3), name="DCE_peak_1")
        DCEpeak2_input = tf.keras.Input(shape=(224, 224, 3), name="DCE_peak_2")
        input_list.extend((DCEpeak1_input, DCEpeak2_input))

        DCEpeak_res = tf.keras.Model(inputs=res_net.input, outputs=res_net.output, name='resnet50_DCE_peak')
        DCEpeak1_res = DCEpeak_res(DCEpeak1_input)
        DCEpeak2_res = DCEpeak_res(DCEpeak2_input)
        res_list.extend((DCEpeak1_res, DCEpeak2_res))

        conc_features = tf.keras.layers.concatenate([DCEpeak1_res, DCEpeak2_res], name='conc_DCE_peak')
        DCE_peak_pred = tf.keras.layers.Dense(2, activation='softmax', name="DCE_peak")(conc_features)
        output_list.append(DCE_peak_pred)
        print("DCE_peak branch added")

    if "DCE_3TP" in branches_list:
        DCE3TP1_input = tf.keras.Input(shape=(224, 224, 3), name="DCE_3TP_1")
        DCE3TP2_input = tf.keras.Input(shape=(224, 224, 3), name="DCE_3TP_2")
        input_list.extend((DCE3TP1_input, DCE3TP2_input))

        DCE3TP_res = tf.keras.Model(inputs=res_net.input, outputs=res_net.output, name='resnet50_DCE_3TP')
        DCE3TP1_res = DCE3TP_res(DCE3TP1_input)
        DCE3TP2_res = DCE3TP_res(DCE3TP2_input)
        res_list.extend((DCE3TP1_res, DCE3TP2_res))

        conc_features = tf.keras.layers.concatenate([DCE3TP1_res, DCE3TP2_res], name='conc_DCE_3TP')
        DCE_3TP_pred = tf.keras.layers.Dense(2, activation='softmax', name="DCE_3TP")(conc_features)
        output_list.append(DCE_3TP_pred)
        print("DCE_3TP branch added")

    conc_features = tf.keras.layers.concatenate(res_list, name='conc_all')
    drop = tf.keras.layers.Dropout(dropout)(conc_features)
    dense = tf.keras.layers.Dense(FC, activation='relu')(drop)

    pCR_pred = tf.keras.layers.Dense(2, activation='softmax', name="pCR")(dense)
    output_list.append(pCR_pred)

    print(input_list)
    print(output_list)

    model = tf.keras.Model(
        inputs=[input_list],
        outputs=[output_list]
    )
    return model


"""
    define focal_loss
"""


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


"""
    test_predict computes the metrics based on the predictions made on the test set. The confusion matrix, with its derivations, as well as roc_auc, with its chart, are considered. Test_predict is called both for slice and patient level, computing these metrics of all the outputs of the model (single subsequnces and ensamble output)
"""
def test_predict(Y_test, Y_prob, level, output_name, folder_time, fold):
    matrix = confusion_matrix(Y_test.argmax(axis=-1), Y_prob.argmax(axis=-1))
    roc = roc_auc_score(Y_test.argmax(axis=-1), Y_prob[:, 1])

    TN = matrix[0][0]
    FN = matrix[1][0]
    TP = matrix[1][1]
    FP = matrix[0][1]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = (TP / (TP + FN)).round(4) * 100.0
    # Specificity or true negative rate
    TNR = (TN / (TN + FP)).round(4) * 100.0
    # Precision or positive predictive value
    PPV = (TP / (TP + FP)).round(4) * 100.0
    # Negative predictive value
    NPV = (TN / (TN + FN)).round(4) * 100.0
    # Fall out or false positive rate
    FPR = (FP / (FP + TN)).round(4) * 100.0
    # False negative rate
    FNR = (FN / (FN + TP)).round(4) * 100.0
    # False discovery rate
    FDR = (FP / (FP + TP)).round(4) * 100.0
    # Overall accuracy
    ACC = ((TP + TN) / (TP + FP + FN + TN)).round(4) * 100.0

    # ROC plot
    FPR_AUC, TPR_AUC, thresholds = roc_curve(Y_test.argmax(axis=-1), Y_prob[:, 1])
    trace_roc = go.Scatter(x=FPR_AUC, y=TPR_AUC, mode='lines', name="Model ROC (AUC={0:.2f})".format(roc))
    trace_random = go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='random guess', line=dict(dash='dash'))
    data = [trace_roc, trace_random]

    layout = dict(title='ROC: {0} - Level: {1} - Fold: {2}'.format(output_name, level, fold + 1),
                  xaxis=dict(title='False Positive Rate', constrain='domain'),
                  yaxis=dict(title='True Positive Rate', scaleanchor="x", scaleratio=1))
    fig = go.Figure(data=data, layout=layout)
    # fig.show()

    save_path = "images/{0}/{1}-level_{2}_ROC_slices{3}_fold{4}.png".format(folder_time, level, output_name,
                                                                            (SLICES * 2 + 1), fold + 1)
    # fig.write_image(save_path)

    if level == "slice":
        print("\nSLICE LEVEL")
    elif level == "patient":
        print("\nPATIENT LEVEL")
    print(output_name, "\n")

    pred_labels = "Fold: {0}\nModel: {1}\nReal labels: {2}\nPred labels: {3}\n".format(fold + 1, output_name,
                                                                                       Y_test.argmax(axis=-1),
                                                                                       Y_prob.argmax(axis=-1))
    confusion_matrix_string = "Fold: {0}\nLevel: {1}\nModel: {2}\nTN: {3}\nFN: {4}\nTP: {5}\nFP: {6}\n".format(fold + 1,
                                                                                                               level,
                                                                                                               output_name,
                                                                                                               TN, FN,
                                                                                                               TP, FP)
    confusion_matrix_deriv = "Fold: {0}\nLevel: {1}\nModel: {2}\n\nAccuracy: {3:.2f}%\nSensitivity (TPR): {3:.2f}%\nSpecificity (TNR): {5:.2f}%\nFall-out (FPR): {6:.2f}%\nFalse Negative Rate (FNR): {7:.2f}%\nPrecision (PPV): {8:.2f}%\nNegative Predictive Value (NPV): {9:.2f}%".format(
        fold + 1, level, output_name, ACC, TPR, TNR, FPR, FNR, PPV, NPV)

    print(pred_labels)
    print(confusion_matrix_string)
    print(confusion_matrix_deriv)

    print("Val AUC: {0:.2f}\n".format(roc))

    test_array = [roc, ACC, TPR, TNR]
    return np.array(test_array)


"""
    test_predict_multi predicts the multi-task result, dealing with the linear regression output both at slice and at patient level
"""


def test_predict_multi(Y_test, Y_prob, output_name, folder_time, fold, feature_multi):
    Y_test = Y_test.flatten()
    Y_prob = Y_prob.flatten()

    pred_labels_slice = "Fold: {0}\nModel: {1}\nFeatures: {2}\nReal labels: {3}\nPred labels: {4}".format(fold + 1,
                                                                                                          output_name,
                                                                                                          feature_multi,
                                                                                                          Y_test,
                                                                                                          Y_prob)
    result_slice = mean_squared_error(Y_test, Y_prob)
    print(pred_labels_slice)
    print("Val MSE: {0}\n".format(result_slice))

    patient_num = len(Y_test) / (SLICES * 2 + 1)
    Y_test_split = []
    Y_prob_split = []

    split = np.array_split(Y_test, patient_num)

    for slice in split:
        all_equal = all(feature == slice[0] for feature in slice)

        if all_equal:
            Y_test_split.append(slice[0])
        else:
            print("Error in slices division per patient!")
            return

    split = np.array_split(Y_prob, patient_num)
    split = np.mean(split, axis=1)

    for slice in split:
        Y_prob_split.append(slice)

    pred_labels_patient = "Fold: {0}\nModel: {1}\nFeatures: {2}\nReal labels: {3}\nPred labels: {4}".format(fold + 1,
                                                                                                            output_name,
                                                                                                            feature_multi,
                                                                                                            Y_test_split,
                                                                                                            Y_prob_split)
    result_patient = mean_squared_error(Y_test_split, Y_prob_split)
    print(pred_labels_patient)
    print("Val MSE: {0}\n".format(result_patient))

    return np.array(result_slice), np.array(result_patient)


"""
    get_patient_level calculates predictions previously made at the slice level at patient level, using a majority voting scheme
"""
def get_patient_level(Y_test, Y_prob):
    patient_num = len(Y_test) / (SLICES * 2 + 1)
    Y_test_split = []
    Y_prob_split = []

    split = np.array_split(Y_test, patient_num)

    for slice in split:
        all_equal = all(pCR == slice.argmax(axis=-1)[0] for pCR in slice.argmax(axis=-1))

        if all_equal:
            Y_test_split.append(slice[0])
        else:
            print("Error in slices division per patient!")
            return

    split = np.array_split(Y_prob, patient_num)
    split = np.mean(split, axis=1)

    for slice in split:
        Y_prob_split.append(slice)

    return np.array(Y_test_split), np.array(Y_prob_split)


"""
    normalize_test is used to apply z-normalization (0 mean and 1 std) to the validation and test set (this normalization step is done by the ImageDataGenerator for augmentation on the training set)
"""
#Questo lo posso evitare se normalizzo ciascuna slice quando vado a caricarla - istanziare il Dataset
def normalize_test(datagen, X_train):
    X_aug = []
    for gen_X in X_train:
        for i in range(len(gen_X)):
            gen_X[i] = datagen.standardize(gen_X[i])
        X_aug.append(gen_X)
    return X_aug


"""
    datagen_flow uses the defined ImageDataGenerator to properly augment each input sub_sequence
"""
def datagen_flow(datagen, X, Y):
    X_aug = []
    for gen_X in X:
        datagen.fit(gen_X) #sembra che lo applichi all'intera modalità (non alla singola slice) - ma qua i pazienti sono tutti concatenati, quindi in quella run tutte le sequenze subiscono stessa aug
        gen_X = datagen.flow(gen_X, Y, batch_size=BATCH_SIZE, shuffle=shuffle, seed=SEED)
        X_aug.append(gen_X)

    # One branch
    if len(X_aug) == 2:
        while True:
            X1i = X_aug[0].next()
            X2i = X_aug[1].next()
            Y_aug = [X1i[1], X1i[1]]
            yield [X1i[0], X2i[0]], Y_aug
    # Two branches
    elif len(X_aug) == 4:
        while True:
            X1i = X_aug[0].next()
            X2i = X_aug[1].next()
            X3i = X_aug[2].next()
            X4i = X_aug[3].next()
            Y_aug = [X1i[1], X1i[1], X1i[1]]
            yield [X1i[0], X2i[0], X3i[0], X4i[0]], Y_aug
    # Three branches
    elif len(X_aug) == 6:
        while True:
            X1i = X_aug[0].next()
            X2i = X_aug[1].next()
            X3i = X_aug[2].next()
            X4i = X_aug[3].next()
            X5i = X_aug[4].next()
            X6i = X_aug[5].next()
            Y_aug = [X1i[1], X1i[1], X1i[1], X1i[1]]
            yield [X1i[0], X2i[0], X3i[0], X4i[0], X5i[0], X6i[0]], Y_aug
    # Four branches
    else:
        while True:
            X1i = X_aug[0].next()
            X2i = X_aug[1].next()
            X3i = X_aug[2].next()
            X4i = X_aug[3].next()
            X5i = X_aug[4].next()
            X6i = X_aug[5].next()
            X7i = X_aug[6].next()
            X8i = X_aug[7].next()
            Y_aug = [X1i[1], X1i[1], X1i[1], X1i[1], X1i[1]]
            yield [X1i[0], X2i[0], X3i[0], X4i[0], X5i[0], X6i[0], X7i[0], X8i[0]], Y_aug

"""
    get_lime performs a lime prediction on the superpixels generated from the MRI1 slices of each branch. In this function the predictions are carried out using the whole multi input / multi output model, giving simultaneously in input to each branch its own perturbed input image
"""
def get_lime(X, model, fold, random_patient, datagen, X_features):
    num = SLICES * 2 + 1
    random_patient = random_patient + fold
    tot_patients = int(len(X[0]) / num)

    if random_patient >= (tot_patients):
        random_patient = tot_patients - fold

    slice = num * random_patient

    img_list = []
    img_array_list = []
    for i in range(len(branches_list) * 2):
        img = X[i][slice]
        img = tf.keras.preprocessing.image.array_to_img(img)
        img_list.append(img)

        img_norm = datagen.standardize(img)
        img_array = np.expand_dims(img_norm, axis=0)
        img_array_list.append(img_array)

    preds = model.predict(img_array_list)
    pred_class = preds[-1].argmax(axis=-1)
    pred_class = pred_class[0]

    superpixels_list = []
    num_superpixels_list = []
    for img in img_list:
        superpixels = skimage.segmentation.quickshift(img, kernel_size=4, max_dist=200, ratio=0.2)
        num_superpixels = np.unique(superpixels).shape[0]
        superpixels_list.append(superpixels)
        num_superpixels_list.append(num_superpixels)

    num_perturb = PERT
    perturbations_list = []
    for num_sup in num_superpixels_list:
        perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_sup))
        perturbations_list.append(perturbations)

    def perturb_image(img, perturbation, segments):
        active_pixels = np.where(perturbation == 1)[0]
        mask = np.zeros(segments.shape)
        for active in active_pixels:
            mask[segments == active] = 1
        perturbed_image = copy.deepcopy(img)
        perturbed_image = perturbed_image * mask[:, :, np.newaxis]
        return perturbed_image

    predictions_list = []
    for i in range(len(perturbations_list)):
        predictions = []
        for pert in perturbations_list[i]:
            pert_img_list = []
            for k, img in enumerate(img_list):
                perturbed_img = perturb_image(img, pert, superpixels_list[k])
                pert_img_list.append(perturbed_img[np.newaxis, :, :, :])
            pred = model.predict(pert_img_list)
            predictions.append(pred[-1])
        predictions_list.append(np.array(predictions))

    kernel_width = 0.25
    num_top_features = LIME
    masks_list = []
    coeff_list = []
    for k, img in enumerate(img_list):
        original_image = np.ones(num_superpixels_list[k])[np.newaxis, :]  # Perturbation with all superpixels enabled
        distances = pairwise_distances(perturbations_list[k], original_image, metric='cosine').ravel()
        weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))  # Kernel function

        simpler_model = LinearRegression()
        predictions = predictions_list[k]
        simpler_model.fit(X=perturbations_list[k], y=predictions[:, :, pred_class], sample_weight=weights)
        coeff = simpler_model.coef_[0]
        coeff_list.append(coeff)
        top_features = np.argsort(coeff)[-num_top_features:]

        mask = np.zeros(num_superpixels_list[k])
        mask[top_features] = True  # Activate top superpixels
        masks_list.append(mask)

    for k, branch in zip(range(0, len(branches_list) * 2, 2), branches_list):
        img_orig = img_list[k].rotate(270)
        img_orig = img_orig.transpose(Image.FLIP_LEFT_RIGHT)

        img_super = skimage.segmentation.mark_boundaries(img_list[k], superpixels_list[k])
        img_super = skimage.transform.rotate(img_super, 90)
        img_super = np.flip(img_super, 0)
        img_super = tf.keras.preprocessing.image.array_to_img(img_super)

        img_pert = perturb_image(img_list[k], perturbations_list[k], superpixels_list[k]).astype(np.uint8)
        img_pert = skimage.transform.rotate(img_pert, 90)
        img_pert = np.flip(img_pert, 0)

        pert_img_final = perturb_image(img_list[k], masks_list[k], superpixels_list[k]).astype(np.uint8)
        pert_img_final = skimage.transform.rotate(pert_img_final, 90)
        pert_img_final = np.flip(pert_img_final, 0)
        pert_img_final = tf.keras.preprocessing.image.array_to_img(pert_img_final)

    return mask


"""
    get_lime_2 provides another approach to lime prediction. In this case, for each MRI1 slice a new model is constructed consisting of an input layer, the corresponding ResNet50 extracted from the main trained model, and an output layer to perform the pCR prediction. Superpixels and perturbations are generated separately from each image; this is run separately for each MRI1 slice of each branch
"""
def get_lime_2(X, model, fold, random_patient, branch, k, datagen, X_features):
    num = SLICES * 2 + 1
    random_patient = random_patient + fold
    tot_patients = int(len(X[0]) / num)

    if random_patient >= (tot_patients):
        random_patient = tot_patients - fold

    slice = num * random_patient

    img_list = []
    img_array_list = []
    for i in range(len(branches_list) * 2):
        img = X[i][slice]
        img = tf.keras.preprocessing.image.array_to_img(img)
        img_list.append(img)

        img_norm = datagen.standardize(img)
        img_array = np.expand_dims(img_norm, axis=0)
        img_array_list.append(img_array)

    preds = model.predict(img_array_list)
    pred_class = preds[-1].argmax(axis=-1)
    pred_class = pred_class[0]

    superpixels = skimage.segmentation.quickshift(img_list[k], kernel_size=4, max_dist=200, ratio=0.2)
    num_superpixels = np.unique(superpixels).shape[0]

    num_perturb = PERT
    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))

    def perturb_image(img, perturbation, segments):
        active_pixels = np.where(perturbation == 1)[0]
        mask = np.zeros(segments.shape)
        for active in active_pixels:
            mask[segments == active] = 1
        perturbed_image = copy.deepcopy(img)
        perturbed_image = perturbed_image * mask[:, :, np.newaxis]
        return perturbed_image

    resnet_name = "resnet50_{0}".format(branch)
    resnet = model.get_layer(resnet_name)
    lime_resnet = tf.keras.Model(inputs=resnet.input, outputs=resnet.output, name='lime_resnet50_{0}'.format(branch))

    input_layer = tf.keras.Input(shape=(224, 224, 3), name="lime_input")
    lime_res = lime_resnet(input_layer)
    lime_pred = tf.keras.layers.Dense(2, activation='softmax', name="lime_pCR")(lime_res)
    lime_model = tf.keras.Model(inputs=input_layer, outputs=lime_pred, name="lime_{0}_model".format(branch))

    predictions = []
    for pert in perturbations:
        perturbed_img = perturb_image(img_list[k], pert, superpixels)
        pred = lime_model.predict(perturbed_img[np.newaxis, :, :, :])
        predictions.append(pred)
    predictions = np.array(predictions)

    kernel_width = 0.25
    num_top_features = LIME
    original_image = np.ones(num_superpixels)[np.newaxis, :]  # Perturbation with all superpixels enabled
    distances = pairwise_distances(perturbations, original_image, metric='cosine').ravel()
    weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))  # Kernel function

    simpler_model = LinearRegression()
    simpler_model.fit(X=perturbations, y=predictions[:, :, pred_class], sample_weight=weights)
    coeff = simpler_model.coef_[0]
    top_features = np.argsort(coeff)[-num_top_features:]

    mask = np.zeros(num_superpixels)
    mask[top_features] = True  # Activate top superpixels

    img_orig = img_list[k].rotate(270)
    img_orig = img_orig.transpose(Image.FLIP_LEFT_RIGHT)

    img_super = skimage.segmentation.mark_boundaries(img_list[k], superpixels)
    img_super = skimage.transform.rotate(img_super, 90)
    img_super = np.flip(img_super, 0)
    img_super = tf.keras.preprocessing.image.array_to_img(img_super)

    img_pert = perturb_image(img_list[k], perturbations, superpixels).astype(np.uint8)
    img_pert = skimage.transform.rotate(img_pert, 90)
    img_pert = np.flip(img_pert, 0)

    pert_img_final = perturb_image(img_list[k], mask, superpixels).astype(np.uint8)
    pert_img_final = skimage.transform.rotate(pert_img_final, 90)
    pert_img_final = np.flip(pert_img_final, 0)
    pert_img_final = tf.keras.preprocessing.image.array_to_img(pert_img_final)

    return mask


"""
    get_compile_dict creates a dictionary used by model.compile that assigns each branch used the correct loss function and its weight, as well as the appropriate metrics
"""
def get_compile_dict(multi_weight, loss_function):
    loss_dict = {}
    loss_weight_dict = {}
    metric_dict = {}

    for branch in accepted_branches:
        if branch in branches_list:
            if loss_function == 'focal_loss':
                loss_dict.update({branch: focal_loss(alpha=.25, gamma=2)})
            else:
                loss_dict.update({branch: loss_function})

            loss_weight_dict.update({branch: multi_weight})

            if loss_function == 'categorical_crossentropy':
                metric_dict.update({branch: ['accuracy', tf.keras.metrics.AUC(name='auc')]})
            else:
                metric_dict.update({branch: [tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                                             tf.keras.metrics.AUC(name='auc')]})
    if loss_function == 'focal_loss':
        loss_dict.update({'pCR': focal_loss(alpha=.25, gamma=2)})
    else:
        loss_dict.update({'pCR': loss_function})

    loss_weight_dict.update({'pCR': 1.})

    if loss_function == 'categorical_crossentropy':
        metric_dict.update({'pCR': ['accuracy', tf.keras.metrics.AUC(name='auc')]})
    else:
        metric_dict.update(
            {'pCR': [tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'), tf.keras.metrics.AUC(name='auc')]})

    return loss_dict, loss_weight_dict, metric_dict


"""
    Keras custom callback to plot results and validation metrics
"""
class Plot(Callback):
    def __init__(self, val_data, folder_time, fold):
        self.val_data_x = val_data[0]
        self.val_data_y = val_data[1]
        self.folder_time = folder_time
        self.fold = fold + 1

    def on_train_begin(self, logs={}):
        self.i = 1
        self.x = []
        self.losses = []
        self.auc = []
        self.bin_acc = []
        self.val_losses = []
        self.val_auc = []
        self.val_bin_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.auc.append(logs.get('pCR_auc'))
        self.bin_acc.append(logs.get('pCR_binary_accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_auc.append(logs.get('val_pCR_auc'))
        self.val_bin_acc.append(logs.get('val_pCR_binary_accuracy'))

        self.x.append(self.i)
        self.i += 1

    def on_train_end(self, epoch, logs={}):
        # Loss/AUC graph
        trace_loss = go.Scatter(x=self.x, y=self.losses, name="Training loss")
        trace_val_loss = go.Scatter(x=self.x, y=self.val_losses, name="Validation loss")
        trace_auc = go.Scatter(x=self.x, y=self.auc,
                               name="Training auc", yaxis='y2')
        trace_val_auc = go.Scatter(x=self.x, y=self.val_auc,
                                   name="Validation auc", yaxis='y2')
        data = [trace_loss, trace_auc, trace_val_loss, trace_val_auc]

        title_string = "Loss and AUC in each epoch - Slices for each sequence: {0} - Fold: {1}".format(SLICES * 2 + 1,
                                                                                                       self.fold)
        layout = dict(title=title_string,
                      xaxis=dict(title='Epoch'),
                      yaxis=dict(title='loss'),
                      yaxis2=dict(title='AUC', overlaying='y', side='right'))
        fig = go.Figure(data=data, layout=layout)
        # fig.show()

        save_path = "images/{0}/loss_auc_slices{1}_fold{2}.png".format(self.folder_time, (SLICES * 2 + 1), self.fold)
        # fig.write_image(save_path)

        # Loss/accuracy graph
        trace_loss = go.Scatter(x=self.x, y=self.losses, name="Training loss")
        trace_val_loss = go.Scatter(x=self.x, y=self.val_losses, name="Validation loss")
        trace_acc = go.Scatter(x=self.x, y=self.bin_acc,
                               name="Training accuracy", yaxis='y2')
        trace_val_acc = go.Scatter(x=self.x, y=self.val_bin_acc,
                                   name="Validation accuracy", yaxis='y2')
        data = [trace_loss, trace_acc, trace_val_loss, trace_val_acc]

        title_string = "Loss and accuracy in each epoch - Slices for each sequence: {0} - Fold: {1}".format(
            SLICES * 2 + 1, self.fold)
        layout = dict(title=title_string,
                      xaxis=dict(title='Epoch'),
                      yaxis=dict(title='loss'),
                      yaxis2=dict(title='accuracy', overlaying='y', side='right'))
        fig = go.Figure(data=data, layout=layout)
        # fig.show()

        save_path = "images/{0}/loss_accuracy_slices{1}_fold{2}.png".format(self.folder_time, (SLICES * 2 + 1),
                                                                            self.fold)
        # fig.write_image(save_path)

        # ROC
        Y_prob = self.model.predict(self.val_data_x)
        roc = roc_auc_score(self.val_data_y[0].argmax(axis=-1), Y_prob[-1][:, 1])
        FPR_AUC, TPR_AUC, thresholds = roc_curve(self.val_data_y[0].argmax(axis=-1), Y_prob[-1][:, 1])
        print("Validation AUC: {0:.2f}\n".format(roc))


"""
    show_img_samples displays a sample for each input sub_sequence
"""
def show_img_samples(X, type, fold, folder_time):
    if fold == 0:
        import math
        col = int(len(X) / 2)
        fig = make_subplots(rows=2, cols=col)
        for i, sub_sequence in enumerate(X):
            img = sub_sequence[0]

            img = np.rot90(img, 3)
            img = np.flip(img, 1)

            i = i + 1
            if (i % 2) == 0:
                i = int(i / 2)
                fig.add_trace(go.Image(z=img), 2, i)
            else:
                i = math.ceil(i / 2)
                fig.add_trace(go.Image(z=img), 1, i)
        if len(X) == 2:
            fig.update_layout(
                title_text='Sample {0}'.format(type),
                yaxis1={'title': 'MRI_1'},
                yaxis2={'title': 'MRI_2'})
        if len(X) == 4:
            fig.update_layout(
                title_text='Sample {0}'.format(type),
                yaxis1={'title': 'MRI_1'},
                yaxis3={'title': 'MRI_2'})
        if len(X) == 6:
            fig.update_layout(
                title_text='Sample {0}'.format(type),
                yaxis1={'title': 'MRI_1'},
                yaxis4={'title': 'MRI_2'})
        if len(X) == 8:
            fig.update_layout(
                title_text='Sample {0}'.format(type),
                yaxis1={'title': 'MRI_1'},
                yaxis5={'title': 'MRI_2'})
        # fig.show()

# MAIN
def main():
    # Main Setup
    folder_time = folder_name + datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
    if not os.path.exists("images"):
        os.mkdir("images")
    if not os.path.exists("images/{0}".format(folder_time)):
        os.mkdir("images/{0}".format(folder_time))
    print("Experiment timestamp:", folder_time, "\n")
    print("Extra slices for each scan:", SLICES * 2)
    print("Branches: ", branches_list, "\n")

    folder_filepath = 'tmp/{0}/'.format(folder_time)
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    if not os.path.exists(folder_filepath):
        os.mkdir(folder_filepath)

    # Train sequences loaded only once
    train_folders = scan_folders()

    val_results = []
    roc_slice = []
    roc_patient = []

    # Kfold CV
    Kfold_list = Kfold_split(train_folders, FOLDS)
    for i, fold in enumerate(Kfold_list):
        checkpoint_filepath = 'tmp/{0}/fold_{1}/checkpoints/'.format(folder_time, i + 1)
        model_filepath = 'tmp/{0}/fold_{1}/model/'.format(folder_time, i + 1)
        if not os.path.exists(folder_filepath):
            os.mkdir(checkpoint_filepath)
            os.mkdir(model_filepath)

        features, labels, scan_list_train, selected_slices = define_subsequences(fold[0], i)
        if selected_slices != SLICES:
            print("Too many extra slices for each scan selected, num set to max:", selected_slices * 2, "\n")
            exit()

        X_train, Y_train = define_input(features, labels)

        if WEIGHT == 1:
            y_integers = np.argmax(Y_train, axis=1)
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
            class_weights = dict(enumerate(class_weights))
            class_weight_dict = {}
            for branch in accepted_branches:
                if branch in branches_list:
                    class_weight_dict.update({branch: class_weights})
            class_weight_dict.update({'pCR': class_weights})

        features, labels, scan_list_val, selected_slices = define_subsequences(fold[1], i)
        if selected_slices != SLICES:
            print("Too many extra slices for each scan selected, num set to max:", selected_slices * 2, "\n")
            exit()

        X_val, Y_val = define_input(features, labels)

        print("Train shape:")
        for info_x in X_train:
            print(info_x.shape)
        print("\nValidation shape:")
        for info_x in X_val:
            print(info_x.shape)
        print("\nLabels shape:")
        print(Y_train.shape)
        print(Y_val.shape)

        show_img_samples(X_train, "input", i, folder_time)

        model = define_model()

        datagen_norm = ImageDataGenerator(samplewise_center=True,
                                          samplewise_std_normalization=True)

        X_val = normalize_test(datagen_norm, X_val)
        Y_val_list = []
        print(branches_list)
        for branch in range(len(branches_list)):
            # append Y for single branch task
            Y_val_list.append(Y_val)
        # append Y for pCR task
        Y_val_list.append(Y_val)
        val_input = (X_val, Y_val_list)

        datagen_train = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
                                           samplewise_center=True,
                                           samplewise_std_normalization=True,
                                           vertical_flip=True,
                                           zoom_range=0.3,
                                           brightness_range=[0.5, 3],
                                           shear_range=0.3)
        
        train_aug = datagen_flow(datagen_train, X_train, Y_train)

        loss_dict, loss_weight_dict, metric_dict = get_compile_dict(secondary_tasks_weight, loss_function)

        optimizer = tf.keras.optimizers.SGD(learning_rate=LR, momentum=momentum)
        model.compile(loss=loss_dict,
                      loss_weights=loss_weight_dict,
                      optimizer=optimizer, metrics=metric_dict)

        # Keras callbacks list
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            save_weights_only=True,
            save_best_only=True,
            verbose=1)

        model_earylstop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1)

        rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.5,
                                                     patience=5,
                                                     min_delta=0.000025,
                                                     verbose=1)
        
        if early_stop == 1:
            callbacks_list = [model_checkpoint_callback, model_earylstop_callback, rlrop,
                                Plot(val_input, folder_time, i)]
        else:
            callbacks_list = [model_checkpoint_callback, rlrop, Plot(val_input, folder_time, i)]

        if load_weights == 1:
            weights_str = "weights/fold_{0}/checkpoints/".format(i + 1)
            if not os.path.exists(weights_str):
                print("Cannot find proper filepath to weights folder in this fold, expected filepath:", weights_str,
                      "\n")
                exit()
            else:
                model.load_weights(weights_str)
        else:
            if loss_function == 'focal_loss' or WEIGHT == 0:
                model.fit(train_aug, steps_per_epoch=X_train[0].shape[0] // BATCH_SIZE, epochs=NB_EPOCH, verbose=0,
                          validation_data=val_input, callbacks=callbacks_list)
            else:
                model.fit(train_aug, steps_per_epoch=X_train[0].shape[0] // BATCH_SIZE, epochs=NB_EPOCH, verbose=1,
                          validation_data=val_input, class_weight=class_weight_dict, callbacks=callbacks_list)

        results = model.evaluate(X_val, Y_val_list, verbose=1)
        results = dict(zip(model.metrics_names, results))
        val_results.append(results)

        Y_probs = model.predict(X_val)

        output_names = []
        for branch in accepted_branches:
            if branch in branches_list:
                output_names.append(branch)
        output_names.append("pCR")

        slice_dict = {}
        patient_dict = {}

        for k, Y_prob in enumerate(Y_probs):
            output_name = output_names[k]
            check_multi = output_name.split("_")
            if "multi" not in check_multi:
                roc_result_slice = test_predict(Y_val, Y_prob, "slice", output_name, folder_time, i)
                slice_dict[output_name] = roc_result_slice
                Y_val_split, Y_prob_split = get_patient_level(Y_val_list[k], Y_prob)
                roc_result_patient = test_predict(Y_val_split, Y_prob_split, "patient", output_name, folder_time, i)
                patient_dict[output_name] = roc_result_patient
            # else:
            #     mse_result_slice, mse_result_patient = test_predict_multi(Y_val_list[k], Y_prob, output_name,
            #                                                               folder_time, i, feature_multi)
            #     mse_slice_dict[output_name] = mse_result_slice
            #     mse_patient_dict[output_name] = mse_result_patient

        roc_slice.append(slice_dict)
        roc_patient.append(patient_dict)

        num = SLICES * 2 + 1
        tot_patients = int(len(Y_train) / num)
        x = np.random.randint(0, tot_patients)
        for k, branch in zip(range(0, len(branches_list) * 2, 2), branches_list):
            #get_gradcam_heatmap(X_train[k], model, branch, folder_time, i, x)
            pass

        get_lime(X_train, model, i, x, datagen_norm, 0)

        # Other way to produce LIME results
        """for k, branch in zip(range(0, len(branches_list)*2, 2), branches_list):
            if FEATURES == 1:
                get_lime_2(X_train, model, i, x, branch, k, datagen_norm, X_features)
            else:
                get_lime_2(X_train, model, i, x, branch, k, datagen_norm, 0)"""

        if load_weights != 1:
            model.save(model_filepath, overwrite=True, include_optimizer=True)

        tf.keras.backend.clear_session()

    # KFold END -----------------------------------------------------------------------------------------
    # Generate utilities and log on neptune
    save_path = "images/{0}/model_plot.png".format(folder_time)
    tf.keras.utils.plot_model(model, to_file=save_path, show_shapes=True)

    # Get mean metrics from k models (model.evaluate) - These metrics are stored but not used for final evaluation, less accurate
    for key in val_results[0].keys():
        std_array = []
        for n,d in enumerate(val_results):
            std_array.append(d[key])
        mean_value = sum(d[key] for d in val_results) / len(val_results)
        str_value = 'mean_{0} = {1}'.format(key, mean_value)
        std_value = np.std(std_array)
        str_std_value = 'std_{0} = {1}'.format(key, std_value)
        print(str_value)
        print(str_std_value)
    print()

    # Get mean metrics from k models (roc_auc_score) - These metrics are used for the final evaluation reported
    log_list = ['auc', 'accuracy', 'sensitivity', 'specificity']
    for key in roc_slice[0].keys():
        std_array_patient = []
        if key == 'pCR':
            for d in roc_patient:
                std_array_patient.append(d[key][0])
        mean_value = sum(d[key] for d in roc_patient) / len(roc_patient)
        std_value = np.std(std_array_patient)
        for i, log in enumerate(log_list):
            if log == 'auc' and key == 'pCR':
                #BOH
                pass
            str_value = '{0}_patient-level {1} mean = {2}'.format(log, key, mean_value[i])
            print(str_value)
            if log == 'auc' and key == 'pCR':
                str_std_value = '{0}_patient-level {1} std = {2}'.format(log, key, std_value)
                print(str_std_value)
        std_array_slice = []
        if key == 'pCR':
            for d in roc_slice:
                std_array_slice.append(d[key][0])
        mean_value = sum(d[key] for d in roc_slice) / len(roc_slice)
        std_value = np.std(std_array_slice)
        for i, log in enumerate(log_list):
            if log == 'auc' and key == 'pCR':
                #BOH
                pass
            str_value = '{0}_slice-level {1} mean = {2}'.format(log, key, mean_value[i])
            print(str_value)
            if log == 'auc' and key == 'pCR':
                str_std_value = '{0}_slice-level {1} std = {2}'.format(log, key, std_value)
                print(str_std_value)

    print()


if __name__ == "__main__":
    main()
