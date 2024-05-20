
from typing import Dict

import os
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
import numpy as np
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve, pairwise_distances
import plotly.graph_objs as go
#from plotly.subplots import make_subplots

accepted_branches = ["DWI","T2","DCE_peak","DCE_3TP"]
branches_list = ["DWI","T2","DCE_peak","DCE_3TP"]

# def load_all_validation_and_compute(model, val_loader, device = None):
#     """
#     Da provare se funziona....
#     """
#     Y_prob = torch.Tensor()
#     Y_test = torch.Tensor()

#     if device is None:
#         device = torch.device("cpu")
#     model = model.eval()

#     for idx, (features, labels) in enumerate(val_loader):
#         features, labels = features.to(device), labels.to(device)
#         features = features.permute(dims=(1,0, *range(2, features.dim())))
#         labels = labels.squeeze(dim=1)

#         with torch.no_grad():
#             logits = model(features)
        
#         #Y_prob = torch.cat([Y_prob, logits.unsqueeze(0)])
#         Y_test = torch.cat([Y_test, labels.unsqueeze(0)])
    
#     return Y_prob, Y_test

def load_all_validation(val_loader, device = None):
    """
  
    """
    Y_prob = torch.Tensor()
    Y_test = torch.Tensor()

    if device is None:
        device = torch.device("cpu")

    for idx, (features, labels) in enumerate(val_loader):
        features, labels = features.to(device), labels.to(device)
        features = features.permute(dims=(1,0, *range(2, features.dim())))
        labels = labels.squeeze(dim=1)
        
        Y_prob = torch.cat([Y_prob, features.unsqueeze(0)])
        Y_test = torch.cat([Y_test, labels.unsqueeze(0)])
    
    return Y_prob, Y_test

def compute_accuracy(model, dataloader, device = None):
    """
    Questo  è il metodo sul tutorial di pytorch / ligthningtorch -> devo riadattarlo al mio.
    
    """
    if device is None:
        device = torch.device("cpu")
    model = model.eval()

    correct = 0.0
    total_examples = 0

    for idx, (featues, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct/total_examples

def compute_loss(y_pred_dict: Dict, y_true: torch.Tensor, secondary_weight: float = 0.2, type="bce", weights = None):
    """
    Compute the loss function considering all the different branches and the corresponding weights.

    Args:
        y_pred_dict (dict): a dictionary containing as values the tensors with probabilities from each branch.
        y_true (torch.Tensor): the tensor containing the true labels.
        secondary_weight (float): the weight for the losses of the single modality branches.
        type (str): Whether to use binary cross entropy or sigmoid focal loss. Defaults to 'bce', binary cross entropy.
    Returns:
        loss (torch.Tensor): a tensor containing the resulting loss.
    """
    y_pred_DWI = y_pred_dict["DWI_probs"]
    y_pred_T2 = y_pred_dict["T2_probs"]
    y_pred_DCEpeak = y_pred_dict["DCEpeak_probs"]
    y_pred_DCE3TP = y_pred_dict["DCE3TP_probs"]
    y_pred_pCR = y_pred_dict["pCR"]

    #print(f"Check where the tensors are: Y_pred_DWI: {y_pred_DWI.device} | y_true: {y_true.device}")
    if weights is not None:
        pos_weight = torch.Tensor([weights[0]/weights[1], weights[1]/weights[0]]).to(weights.device) #nel caso funzionasse meglio, questo lo faccio già nei weights che passo in firma.
    else:
        pos_weight = None

    if type == "bce":
        loss_DWI = binary_cross_entropy_with_logits(y_pred_DWI, y_true, reduction="mean", pos_weight=pos_weight )#weight=weights) 
        loss_T2 = binary_cross_entropy_with_logits(y_pred_T2, y_true, reduction="mean", pos_weight=pos_weight)#weight=weights)
        loss_DCEpeak = binary_cross_entropy_with_logits(y_pred_DCEpeak, y_true, reduction="mean", pos_weight=pos_weight) #weight=weights)
        loss_DCE3TP = binary_cross_entropy_with_logits(y_pred_DCE3TP, y_true, reduction="mean", pos_weight=pos_weight)#weight=weights)
        loss_pcr = binary_cross_entropy_with_logits(y_pred_pCR, y_true, reduction="mean", pos_weight=pos_weight)#weight=weights)
    else:
        loss_DWI = sigmoid_focal_loss(y_pred_DWI, y_true, reduction="mean") 
        loss_T2 = sigmoid_focal_loss(y_pred_T2, y_true, reduction="mean")
        loss_DCEpeak = sigmoid_focal_loss(y_pred_DCEpeak, y_true, reduction="mean")
        loss_DCE3TP = sigmoid_focal_loss(y_pred_DCE3TP, y_true, reduction="mean")
        loss_pcr = sigmoid_focal_loss(y_pred_pCR, y_true, reduction="mean")

    print(f"loss_DWI: {loss_DWI}")
    print(f"loss_T2: {loss_T2}")
    print(f"loss_DCEpeak: {loss_DCEpeak}")
    print(f"loss_DCE3TP: {loss_DCE3TP}")
    print(f"loss_pcr: {loss_pcr}")

    loss = secondary_weight*loss_DWI + secondary_weight*loss_T2 + secondary_weight*loss_DCEpeak + secondary_weight*loss_DCE3TP + 1.* loss_pcr
    print(loss)
    
    return loss

def get_patient_level(Y_test: torch.Tensor, Y_prob: torch.Tensor, slices: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregates the probabilities tensors for patient, by applying means to the classes probs. 
    For the ground truth labels, the labels of the first sample for each patient are taken, after checking that all the slices for the same patient have the same
    labels (since they are labeled at patient level).\n
    Example:
        Y_test = [[1.,0.], [1.,0.]] \n
        Y_prob = [[0.43,0.21],[0.10,0.08]] \n
        >>\n
        Y_test_split = [[1.,0.]] \n
        Y_prob_split = [[.265, .145]]

    Args:
        Y_test (torch.Tensor): the tensor containing the true labels, at slice label. It has dimensions [num_patients x num_slices, 2]
        Y_prob (torch.Tensor): the tensor containing the predicted probabilities, at slice label. It has dimensions [num_patients x num_slices, 2]
    
    Returns:
        Y_test_split (torch.Tensor): the tensor containing the true labels, at patient level. It has dimensions [num_patients, 2]
        Y_prob_split (torch.Tensor): the tensor containing the predicted probabilities, at patient level. It has dimensions [num_patients, 2]
    """
    print(f"Samples: {len(Y_test)}")
    print(f"Slices: {slices*2+1}")
    patient_num = len(Y_test) // (slices * 2 + 1)
    sequence_length = slices * 2 + 1
    print(f"Patient num: {patient_num}")
    Y_test_split = torch.Tensor()
    Y_prob_split = torch.Tensor()

    Y_test_cpu = Y_test.cpu()
    split = torch.split(Y_test_cpu, sequence_length)

    for slice in split:
        #all_equal = all(pCR == slice.argmax(axis=-1)[0] for pCR in slice.argmax(axis=-1))   #tutte le labels dello stesso paziente devono essere uguali
        all_equal = all(pCR == torch.max(slice, dim=-1).indices[0] for pCR in torch.max(slice, dim=-1).indices)
        if all_equal:
            Y_test_split = torch.concat((Y_test_split, slice[0].unsqueeze(0)))
        else:
            print("Error in slices division per patient!")
            return

    Y_prob_cpu = Y_prob.cpu()
    split = torch.split(Y_prob_cpu, sequence_length)
    print(split)
                 #il voto di maggioranza è eseguito così: si fa la media delle probabilità per 0 e la media delle probabilità per 1, invece che passare già dalle predicted labels

    for slice in split:
        slice = torch.mean(slice, dim=0)
        Y_prob_split = torch.concat((Y_prob_split, slice.unsqueeze(0)))

    return Y_test_split, Y_prob_split

def test_predict(Y_test, Y_prob, level: str = "slice", output_name: str = None, folder_time: str = None, fold: int = 0, slices: int = 3):
    """
    Given the predicted probabilities and the true labels, compute Confusion matrix and derivated measures.
    Additionally, plot results and save them as png images.

    Returns:
        test_array (np.Array): a numpy array containing 4 metrics computed on the predictions: roc, ACC, TPR, TNR. 
    """
    print(Y_test.shape)
    #Y_test = Y_test.int()
    Y_test_cpu = Y_test.cpu()
    Y_prob_cpu = Y_prob.cpu()
    #.argmax(axis=-1) significa che calcola il massimo rispetto all'ultimo asse Example: [[8,1],[2,33]] --> [0,1]
    matrix = confusion_matrix(Y_test_cpu.argmax(axis=-1), Y_prob_cpu.argmax(axis=-1), labels=[0,1])
    roc = roc_auc_score(Y_test_cpu.argmax(axis=-1), Y_prob_cpu[:, 1], labels=[0,1])
    # matrix = confusion_matrix(torch.max(Y_test, dim=-1).indices, torch.max(Y_prob, dim=-1).indices, labels=[0,1])
    # roc = roc_auc_score(torch.max(Y_test, dim=-1).indices, Y_prob[:, 1], labels=[0,1])

    TN = matrix[0][0]
    FN = matrix[1][0]
    TP = matrix[1][1]
    FP = matrix[0][1]

    def compute_ratio_safe(x,y):
        #try - non funziona, continua a scatenare l'error RuntimeWarning: invalid value encountered in scalar divide
        if (x+y)>0:
            z = (x/(x+y)).round(4)*100.0
        else:
            z = 100.0
        return z
    
    TPR = compute_ratio_safe(TP, FN) # Sensitivity, hit rate, recall, or true positive rate
    TNR = compute_ratio_safe(TN, FP) # Specificity or true negative rate
    PPV = compute_ratio_safe(TP, FP) # Precision or positive predictive value
    NPV = compute_ratio_safe(TN, FN) # Negative predictive value
    FPR = compute_ratio_safe(FP, TN) # Fall out or false positive rate
    FNR = compute_ratio_safe(FN, TP) # False negative rate
    FDR = compute_ratio_safe(FP, TP) # False discovery rate
    
    ACC = ((TP + TN) / (TP + FP + FN + TN)).round(4) * 100.0 # Overall accuracy

    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = (TP / (TP + FN)).round(4) * 100.0
    # # Specificity or true negative rate
    # TNR = (TN / (TN + FP)).round(4) * 100.0
    # # Precision or positive predictive value
    # PPV = (TP / (TP + FP)).round(4) * 100.0
    # # Negative predictive value
    # NPV = (TN / (TN + FN)).round(4) * 100.0
    # # Fall out or false positive rate
    # FPR = (FP / (FP + TN)).round(4) * 100.0
    # # False negative rate
    # FNR = (FN / (FN + TP)).round(4) * 100.0
    # # False discovery rate
    # FDR = (FP / (FP + TP)).round(4) * 100.0
    # # Overall accuracy
    # ACC = ((TP + TN) / (TP + FP + FN + TN)).round(4) * 100.0

    # ROC plot
    FPR_AUC, TPR_AUC, thresholds = roc_curve(Y_test_cpu.argmax(axis=-1), Y_prob_cpu[:, 1])
    #FPR_AUC, TPR_AUC, thresholds = roc_curve(torch.max(Y_test, dim=-1).indices, Y_prob[:, 1])
   
    trace_roc = go.Scatter(x=FPR_AUC, y=TPR_AUC, mode='lines', name="Model ROC (AUC={0:.2f})".format(roc))
    trace_random = go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='random guess', line=dict(dash='dash'))
    data = [trace_roc, trace_random]

    layout = dict(title='ROC: {0} - Level: {1} - Fold: {2}'.format(output_name, level, fold + 1),
                  xaxis=dict(title='False Positive Rate', constrain='domain'),
                  yaxis=dict(title='True Positive Rate', scaleanchor="x", scaleratio=1))

    fig = go.Figure(data=data, layout=layout)
    #fig.show() To stay commented while executing on HPC

    save_path = "images/{0}/{1}-level_{2}_ROC_slices{3}_fold{4}.png".format(folder_time, level, output_name,
                                                                            (slices * 2 + 1), fold + 1)
    directory = os.path.dirname(save_path)
    os.makedirs(directory, exist_ok=True)

    fig.write_image(save_path)

    if level == "slice":
        print("\nSLICE LEVEL")
    elif level == "patient":
        print("\nPATIENT LEVEL")
    print(output_name, "\n")

    pred_labels = "Fold: {0}\nModel: {1}\nReal labels: {2}\nPred labels: {3}\n".format(fold + 1, output_name,
                                                                                      #  Y_test.argmax(axis=-1),
                                                                                      #  Y_prob.argmax(axis=-1))
                                                                                      torch.max(Y_test, dim=-1).indices,
                                                                                      torch.max(Y_prob, dim=-1).indices)
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
