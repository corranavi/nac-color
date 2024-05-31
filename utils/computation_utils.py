
from typing import Dict

import os
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits
import numpy as np
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve, pairwise_distances
import plotly.graph_objs as go
#from plotly.subplots import make_subplots

accepted_branches = ["DWI","T2","DCE_peak","DCE_3TP"]
branches_list = ["DWI","T2","DCE_peak","DCE_3TP"]


def compute_loss(y_pred_dict: Dict, y_true: torch.Tensor, secondary_weight: float = 0.2, type="cross_entropy", weights = None):
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
    
    if type == "bce":

        if weights is not None:
            pos_weight = torch.Tensor([weights[1]/weights[0], weights[0]/weights[1]]).to(weights.device)
        else:
            pos_weight = None
        loss_DWI = binary_cross_entropy_with_logits(y_pred_DWI, y_true, reduction="mean", pos_weight=pos_weight )
        loss_T2 = binary_cross_entropy_with_logits(y_pred_T2, y_true, reduction="mean", pos_weight=pos_weight)
        loss_DCEpeak = binary_cross_entropy_with_logits(y_pred_DCEpeak, y_true, reduction="mean", pos_weight=pos_weight) 
        loss_DCE3TP = binary_cross_entropy_with_logits(y_pred_DCE3TP, y_true, reduction="mean", pos_weight=pos_weight)
        loss_pcr = binary_cross_entropy_with_logits(y_pred_pCR, y_true, reduction="mean", pos_weight=pos_weight)

    elif type =="cross_entropy":
        ce_loss = torch.nn.CrossEntropyLoss(weight=weights,reduction='mean')

        loss_DWI = ce_loss(y_pred_DWI, y_true.argmax(axis=-1))
        loss_T2 = ce_loss(y_pred_T2, y_true.argmax(axis=-1))
        loss_DCEpeak = ce_loss(y_pred_DCEpeak, y_true.argmax(axis=-1))
        loss_DCE3TP = ce_loss(y_pred_DCE3TP, y_true.argmax(axis=-1))
        loss_pcr = ce_loss(y_pred_pCR, y_true.argmax(axis=-1))

    else:
        return torch.tensor([100.], dtype = torch.float32)
        # loss_DWI = sigmoid_focal_loss(y_pred_DWI, y_true, reduction="mean") 
        # loss_T2 = sigmoid_focal_loss(y_pred_T2, y_true, reduction="mean")
        # loss_DCEpeak = sigmoid_focal_loss(y_pred_DCEpeak, y_true, reduction="mean")
        # loss_DCE3TP = sigmoid_focal_loss(y_pred_DCE3TP, y_true, reduction="mean")
        # loss_pcr = sigmoid_focal_loss(y_pred_pCR, y_true, reduction="mean")

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
        all_equal = all(pCR == torch.max(slice, dim=-1).indices[0] for pCR in torch.max(slice, dim=-1).indices)
        if all_equal:
            Y_test_split = torch.concat((Y_test_split, slice[0].unsqueeze(0)))
        else:
            print("Error in slices division per patient!")
            return

    Y_prob_cpu = Y_prob.cpu()
    split = torch.split(Y_prob_cpu, sequence_length)
    #print(split)

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
    Y_test_cpu = Y_test.cpu()
    Y_prob_cpu = Y_prob.cpu()
    Y_prob_cpu = nn.Softmax(dim=1)(Y_prob_cpu)

    matrix = confusion_matrix(Y_test_cpu.argmax(axis=-1), Y_prob_cpu.argmax(axis=-1)) #, labels=[0,1])
    roc = roc_auc_score(Y_test_cpu.argmax(axis=-1), Y_prob_cpu[:, 1]) #, labels=[0,1])

    TN = matrix[0][0]
    FN = matrix[1][0]
    TP = matrix[1][1]
    FP = matrix[0][1]

    def compute_ratio_safe(x,y):
        if (x+y)>0:
            z = (x/(x+y)).round(4)*100.0
        else:
            z = -10000
        return z
    
    TPR = compute_ratio_safe(TP, FN) # Sensitivity, hit rate, recall, or true positive rate    (TP / (TP + FN))
    TNR = compute_ratio_safe(TN, FP) # Specificity or true negative rate                       (TN / (TN + FP))
    PPV = compute_ratio_safe(TP, FP) # Precision or positive predictive value                  (TP / (TP + FP))
    NPV = compute_ratio_safe(TN, FN) # Negative predictive value                               (TN / (TN + FN))
    FPR = compute_ratio_safe(FP, TN) # Fall out or false positive rate                         (FP / (FP + TN))
    FNR = compute_ratio_safe(FN, TP) # False negative rate                                     (FN / (FN + TP))
    FDR = compute_ratio_safe(FP, TP) # False discovery rate                                    (FP / (FP + TP))
    F1_SCORE = (2*PPV*TPR) / (PPV + TPR)
    
    ACC = ((TP + TN) / (TP + FP + FN + TN)).round(4) * 100.0 # Overall accuracy

    # ROC plot
    FPR_AUC, TPR_AUC, thresholds = roc_curve(Y_test_cpu.argmax(axis=-1), Y_prob_cpu[:, 1])
   
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

    intro_label = f"Fold: {fold+1} - Model: {output_name} - Level: {level}\n"
    pred_labels = "Real labels: {0}\nPred labels: {1}\n".format(torch.max(Y_test, dim=-1).indices, torch.max(Y_prob, dim=-1).indices)
                                                     
    confusion_matrix_string = "TN: {0}\nFN: {1}\nTP: {2}\nFP: {3}\n".format(TN, FN,TP, FP)

    confusion_matrix_deriv = "Accuracy: {0:.2f}%\nF1-Score: {7:.2f}%\nSensitivity (TPR): {1:.2f}%\nSpecificity (TNR): {2:.2f}%\nFall-out (FPR): {3:.2f}%\nFalse Negative Rate (FNR): {4:.2f}%\nPrecision (PPV): {5:.2f}%\nNegative Predictive Value (NPV): {6:.2f}%".format(
        ACC, TPR, TNR, FPR, FNR, PPV, NPV, F1_SCORE)

    print(intro_label)
    print(pred_labels)
    print(confusion_matrix_string)
    print(confusion_matrix_deriv)
    print("Val AUC: {0:.2f}\n".format(roc))

    test_array = [roc, ACC, TPR, TNR, F1_SCORE]
    return np.array(test_array)
