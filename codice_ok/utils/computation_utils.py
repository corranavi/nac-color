import configs

import torch
import numpy as np
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, roc_curve, pairwise_distances

SLICES = configs.SLICES


def focal_loss(alpha = .25, gamma = 2):
    
    def focal_loss_fixed(y_true, y_pred):
        
        pt_1 = torch.where(torch.eq(y_true, 1), y_pred, torch.ones_like(y_pred))
        pt_0 = torch.where(torch.eq(y_true, 0), y_pred, torch.zeros_like(y_pred))

        return -torch.mean(alpha * torch.pow(1.-pt_1, gamma) * torch.log(pt_1)) + \
                - torch.mean((1-alpha) * torch.pow(pt_0, gamma) * torch.log(1.-pt_0))
    
    return focal_loss_fixed
    
    #TODO 
    #suggestion - usa quella già implementata [VUOLE I LOGITS]
    # https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html#torchvision.ops.sigmoid_focal_loss

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