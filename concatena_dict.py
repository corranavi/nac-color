import numpy as np
import pandas as pd

num_folds = 4

dict_1 = {
    "casa": np.array([1.,0.,3.,1.,2.]),
    "giardino": np.array([1.,0.,3.,1.,2.])
}
dict_2 = {
    "casa": np.array([6.,4.,1.,0.,9.]),
    "giardino": np.array([5.,2.,4.,2.,4.])
}

lista_di_dicts_1 = []
for i in range(num_folds):
    lista_di_dicts_1.append(dict_1)
lista_di_dicts_2 = []
for i in range(num_folds):
    lista_di_dicts_2.append(dict_2)

# ----------------------------- fin qua ho più o meno il mio dataset.
# metrics_dict = {}

# branches_list = list(lista_di_dicts[0].keys())
# print(branches_list)
#metriche =["AUC", "Accuracy", "Specificity","Sensitivity", "f1score"]

# mega_dict = {}
# for k in branches_list:
#     mega_dict[k]={}

# for branch in branches_list:
#     auc_values = np.array([])
#     acc_values = np.array([])
#     spe_values = np.array([])
#     se_values = np.array([])
#     f1_values = np.array([])
#     for dizionario in lista_di_dicts:
#         auc_values = np.append(auc_values, dizionario[branch][0])
#         acc_values= np.append(acc_values, dizionario[branch][1])
#         spe_values= np.append(spe_values, dizionario[branch][2])
#         se_values= np.append(se_values, dizionario[branch][3])
#         f1_values= np.append(f1_values, dizionario[branch][4])

#     auc_mean = np.mean(auc_values)
#     auc_std = np.std(auc_values)
#     acc_mean = np.mean(acc_values)
#     acc_std = np.std(acc_values)
#     spe_mean = np.mean(spe_values)
#     spe_std = np.std(spe_values)
#     se_mean = np.mean(se_values)
#     se_std = np.std(se_values)
#     f1_mean = np.mean(f1_values)
#     f1_std = np.std(f1_values)

#     mega_dict[branch]["auc_values"] = auc_values
#     mega_dict[branch]["auc_mean"] = auc_mean
#     mega_dict[branch]["auc_std"] = auc_std

#     mega_dict[branch]["acc_values"] = acc_values
#     mega_dict[branch]["acc_mean"] = acc_mean
#     mega_dict[branch]["acc_std"] = acc_std

#     mega_dict[branch]["spe_values"] = spe_values
#     mega_dict[branch]["spe_mean"] = spe_mean
#     mega_dict[branch]["spe_std"] = spe_std

#     mega_dict[branch]["se_values"] = se_values
#     mega_dict[branch]["se_mean"] = se_mean
#     mega_dict[branch]["se_std"] = se_std

#     mega_dict[branch]["f1_values"] = f1_values
#     mega_dict[branch]["f1_mean"] = f1_mean
#     mega_dict[branch]["f1_std"] = f1_std

# df = pd.DataFrame(mega_dict)
# df_trans = df.transpose()
# print(df_trans)

def export_result_as_df(slice_dictionaries, patient_dictionaries):
    
    lists_of_dictionaries = [slice_dictionaries, patient_dictionaries]
    levels = ["SLICE", "PATIENT"]
    #branches_list = ["pCR", "DWI_probs", "T2_probs", "DCEpeak_probs", "DCE3TP_probs"]
    branches_list = list(slice_dictionaries[0].keys())
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
                spe_values= np.append(spe_values, fold_dict[branch][2])
                se_values= np.append(se_values, fold_dict[branch][3])
                f1_values= np.append(f1_values, fold_dict[branch][4])

            print(branch)
            print("EGGIA")
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

            dataframe_dict[branch]["spe_values"] = spe_values
            dataframe_dict[branch]["spe_mean"] = spe_mean
            dataframe_dict[branch]["spe_std"] = spe_std

            dataframe_dict[branch]["se_values"] = se_values
            dataframe_dict[branch]["se_mean"] = se_mean
            dataframe_dict[branch]["se_std"] = se_std

            dataframe_dict[branch]["f1_values"] = f1_values
            dataframe_dict[branch]["f1_mean"] = f1_mean
            dataframe_dict[branch]["f1_std"] = f1_std

        df = pd.DataFrame(dataframe_dict)
        df_t = df.transpose()
        output_list.append(df_t)
    
    df1 = output_list[0]
    df2 = output_list[1]
    with pd.ExcelWriter("analisi_dei_risultati_aggregati.xlsx") as writer:
        df1.to_excel(writer, sheet_name=f'{levels[0]}_level')
        df2.to_excel(writer, sheet_name=f'{levels[1]}_level')
        

export_result_as_df(lista_di_dicts_1, lista_di_dicts_2)