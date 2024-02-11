import os
import ast
import SimpleITK as sitk
import numpy as np
import torch 

from torch.nn import functional as F

import logging

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

def retrieve_slice_and_bounds(scan_string: str, images_sequence:list, k:int) -> (int, int, int, int):

    # questa parte verrà poi rimossa, arriva già la sequenza delle immagini ----------
    dicom_files = []
    sequence = images_sequence
    images = []

    for r, d, f in os.walk(sequence):
            f.sort()
            for file in f:
                if '.dcm' in file or '.IMA' in file:
                    dicom_files.append(os.path.abspath(os.path.join(r,file)))

    for dicom_file in dicom_files:
        images.append(sitk.ReadImage(dicom_file)) 

    #---------------------------------------------------------------------------------

    num_of_slices = len(images)
    first_slice = np.inf

    for img in images:
        slice_id = int(img.GetMetaData('0020|0013'))
        if slice_id < first_slice:
            first_slice = slice_id
        if img.GetMetaData('0008|103e') == "IndexSlice":
            index_slice = slice_id

    #ipotizziamo che il for mi identifichi qual è l'index slice.
    last_slice = first_slice + (num_of_slices - 1) 

    #check on maximal index
    if (index_slice + k <= last_slice):
        upper_index = index_slice + k
    else:
        upper_index = last_slice
        k = last_slice - index_slice

    #check on minimal index
    if (first_slice <= index_slice - k):
        lower_index = index_slice - k
    else:
        lower_index = first_slice
        k = index_slice - first_slice
        upper_index = index_slice + k
    
    resulting_subsequence_length = (upper_index - lower_index) + 1

    return index_slice, lower_index, upper_index, resulting_subsequence_length


    # for img in images_sequence:
    #     print(img.GetMetaData('0008|103e'))
    #     if img.GetMetaData('0008|103e') == "IndexSlice":  #se la descrizione di questa immagine è "indexSlice", allora vado ad incrementare il contatore delle immagini idice
    #         index_count += 1
    #         image_count += 1
    #         index_instance = int(img.GetMetaData('0020|0013'))

    #         upper_bound = min(max_instance, (index_instance + extra_slices_per_side))
    #         lower_bound = max(min_instance, (index_instance - extra_slices_per_side))

    #         if (index_instance + extra_slices_per_side) > max_instance:
    #             diff_bound = extra_slices_per_side - (max_instance - index_instance)
    #             lower_bound = max(min_instance, (index_instance - extra_slices_per_side - diff_bound))

    #         if (index_instance - extra_slices_per_side) < 0:
    #             diff_bound = extra_slices_per_side - (index_instance)
    #             upper_bound = min(max_instance, (index_instance + extra_slices_per_side + diff_bound))
            
    #         if scan_string != "DCE":
    #             slices.append(sitk.GetArrayFromImage(img))  #trasformo l'immagine in array (numpy? da verificare. Devo poi convertire in tensore EH)
    #         else:
    #             slices.append(img) 

    return index_instance, lower_bound, upper_bound, index_count, image_count, slices

def define_subsequences(folders: "list[str]", fold):
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
    DWI_1, DWI_2, T2_1, T2_2, T1_1, T1_2 = ([] for i in range(6))   #_1 sta sempre per pre-NAC, _2 per post-NAC
    #sub_sequences = [DWI_1, DWI_2, T2_1, T2_2, T1_1, T1_2] #lista di liste

    sub_sequences_dict = {"DWI_1":DWI_1,"DWI_2": DWI_2, "T2_1": T2_1,"T2_2": T2_2,"T1_1": T1_1,"T1_2": T1_2}

    try:
        with open("labels/pCR.txt", "r") as file: #pCR.txt è il file contenente le ground truth di OGNI SEQUENZA
            contents = file.read()
            dict = ast.literal_eval(contents)
    except Exception as error:
        print("Cannot find file in labels/ folder, please put the file pCR.txt inside \n")
        if not os.path.exists("labels"):
            os.mkdir("labels")
        exit()
    
    for sequence in folders:
        sequence_count += 1       
        extra_slices_per_side = 4 #config.SLICES -- facciamo finta che dal parser ho imposto che le extra slices sono 4
        patient_flag = False      #....non ho ancora flaggato(?) quel paziente
        dicom_files = []          #una sequenza contiene più DICOM files?
        images = []               #lista di tutte le immagini recuperate dai dicom files
        slices = []               #lista degli indici di fette

        #adesso inizio a iterare sugli elementi della sequenza. 
        for r, d, f in os.walk(sequence):
            f.sort()
            for file in f:
                if '.dcm' in file or '.IMA' in file:
                    dicom_files.append(os.path.abspath(os.path.join(r,file)))
        print("\n",dicom_files[0],end="")  # così vedo quanti ce ne sono.... in una sequenza mi aspettavo di trovare solo un dcm file. Oppure ne può contenere di più? --> verificare sulle cartelle
                    
        #Adesso leggo l'immagine
        reader = sitk.ImageFileReader()
        reader.SetFileName(dicom_files[0]) #--> vedi che comunque prendi la prima?? E se ne ho più di una?
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        # questi 4 step sono necessari per istanziare e attivare il sitk ImageFileReader. Ora si può iniziare ad usarlo

        description = reader.GetMetaData("0008|103e").strip()
        #print(" - "*10,"DESCRIZIONE: ",description)
        name_num = description.split('_')[0] #!!!!! vai a vedere quale sarebbe il name_num e dagli un nome più parlante   "1" / "2" / ... / "37"
        if name_num != new_patient:    #    questo è per essere sicuro che la sequenza sia dello stesso / nuovo paziente
            new_patient = name_num
            patient_flag = True
        mri_string = description.split('_')[1]      #la stringa che identifica se sia MRI pre-nac o post nac: "MRI1" / "MRI2"
        scan_string = description.split('_')[2]     #la stringa che identifica il tipo di risonanza: "DCE" / "DWI" / "T2"

        if scan_string in ["DWI","T2","DCE"]: # in realtà questi tagli sono contenuti in config.scans:
            #Se il taglio è uno dei precedenti, allora lo analizza
            for dicom_file in dicom_files:
                images.append(sitk.ReadImage(dicom_file)) 
            slices_num = len(images) #la sequenza che ha individuato, di quante slice è composta?  ----------------------------------------------> Tutto sto check sul numero di slices si può mettere in funzione dedicata
            max_slices = int(((slices_num-1)/2))
            if extra_slices_per_side > max_slices:
                extra_slices_per_side = max_slices   #non posso chiederne di più del massimo disponibile, ovvero best case con slice al centro
            
            # Vado a settare i boundaries per quella sequenza
            max_instance = - np.inf
            min_instance = np.inf
            for img in images:  
                img_instance = int(img.GetMetaData('0020|0013'))
                if img_instance > max_instance:
                    max_instance = img_instance
                if img_instance < min_instance:
                    min_instance = img_instance

            for img in images:
                if img.GetMetaData('0008|103e') == "IndexSlice":  #se la descrizione di questa immagine è "indexSlice", allora vado ad incrementare il contatore delle immagini idice
                    index_count += 1
                    image_count += 1
                    index_instance = int(img.GetMetaData('0020|0013'))
                    print(index_instance)

                    upper_bound = min(max_instance, (index_instance + extra_slices_per_side))
                    lower_bound = max(min_instance, (index_instance - extra_slices_per_side))

                    if (index_instance + extra_slices_per_side) > max_instance:
                        diff_bound = extra_slices_per_side - (max_instance - index_instance)
                        lower_bound = max(min_instance, (index_instance - extra_slices_per_side - diff_bound))

                    if (index_instance - extra_slices_per_side) < 0:
                        diff_bound = extra_slices_per_side - (index_instance)
                        upper_bound = min(max_instance, (index_instance + extra_slices_per_side + diff_bound))

                    #if scan_string != "DCE":
                    #    slices.append(sitk.GetArrayFromImage(img))  #trasformo l'immagine in array (numpy? da verificare. Devo poi convertire in tensore EH)
                    #else:
                    #    slices.append(img)   #se il taglio è DCE (Dynamic Contrast Enhanced) allora la tengo così com'è perchè devo processarla dopo.
            #index_instance, lb, ub, index_count, image_count, slices = retrieve_slice_and_bounds(scan_string, images, extra_slice_per_side, min_instance, max_instance, slices, index_count, image_count)
            #print(f"Index slice: {index_slice}\nLower Bound: {lb}\nUpper Bound: {ub}")
            print(f"Per questa sequenza i bound sono: LB {lower_bound} - INDEX {index_instance} - UB {upper_bound}")    
            for img in images:
                #continuo a scansionare tutte le immagini della sequenza
                img_instance = int(img.GetMetaData('0020|0013'))

                #se l'indice di questa istanza è oltre a quella dell'INDICE SELEZIONATO + L'upper bound la ignoro ,altrimenti la prendo
                if img_instance >= index_instance and img_instance <= upper_bound:
                    image_count += 1
                    print(f"Presa {img_instance}")
                    if scan_string != "DCE":   #come sempre c'è un controllo: probabilmente legato alle immagini di tipo DCE che avranno un tipo diverso
                        float_slice = sitk.GetArrayFromImage(img)
                        slices.append(float_slice.astype(np.float32))
                    else:
                        slices.append(img)

                #se non è oltre l'indice, allora potrebbe essere una extra di quelle "prima" -> quindi c'è l'esatto check di prima. Anche qua si potrebbe ingegnerizzare diversamente
                if img_instance < index_instance and img_instance >= lower_bound:
                    image_count += 1
                    print(f"Presa {img_instance}")
                    if scan_string != "DCE":
                        float_slice = sitk.GetArrayFromImage(img)
                        slices.append(float_slice.astype(np.float32))
                    else:
                        slices.append(img)

                #print(f"La sequenza totale ora ha {len(slices)} immmagini.")
        #ora è finito il ciclo sulle immagini, le ho considerate tutte (all'interno della stessa sequenza).
        name_string = "NAC_" + name_num
        if name_string in dict:     #---> dict è un dizionario con tutte le label.
            label = list(str(dict[name_string]))  #--> qua dice: prendi le label, castale come stringhe e poi costruiscine una lista
        
        if patient_flag:         #se il patient_flag è true vuol dire che ho finito di considerare le immagini di un paziente                
            for i in range(len(slices)):
                labels.append(label)            #la stessa label è ripetuta per il numero di fette considerate
                scan_list.append(name_string)   #nella lista delle scansioni aggiungo "NAC_1" ecc ecc
                paz_list.append(name_num)

        #adesso faccio check su quale modello sto considerando, perchè ci sono modelli che usano solo alcuni branch rispetto ad altri. Ricorda infatti che stiamo parlando
                #di risonanze magnetiche MultiParametriche

        branches_list =["DWI","T2","DCE_peak","DCE_3TP"]
        #---------------per ora teniamo tutto commentato--------------------------------------------------------#
        
        if "DWI" in branches_list: #config.branches_list:  #nei branch ho specificato anche DWI? Ok
            if scan_string == "DWI":       #verifico che questa scansione sia relativa a DWI. Se sì,  
                if mri_string == "MRI1":   #      allora metto quelli che hanno MRI1 nella lista DWI_1, gli altri in DWI_2 (che sarebbero prima e dopo NAC)
                    DWI_1.extend(slices)   #      il fatto è che qua è una EXTEND, quindi devo capire quale sia il data type.
                else:                      #    .  
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

    #Infine, ho considerato tutte le sequenze, ovveo tutte le folder. Quindi posso chiudere.
    lista_dei_tagli = []
    for name,sub_sequence in sub_sequences_dict.items():  #ovvero [DWI_1, DWI_2, T2_1, T2_2, T1_1, T1_2], lista di liste
        lista_dei_tagli.append(name)
        features.append(sub_sequence)
        #quindi features sarà del tipo :  [DWI_1=[sottosequenza1,sottosequenza2,sottose...], DWI_2=[sottoseq1,sotto, ....] ovvero anche qui una bella lista di liste

    sub_sequence_div = (10/len(branches_list))/2   #config.branches_list
    #print("Fold:", fold+1)              Considera che c'è una funzione di fold che mi suddivide i dati ma..... al momento chissene
    print("Total DICOM Sequences:", round(sequence_count/sub_sequence_div))
    print("Total DICOM Index Slices:", round(index_count/sub_sequence_div))
    print("Total DICOM Selected Slices:", round(image_count/sub_sequence_div), "\n")
              
    #Le classi son bilanciate? ###########    
    print(class_balance(np.array(labels)))   #   
    ######################################    
    labels = np.array(labels, dtype=int)
    labels = torch.from_numpy(labels).to(torch.float32) #python uses Float64 in numpy -> Float32 is more efficient
    #print(labels.dtype)
    #print(lista_dei_tagli,labels, scan_list, extra_slices_per_side, paz_list)
    return features, labels, scan_list, extra_slices_per_side, paz_list

def class_balance(labels_array: np.array)->dict:
    """
    Count the frequences of the two classes.
    Parameters:
    - labels_array(np.array): array with the labels
    Returns:
    - dictionary with the count of the occurrencies for each label
    """
    counter_0 = 0
    counter_1 = 0
    for i in range(labels_array.shape[0]):
        if labels_array[i][0]=='0':
            counter_0+=1
        else:
            counter_1+=1
    return {'0':counter_0/(4*2+1), '1':counter_1/(4*2+1)}  #4*2+1 tiene conto di quante slice ho per ciascun paziente. 

def define_DCE(T1_1, T1_2):
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
        patient_time = np.array(patient_time)#.astype(np.float)
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
                    print(f"Tempo: {time}\n DCE_pre={DCE_pre}\nDCE_peak: {DCE_peak}")
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
                            print(i)
                            image_3TP = np.vstack((pre_array[i], peak_array[i], post_array[i])) #forse qua dstack
                            print(f"L'immagine DCE 3TP ha dimensioni: {image_3TP.shape}")
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
        patient_time = np.array(patient_time)#.astype(np.float)
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
                    print(f"Tempo: {time}\n DCE_pre={DCE_pre}\nDCE_peak: {DCE_peak}")
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
                            print(f"L'immagine DCE 3TP ha dimensioni: {image_3TP.shape}")
                            DCE_3TP2.append(image_3TP)

    for sub_sequence in sub_sequences:
        features.append(np.array(sub_sequence))
    return features

def define_input(patient_list, features, labels):
    branches_list =["DWI","T2","DCE_peak","DCE_3TP"] #new
    if "DCE_peak" or "DCE_3TP" in branches_list: #config.branches_list:
        DCE_features = define_DCE(features[-2], features[-1])
        features = features[:len(features)-2]
        for DCE_feature in DCE_features:
            features.append(DCE_feature)
    X_patient = []
    X = []

    for feature in features: 
        feature = torch.tensor(feature)
        if feature.any():
            X_sub = feature
            if feature.shape[1] == 1:
                X_sub = torch.repeat_interleave(X_sub,repeats = 3, dim=1)
            X_sub = X_sub.to(torch.float32) 
             # [(num di slice * num pazienti), C, H, W]
            X.append(X_sub)          #indice X[tipo_risonanza][numero_di_immagine][channels][height][weight]
    
    current_patient=''
    for feature in features:
        subsequence=[]
        for slice, patient in zip(feature,patient_list):
            if patient!=current_patient:
                current_patient=patient
                subsequence.append(slice)


    Y = F.one_hot(labels.to(torch.int64), 2) #config.NB_CLASSES)
    print(f"Shape of X: {len(X)} elements of shape {X[0].shape}, Of Y: {Y.shape}")
                 #Y ha dimensioni "2" perchè la label ha subito one-hot-encoding
    return X, Y

def rearrange_feature_list(features):
    final_features_list = [ ]
    positions=[(0,1),(2,3),(4,5),(6,7)]
    for couple in positions:
        print(f"Pair: {couple}")
        pre_nac_list = features[couple[0]]
        post_nac_list = features[couple[1]]

        print(f"Lunghezza della pre_nac_list: {len(pre_nac_list)}")
        print(f"Lunghezza della post_nac_list: {len(post_nac_list)}")

        #prendi i primi 2k+1 da pre_nac e poi concatena i successivi 2k+1 da post_nac
        pre_nac_shaded_list = []
        for i in range(0,len(pre_nac_list), 2*k+1):
            individual_patient=[]
            for j in range(i,i+2*k+1):
                individual_patient.append(pre_nac_list[j])
                print(f"LEN INDIVIDUAL PATIENT: {len(individual_patient)}")
            pre_nac_shaded_list.append(individual_patient)
        print(f"PRE NAC shaded list ha lunghezza: {len(pre_nac_shaded_list)}")

        post_nac_shaded_list = []
        for i in range(0,len(post_nac_list), 2*k+1):
            individual_patient=[]
            for j in range(i,i+2*k+1):
                individual_patient.append(post_nac_list[j])
            post_nac_shaded_list.append(individual_patient)
        print(len(post_nac_shaded_list))

        if len(pre_nac_shaded_list)!=len(post_nac_shaded_list):
            print("Le due liste non hanno la stessa lunghezza")
            exit()
        
        modality_joined = []

        print(f"Line 519 - {len(pre_nac_shaded_list)}")
        print(f"Line 520 - {len(post_nac_shaded_list)}")
        print(f"Il tipo è: {type(pre_nac_shaded_list)}")

        for patient_idx in range(len(post_nac_shaded_list)):
            print(f"Paziente: NAC_{patient_idx+1}")
            pre_nac_shaded_list[patient_idx].extend(post_nac_shaded_list[patient_idx])
            print(f"Lunghezza della lista di NAC_{patient_idx+1} post extend: {len(pre_nac_shaded_list[patient_idx])}")
            print("")
            modality_joined.append(pre_nac_shaded_list[patient_idx])

        final_features_list.append(modality_joined)

    print("Analisi delle dimensioni sulla final_features_list:")
    print(f"Mi aspetto siano 4: {len(final_features_list)}")

    print(f"La lista con tutte le features ha {len(final_features_list)} elementi. ")
    for i in range(len(final_features_list)):
        print(f"La modalità numero {i+1} ha {len(final_features_list[i])} liste, ovvero pazienti:")
        for j in range(len(final_features_list[i])):
            print(f"La sequenza {i+1} del paziente {j+1} ha {len(final_features_list[i][j])} slices.")
        print("")
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
    return full_stacked
    
    #---------------  Printing of shape infos:
    print(f"Stackata la lista di tensori. Le dimensioni del nuovo tensore sono: {full_stacked.shape}")
    print(f"A livello di paziente le dimensioni sono: {full_stacked[0].shape}.")
    print(f"\t- {full_stacked[0].shape[0]} modalities;\n\t- {full_stacked[0].shape[1]} immagini;\n\t- {full_stacked[0].shape[2]}x{full_stacked[0].shape[3]}x{full_stacked[0].shape[4]};")
    print(f"A livello di modalità le dimensioni sono: {full_stacked[0][0].shape}.")
    print(f"\t- {full_stacked[0][0].shape[0]} immagini;\n\t- {full_stacked[0][0].shape[1]}x{full_stacked[0][0].shape[2]}x{full_stacked[0][0].shape[3]};")
    print(f"A livello di singola immagine le dimensioni sono {full_stacked[0][0][0].shape}")
    print(f"\t- {full_stacked[0][0][0].shape[0]}x{full_stacked[0][0][0].shape[1]}x{full_stacked[0][0][0].shape[2]};")


if __name__ == "__main__":
    folder_list = retrieve_folders_list("C:\\Users\\c.navilli\\Desktop\\Prova\\dataset_mini")
    print(f"Folders scansionate. Totale: {len(folder_list)} liste.")
    for i,folder in enumerate(folder_list):
        print(i+1,") ",folder)
    print("")

    k=4
    maximal_subsequences_count = 0
    print("1. Defining Subsequences")
    features, labels, scan_list, extra_slices_per_side, patient_list = define_subsequences(folder_list,10)
    print(f"\n{len(features)}")
    print(f"FEATURES[4]: {features[4]}")

    summary_patients = []
    for i in range(0,len(patient_list), 2*k+1):
        summary_patients.append(patient_list[i])
    print(f"Lista pazienti: {summary_patients}")

    print("\n2. Defining inputs")
    X,Y = define_input(patient_list, features, labels)
 
    print("\n3. Rearrange inputs")
    rearrange_feature_list(X)
    #define_input(patient_list, features, labels)
    exit()

    for folder in folder_list:
        index_slice, lower_index, upper_index, resulting_subsequence_length = retrieve_slice_and_bounds("Ciao", folder, k)
        #print(f"Folder: {folder}\nIndex slice: {index_slice}\nFirst slice subsequence: {lower_index}\nLast slice subsequence: {upper_index}\nLength subseq: {resulting_subsequence_length}\n")
        if resulting_subsequence_length==2*k+1:
            maximal_subsequences_count+=1
    #print(f"\nPercentage of full sequences: {maximal_subsequences_count/len(folder_list):.2%}")
    #features,labels, scan_list, extra_slices_per_side = define_subsequences(folder_list,1)
        
#lista di 8 elementi [(2k+1)*numpaz, 3,225,225)]
# [num pazienti, [slices, canali, h, w] #DCE
            #    [slices, canali, h, w] #T2
            #    