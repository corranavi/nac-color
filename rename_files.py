import os

# Script per rinominare i file in modo che il numero di immagine sia su tre cifre e 
# questo non causi errori nelle operazioni di sorting.

root_dir = "C:\\Users\\c.navilli\\Desktop\\Prova\\NAC_Input"
for paziente in os.listdir(root_dir):
    paziente_dir = os.path.join(root_dir,paziente)
    for pre_post_nac in os.listdir(paziente_dir):
        mri_dir = os.path.join(paziente_dir, pre_post_nac)
        for taglio_scan in os.listdir(mri_dir):
            taglio_dir = os.path.join(mri_dir, taglio_scan)
            for old_file_name in os.listdir(taglio_dir):
                
                ext = old_file_name.split(".")[-1]
                num = int(old_file_name.split(".")[0].split('_')[-1])
                new_file_name = old_file_name.split('_')[0] + f"_{num:03d}."+ext
                
                old_path = os.path.join(root_dir, paziente_dir, mri_dir, taglio_dir,old_file_name)
                new_path = os.path.join(root_dir, paziente_dir, mri_dir, taglio_dir,new_file_name)
                print(old_path," ---> ", new_path)

                os.rename(old_path, new_path)
                
            print("")
        print("")
    print("")
print("")