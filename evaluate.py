from datetime import datetime
import wandb
import os
import torch
import numpy as np
import sys
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from utils.configurations import parse_arguments
from utils.callbacks_utils import get_callbacks
from utils.train_utils import retrieve_folders_list, Kfold_split, export_result_as_df, define_model, retrieve_ckpt_path_for_evaluate
from dataset_lib import MRIDataModule

if __name__ =="__main__":

    args = parse_arguments()
    eval_start = datetime.now()
    # SEED
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pl.seed_everything(args.seed, workers=True)     #https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
    
    # RETRIEVE DATASET PATHS
    folders_list = retrieve_folders_list(args.input_path) #("C:\\Users\\c.navilli\\Desktop\\Prova\\dataset_mini") #
    folder_time = args.exp_name + datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
    datasets_list = Kfold_split(folders_list, args.folds)

    roc_slice = [] 
    roc_patient = [] 

    for i, fold in enumerate(datasets_list):
        fold_start = datetime.now()
        lr_label = len(str(args.learning_rate).split('.')[1])
        wd_label = len(str(args.l2_reg).split('.')[1])
        #logger_wandb = WandbLogger(save_dir= f"wandb_logs_{i}" , name=f"{args.exp_name}_{args.architecture[:4]}_lr{lr_label}_wd{wd_label}_fold_{i+1}", project="NACtry")
        dm = MRIDataModule(fold[0], fold[1], slices=args.slices, batch_size=args.batch, preprocess=args.preprocess)
        
        #print(f"CLASS WEIGHTS: {dm.class_weights}")
        class_weights = dm.class_weights if args.class_weight == 1 else None

        # RETRIEVE LIGHTNING TORCH MODEL
        litmodel = define_model(fold=i, args=args, class_weights=class_weights, folder_time=folder_time)

        #cb_list = get_callbacks(checkpoint = True, earlystop = False, lr_monitor = True, 
                                #fold_num=i+1 , exp_name= args.exp_name, architecture=args.architecture, preprocess=args.preprocess)

        # TRAINER INSTANCE
        trainer = pl.Trainer(
            #logger = logger_wandb,
            accelerator = 'auto',
            precision='bf16-mixed', #bf16',
            deterministic=True
        )
        
        ckpt_fold = retrieve_ckpt_path_for_evaluate(args.architecture, args.exp_name, args.preprocess, i)
        trainer.test(model=litmodel, datamodule=dm, ckpt_path=ckpt_fold)

        roc_patient.append(litmodel.patient_dict_test)
        roc_slice.append(litmodel.slice_dict_test)
    
    #End of the Kfold
    print("CV concluded. Writing the results on file...")
    output_file_name = export_result_as_df(roc_slice, roc_patient, args)
    print(f"Results are available inside the file {output_file_name}")
    print(f"Evaluation took: {datetime.now()-eval_start}.")




