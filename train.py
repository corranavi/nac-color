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
from utils.train_utils import retrieve_folders_list, Kfold_split, export_result_as_df, define_model
from dataset_lib import MRIDataModule

if __name__ =="__main__":

    args = parse_arguments()

    # SEED
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pl.seed_everything(args.seed, workers=True)     #https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
    
    # RETRIEVE DATASET PATHS
    folders_list = retrieve_folders_list(args.input_path)
    folder_time = args.architecture+"_"+args.exp_name+"_"+ datetime.now().strftime('%Y-%m-%d_%H-%M')
    datasets_list = Kfold_split(folders_list, args.folds)

    roc_slice = [] 
    roc_patient = [] 

    for i, fold in enumerate(datasets_list):
        print(f"\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nESECUZIONE fold # {i+1} \n")
        fold_start = datetime.now()
        lr_label = len(str(args.learning_rate).split('.')[1])
        wd_label = len(str(args.l2_reg).split('.')[1])
        logger_wandb = WandbLogger(save_dir= f"wandb_logs_{i}" , name=f"{args.exp_name[:3]}_{args.architecture[:4]}_lr{lr_label}_wd{wd_label}_fold_{i+1}_seed{args.seed}", project=args.wanb_project_name)
        dm = MRIDataModule(fold[0], fold[1], slices=args.slices, batch_size=args.batch, preprocess=args.preprocess)
        
        print(f"CLASS WEIGHTS: {dm.class_weights}")
        class_weights = dm.class_weights if args.class_weight == 1 else None

        # RETRIEVE LIGHTNING TORCH MODEL
        litmodel = define_model(fold=i, args=args, class_weights=class_weights, folder_time=folder_time)

        cb_list = get_callbacks(checkpoint = True, earlystop = False, lr_monitor = True, 
                                fold_num=i+1 ,args=args)

        # TRAINER INSTANCE
        trainer = pl.Trainer(
            logger = logger_wandb,
            accelerator = 'auto',
            default_root_dir='./LOGS',
            num_sanity_val_steps=0,
            precision='bf16-mixed', #bf16',
            min_epochs=1,
            max_epochs=args.epochs,
            check_val_every_n_epoch=1,
            callbacks=cb_list,  
            reload_dataloaders_every_n_epochs=1,
            deterministic=True
        )

        print(f"LR: {args.learning_rate}, WD: {args.l2_reg}")
        trainer.fit(model=litmodel, datamodule=dm)
        trainer.validate(model=litmodel, datamodule=dm)
        
        #ckpt_best = cb_list[0].best_model_path
        trainer.test(model=litmodel, datamodule=dm) #, ckpt_path=ckpt_best)
        final_model_path = os.path.join(f"./model_weights/{args.architecture}",f"Fold_{i+1}",f"trained_{args.architecture}_{args.exp_name}_lr{args.learning_rate}_wd{args.l2_reg}_seed{args.seed}.ckpt")
        
        trainer.save_checkpoint(final_model_path)
        wandb.finish()

        roc_patient.append(litmodel.patient_dict_test)
        roc_slice.append(litmodel.slice_dict_test)

        print(f"\n\nFOLD {i+1} TIME: {datetime.now()-fold_start}")
    
    #End of the Kfold
    print("CV concluded. Writing the results on file...")
    output_file_name = export_result_as_df(roc_slice, roc_patient, args)
    print(f"Results are available inside the file {output_file_name}")




