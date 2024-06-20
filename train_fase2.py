from datetime import datetime
import wandb
import os
import torch
import numpy as np
import sys
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from utils.configurations import parse_arguments, get_model_setup
from utils.callbacks_utils import get_callbacks, get_LR_scheduler
from utils.train_utils_old import retrieve_folders_list, Kfold_split, print_results, export_result_as_df, retrieve_ckpt_path, define_model
from model_fase2 import LightningModel
from model_MONO import MonobranchLitModel
from dataset_lib import MRIDataModule

if __name__ =="__main__":
    print("Start")
    args = parse_arguments()
    print(args.dropout)
    print(args.slices)
    print(args.fc)

    #SEED
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pl.seed_everything(args.seed, workers=True)     #https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
    
    folders_list = retrieve_folders_list(args.input_path)
    folder_time = args.exp_name + datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]

    roc_slice = [] 
    roc_patient = [] 

    datasets_list = Kfold_split(folders_list, args.folds)
    print(f"La lista dei dataset ha {len(datasets_list)} elementi")
    print(f"Le folds selezionate sono {args.folds}")

    for i, fold in enumerate(datasets_list):
        print(f"\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nESECUZIONE fold # {i+1} \n")
        fold_start = datetime.now()
        logger_wandb = WandbLogger(save_dir= f"wandb_logs_{i}" , name=f"{args.exp_name}_{args.architecture}_fold_{i+1}", project="NACtry")
        dm = MRIDataModule(fold[0], fold[1], slices=args.slices, batch_size=args.batch, preprocess=args.preprocess)
        
        print(f"CLASS WEIGHTS: {dm.class_weights}")
        class_weights = dm.class_weights if args.class_weight == 1 else None

        litmodel = define_model(fold=i, args=args, class_weights=class_weights, folder_time=folder_time)

        cb_list = get_callbacks(checkpoint = True, earlystop = False, lr_monitor = True, 
                                fold_num=i+1 , exp_name= args.exp_name, architecture=args.architecture, preprocess=args.preprocess)

        #Istantiate a trainer 
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
        
        ckpt_best = cb_list[0].best_model_path
        print(f"Best checkpoint path: {ckpt_best}")
        
        trainer.test(model=litmodel, datamodule=dm, ckpt_path=ckpt_best)
        wandb.finish()

        print(f"Linea 66")
        roc_patient.append(litmodel.patient_dict_test)
        roc_slice.append(litmodel.slice_dict_test)
        print("Linea 69")

        print(f"\n\nFOLD {i+1} TIME: {datetime.now()-fold_start}")
    
    #End of the Kfold
    print("CV concluded. Writing the results on file...")
    output_file_name = export_result_as_df(roc_slice, roc_patient, args)
    print(f"Results are available inside the file {output_file_name}")




