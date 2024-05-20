from datetime import datetime
import wandb
import os
import torch
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from utils.configurazioni import parse_arguments
from utils.callbacks_utils import get_callbacks
from utils.train_utils import retrieve_folders_list, Kfold_split, print_results
from model import LightningModel
from dataset_lib import MRIDataModule

if __name__ =="__main__":
    args = parse_arguments()
    print(args.dropout)
    print(args.slices)
    print(args.fc)

    #SEED
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #dataset_relative_path = args.input_path
    #current_dir = os.path.dirname(os.path.abspath(__file__))
    #dataset_folder = os.path.join(current_dir, dataset_relative_path )
    #print(dataset_folder)
    
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
        
        #logger = TensorBoardLogger(f"tb_logs_{i}", name="mri_model_v0")
        logger_wandb = WandbLogger(save_dir= f"wandb_logs_{i}" , name=f"mri_model_MINI_fold_{i+1}", project="NACtry")
        dm = MRIDataModule(fold[0], fold[1], slices=args.slices, batch_size=args.batch)
        print(f"CLASS WEIGHTS: {dm.class_weights}")
        model = LightningModel(args.slices, args.fc, args.dropout, args.exp_name, args.optim, args.learning_rate, args.l2_reg, args.secondary_weight, dm.class_weights, folder_time, i)

        cb_list = get_callbacks(checkpoint = True, earlystop = True, lr_monitor = True)

        #Istantiate a trainer 
        trainer = pl.Trainer(
            logger = logger_wandb, #[logger, logger_wandb]
            accelerator = 'auto',
            default_root_dir='./LOGS',
            num_sanity_val_steps=0,
            precision='bf16',
            min_epochs=1,
            max_epochs=args.epochs,
            check_val_every_n_epoch=1,
            callbacks=cb_list, 
            reload_dataloaders_every_n_epochs=1
        )

        print(f"LR: {args.learning_rate}, WD: {args.l2_reg}")
        trainer.fit(model=model, datamodule=dm)
        trainer.validate(model=model, datamodule=dm)
        
        ckpt_best = cb_list[0].best_model_path
        print(f"Best checkpoint path: {ckpt_best}")

        trainer.test(model=model, datamodule=dm, ckpt_path=ckpt_best)
        wandb.finish()

        print(f"Linea 66")
        roc_patient.append(model.patient_dict_test)
        roc_slice.append(model.slice_dict_test)
        print("Linea 69")

        print(f"\n\nFOLD {i+1} TIME: {datetime.now()-fold_start}")
    
    #End of the Kfold
    print("Linea 72")
    print(f"ROC SLICE:\n{roc_slice}")
    print_results(roc_slice, roc_patient)
    print("Linea 74")
    print()




