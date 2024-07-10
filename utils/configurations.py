import os
import argparse

MODEL_SETUP = {
    "zero_level": {
        "COLORIZE": False,
        "FREEZE_BACKBONE": True
    },
    "baseline":{
        "COLORIZE": False,
        "FREEZE_BACKBONE": False
    },
    "colorization":{
        "COLORIZE": True,
        "FREEZE_BACKBONE": True
    },
    "all":{
        "COLORIZE": True,
        "FREEZE_BACKBONE": False
    }
}

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Deep Learning on NAC Data')
    parser.add_argument('--exp_name', type=str, default="colorization", help="Name of the experiment", choices=["zero_level","baseline", "colorization", "all"])

    # Dataset parameters
    parser.add_argument('--input_path', type=str, default="dataset\\dataset_mini", required=False, help='Dataset input path')
    parser.add_argument('--slices', type=int, default=3, required=False,
                        help='How many more slice to add to the dataset besides the index one')
    parser.add_argument('--preprocess', type=str, default="12bit", required=False, help="preprocess type")

    # Training parameters
    parser.add_argument('--folds', type=int, default=4, required=False, help='Number of k-folds for CV')
    parser.add_argument('--epochs', type=int, default=15, required=False, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=12, required=False, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, required=False, help='Learning rate')
    parser.add_argument('--l2_reg', type=float, default=0.0001, required=False, help='L2 regularization alpha value')
    parser.add_argument('--optim', type=str, default='sgd', required=False, help='Optimizer')                             #not in use
    parser.add_argument('--momentum', type=float, default=0.9, required=False, help='Momentum value')                     #not in use
    parser.add_argument('--loss_function', type=str, default='binary_crossentropy', required=False, help='Loss function') #not in use
    parser.add_argument('--secondary_weight', type=float, default=0.2, required=False,
                        help='Weight used to scale loss terms from secondary tasks')
    parser.add_argument('--class_weight', type=int, default=1, required=False,
                        help='Whether to use class weight during training (0=no, 1=yes)')
    parser.add_argument('--seed', type=int, default=42, required=False, help='Seed for reproducible results')
    parser.add_argument('--early_stop', type=int, default=0, required=False,
                        help='Whether to use an early stopping policy (0=no, 1=yes)')  #not in use
    parser.add_argument('--checkpoint', type=str, default="", required=False, help="Checkpoint path")
    parser.add_argument('--epoch_for_ckpt', type=int, default=15, required=False, help="Epoch number of the checkpoint to be evaluated")

    # Architecture parameters
    parser.add_argument('--architecture', type=str, default="multibranch", help="Type of the architecture, mono or multibranch", choices=["multibranch", "monobranch"])
    parser.add_argument('--branches', type=str, action='append', required=False,
                help='Choose which branches of the architecture to use: DWI, T2, DCE_peak, DCE_3TP')                     #not in use
    parser.add_argument('--colorization_option', type=str, default="multicolor", help="Type of the colorizer architecture, one for each channel or a unique module.", choices=["multicolor", "unique"])
    parser.add_argument('--load_weights', type=int, default=0, required=False,
                        help='Whether to load weights from trained model (to be placed in the folder "weights/") (0=no, 1=yes)') #not in use
    parser.add_argument('--backbone', type=str, default="ResNet50", required=False, help="Backbone for the feature extraction step")
    parser.add_argument('--dropout', type=float, default=0.5, required=False, help='Dropout rate')
    parser.add_argument('--fc', type=int, default=128, required=False, help='Size of last FC layer')

    # Extra parameters
    parser.add_argument('--lime_top', type=int, default=4, required=False, help='Number of relevant lime superpixels')
    parser.add_argument('--lime_pert', type=int, default=100, required=False, help='Number of lime perturbations')
    parser.add_argument('--wanb_project_name', type=str, default="NACtry", required=False, help='Name of wandb project name')
    
    args = parser.parse_args()
    return args

def get_model_setup(exp_name:str):
    return MODEL_SETUP[exp_name]["COLORIZE"], MODEL_SETUP[exp_name]["FREEZE_BACKBONE"]