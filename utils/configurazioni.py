import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Deep Learning on NAC Data')

    # Training parameters

    # Architecture parameters

    # Visualization parameters

    # Path parameters

    parser.add_argument('--branches', type=str, action='append', required=False,
                    help='Choose which branches of the architecture to use: DWI, T2, DCE_peak, DCE_3TP')
    parser.add_argument('--slices', type=int, default=3, required=False,
                        help='How many more slice to add to the dataset besides the index one')
    parser.add_argument('--load_weights', type=int, default=0, required=False,
                        help='Whether to load weights from trained model (to be placed in the folder "weights/") (0=no, 1=yes)')
    parser.add_argument('--secondary_weight', type=float, default=0.2, required=False,
                        help='Weight used to scale loss terms from secondary tasks')
    parser.add_argument('--class_weight', type=int, default=1, required=False,
                        help='Whether to use class weight during training (0=no, 1=yes)')
    parser.add_argument('--folds', type=int, default=2, required=False, help='Number of k-folds for CV')  #4
    parser.add_argument('--early_stop', type=int, default=0, required=False,
                        help='Whether to use an early stopping policy (0=no, 1=yes)')
    parser.add_argument('--epochs', type=int, default=1, required=False, help='Number of epochs') #15
    parser.add_argument('--batch', type=int, default=12, required=False, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, required=False, help='Learning rate')
    parser.add_argument('--optim', type=str, default='sgd', required=False, help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, required=False, help='Momentum value')
    parser.add_argument('--dropout', type=float, default=0.5, required=False, help='Dropout rate')
    parser.add_argument('--l2_reg', type=float, default=0.0001, required=False, help='L2 regularization alpha value')
    parser.add_argument('--fc', type=int, default=128, required=False, help='Size of last FC layer')
    parser.add_argument('--seed', type=int, default=42, required=False, help='Seed for reproducible results')
    parser.add_argument('--loss_function', type=str, default='binary_crossentropy', required=False, help='Loss function')
    parser.add_argument('--lime_top', type=int, default=4, required=False, help='Number of relevant lime superpixels')
    parser.add_argument('--lime_pert', type=int, default=100, required=False, help='Number of lime perturbations')
    parser.add_argument('--input_path', type=str, default="dataset\\dataset_mini", required=False, help='Dataset input path')
    parser.add_argument('--neptune_token', type=str, default="", help="NEPTUNE API TOKEN")
    parser.add_argument('--exp_name', type=str, default="", help="Folder name")
    parser.add_argument('--checkpoint', type=str, default="", required=False, help="Checkpoint path")
    args = parser.parse_args()
    return args

# class Configurator():

#     def __init__(self):
#         self.name = "Deep learning experiment configurator"
#         pass

#     def get_parser_args(self):
#         parser = argparse.ArgumentParser(description='Deep Learning on NAC Data')
#         parser.add_argument('--branches', type=str, action='append', required=False,
#                         help='Choose which branches of the architecture to use: DWI, T2, DCE_peak, DCE_3TP')
#         parser.add_argument('--slices', type=int, default=3, required=False,
#                             help='How many more slice to add to the dataset besides the index one')
#         parser.add_argument('--load_weights', type=int, default=0, required=False,
#                             help='Whether to load weights from trained model (to be placed in the folder "weights/") (0=no, 1=yes)')
#         parser.add_argument('--secondary_weight', type=float, default=0.2, required=False,
#                             help='Weight used to scale loss terms from secondary tasks')
#         parser.add_argument('--class_weight', type=int, default=1, required=False,
#                             help='Whether to use class weight during training (0=no, 1=yes)')
#         parser.add_argument('--folds', type=int, default=4, required=False, help='Number of k-folds for CV')
#         parser.add_argument('--early_stop', type=int, default=0, required=False,
#                             help='Whether to use an early stopping policy (0=no, 1=yes)')
#         parser.add_argument('--epochs', type=int, default=15, required=False, help='Number of epochs')
#         parser.add_argument('--batch', type=int, default=12, required=False, help='Batch size')
#         parser.add_argument('--learning_rate', type=float, default=0.0001, required=False, help='Learning rate')
#         parser.add_argument('--momentum', type=float, default=0.9, required=False, help='Momentum value')
#         parser.add_argument('--dropout', type=float, default=0.5, required=False, help='Dropout rate')
#         parser.add_argument('--l2_reg', type=float, default=0.0001, required=False, help='L2 regularization alpha value')
#         parser.add_argument('--fc', type=int, default=128, required=False, help='Size of last FC layer')
#         parser.add_argument('--seed', type=int, default=42, required=False, help='Seed for reproducible results')
#         parser.add_argument('--loss_function', type=str, default='binary_crossentropy', required=False, help='Loss function')
#         parser.add_argument('--lime_top', type=int, default=4, required=False, help='Number of relevant lime superpixels')
#         parser.add_argument('--lime_pert', type=int, default=100, required=False, help='Number of lime perturbations')
#         parser.add_argument('--input_path', type=str, default='NAC_Input', required=False, help='Dataset input path')
#         parser.add_argument('--neptune_token', type=str, default="", help="NEPTUNE API TOKEN")
#         parser.add_argument('--name', type=str, default="", help="Folder name")
#         self.args = parser.parse_args()

#     def set_env_vars_from_parser(self):
#         """
#         Process the args set through the parser and set them as environment variables.
#         """
#         os.environ["args"] = str(self.args)
#         os.environ["SEED"] = self.args.seed
#         os.environ["FC"] = self.args.fc
#         os.environ["SLICES"] = self.args.slices
#         os.environ["branches_list"] = self.args.branches_list
#         os.environ["BATCH_SIZE"] = self.args.batch_size
#         os.environ["LR"] = self.args.lr
#         os.environ["max_epochs"] = self.args.max_epochs
#         os.environ["momentum"] = self.args.momentum
#         os.environ["weight_decay"] = self.args.weight_decay
#         os.environ["secondary_weights"] = self.args.secondary_weights
#         os.environ["fold_num"] = self.args.fold_num
#         os.environ["name"] = self.args.name

#     def set_experiment_configuration(self):
#         """
#         Set the experiment configuration, as environment variables.
#         """
#         self.get_parser_args()
#         self.set_env_vars_from_parser()
#         print("Configuration set through environment variables.")

# CONFIGS = {
#     "SLICES": 3,
#     "branches_list": ["DWI","T2","DCE_peak","DCE_3TP"],
#     "accepted_branches" : ["DWI", "T2", "DCE_peak", "DCE_3TP"],
#     "scans" : ["DWI", "T2", "DCE"],
#     "SEED": 42,
#     "BATCH_SIZE": 3,
#     "FC": 128,
#     "LR": 0.0001,
#     "max_epochs": 1,
#     "momentum": 0.9,
#     "weight_decay": 0.0001,
#     "secondary_weights":0.2,
#     "fold_num":1,
#     "name": "experiment_"
# }

# def set_env_vars():
#     os.environ["SEED"] = CONFIGS["SEED"]
#     os.environ["FC"] = CONFIGS["FC"]
#     os.environ["SLICES"] = CONFIGS["SLICES"]
#     os.environ["branches_list"] = CONFIGS["branches_list"]
#     os.environ["BATCH_SIZE"] = CONFIGS["BATCH_SIZE"]
#     os.environ["LR"] = CONFIGS["LR"]
#     os.environ["max_epochs"] = CONFIGS["max_epochs"]
#     os.environ["momentum"] = CONFIGS["momentum"]
#     os.environ["weight_decay"] = CONFIGS["weight_decay"]
#     os.environ["secondary_weights"] = CONFIGS["secondary_weights"]
#     os.environ["fold_num"] = CONFIGS["fold_num"]
#     os.environ["name"] = CONFIGS["name"]

# def set_env_vars_from_parser(args):
#     os.environ["SEED"] = args.seed
#     os.environ["FC"] = args.fc
#     os.environ["SLICES"] = args.slices
#     os.environ["branches_list"] = args.branches_list
#     os.environ["BATCH_SIZE"] = args.batch_size
#     os.environ["LR"] = args.lr
#     os.environ["max_epochs"] = args.max_epochs
#     os.environ["momentum"] = args.momentum
#     os.environ["weight_decay"] = args.weight_decay
#     os.environ["secondary_weights"] = args.secondary_weights
#     os.environ["fold_num"] = args.fold_num
#     os.environ["name"] = args.name