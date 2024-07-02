import torch
import lightning.pytorch as pl
import torchmetrics
import torchvision
import os
import torch.nn as nn

#from pytorch_lightning import loggers as pl_loggers
import logging
import wandb

# from architectures_fase2 import NACColorizedMultimodel, MultiParametricMRIModel
from architectures_monobranch import NACColorizedMONOmodel
from architectures_multibranch import NACColorizedMultimodel
from utils.computation_utils import compute_loss,compute_loss_MONO, get_patient_level, test_predict
from utils.callbacks_utils import get_LR_scheduler

class NACLitModel(pl.LightningModule):

    def __init__(self, num_slices = 3, fc_dimension = 128, dropout = 0.5, architecture="multibranch", exp_name="evaluation", colorize=True, colorization_option="", freeze_backbone=False, backbone="ResNet50", optim = "sgd", lr =0.0001, wd=0.001, class_weights=None, folder_time='', fold_num=1, preprocess=""):
        super().__init__()
        self.exp_name = exp_name
        self.backbone = backbone
        self.num_slices = num_slices
        self.optimizer = optim
        self.lr= lr
        self.wd = wd
        self.folder_time = folder_time
        self.fold_num = fold_num

        if architecture == "monobranch":
            self.model = NACColorizedMONOmodel(fc_dimension, dropout, self.backbone, colorize = colorize, freeze_backbone=freeze_backbone)
            self.loss_fn = compute_loss_MONO
        else:
            self.model = NACColorizedMultimodel(fc_dimension, dropout, self.backbone, colorize = colorize, colorization_option=colorization_option, freeze_backbone=freeze_backbone)
            self.loss_fn = compute_loss

        if class_weights is not None: 
            self.weights = class_weights.to(self.device)
        else:
            self.weights = None


        # torchmetrics metrics - auroc may differ from the one computed with sklearn
        self.roc = torchmetrics.ROC(task="binary")
        self.auroc = torchmetrics.AUROC(task="binary")
        self.binary_accuracy = torchmetrics.Accuracy(task="binary")

        self.validation_labels_all = torch.Tensor().to(self.device)
        self.test_labels_all = torch.Tensor().to(self.device)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.slice_dict = {}
        self.patient_dict = {}

        self.slice_dict_test = {}
        self.patient_dict_test = {}

    def forward(self, images):
        colorized_x, probs = self.model(images)
        return colorized_x, probs
    
    def configure_optimizers(self):
        optimizer =  torch.optim.SGD(
            self.parameters(),
            lr = self.lr,
            weight_decay=self.wd,
            momentum=0.9
        )
        scheduler = get_LR_scheduler(optimizer) 
        
        optimizer_dict = {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                    } 
                }
        return optimizer_dict
    
    def loss_function(self, probs, labels, weights):
        loss = self.loss_fn(probs, labels, type="cross_entropy", weights=weights)
        return loss
    
    def _common_step(self, batch, batch_idx, train=False):
        if train and self.weights is not None:
            weights = self.weights.to(self.device)
        else:
            weights = None
        
        images, labels = batch
        images = images.permute(dims=(1,0,*range(2, images.dim()))).to(self.device) #batch and modalities channels must be switched
        labels = labels.squeeze(dim=1).to(self.device)
        colorized_x, logits = self.model(images)
        if not train:
            print(logits)
        
        if batch_idx == 1:
            colorized_x = colorized_x.permute(dims=(1,0,2,4,3))   #swapping H and W so that depicted images are "vertical"
            #print(f"Per il batch numero 1, x_colorized ha shape: {colorized_x.shape}")
            x = colorized_x[0]
            grid = torchvision.utils.make_grid(x.view(-1,3,224,224))
            self.logger.experiment.log({"Colorized Scans": [wandb.Image(grid, caption = "Colorized Scans")]})

            images = images.permute(dims=(1,0,2,4,3))   #swapping H and W so that depicted images are "vertical"
            print(f"Training step: le immagini caricate hanno shape {images.shape}")
            images = images[0]
            grid = torchvision.utils.make_grid(images.view(-1,3,224,224))
            self.logger.experiment.log({"Original": [wandb.Image(grid, caption = "Original MRIs")]})
        
        loss = self.loss_function(logits, labels, weights)
        return loss, logits, labels

    # Stepwise computations
    def training_step(self, batch, batch_idx):
        #x,_ = batch
        loss, logits, labels = self._common_step(batch, batch_idx, train=True)
        accuracy = self.binary_accuracy(logits["pCR"].argmax(axis=-1), labels.argmax(axis=-1))
        auroc = self.auroc(nn.Softmax(dim=1)(logits["pCR"])[:,1], labels[:,1])
        self.log_dict({"train_loss": loss, "train_acc":accuracy, "train_auroc":auroc}, 
                 on_step=False, on_epoch=True, prog_bar=True, logger=True) 
        
        output = {'loss':loss, 'probs': logits, 'labels': labels}
        self.training_step_outputs.append(output)
        return output
    
    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        accuracy = self.binary_accuracy(logits["pCR"].argmax(axis=-1), labels.argmax(axis=-1)) 
        auroc = self.auroc(nn.Softmax(dim=1)(logits["pCR"])[:,1], labels[:,1])
        self.log_dict({"val_loss": loss, "val_acc":accuracy, "val_auroc":auroc}, 
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)

        output = {'loss':loss, 'probs': logits, 'labels': labels}
        self.validation_step_outputs.append(output)
        return output
    
    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        accuracy = self.binary_accuracy(logits["pCR"].argmax(axis=-1), labels.argmax(axis=-1)) 
        auroc = self.auroc(nn.Softmax(dim=1)(logits["pCR"])[:,1], labels[:,1])
        self.log_dict({"test_loss": loss, "test_acc":accuracy, "test_auroc":auroc}, 
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)

        output = {'loss':loss, 'probs': logits, 'labels': labels}
        self.test_step_outputs.append(output)
        return output

    # Hooks for on_epoch = True
    def on_train_epoch_end(self):
        self.training_step_outputs = []  #clean the outputs list

    def on_validation_epoch_end(self) -> None:
        # #clean the outputs --- only if batch size for validation == len(val_dataset), otherwise metrics computation occurs here
        self.validation_step_outputs = []

    def on_test_end(self):
        print("\n\nON TEST END")
        outputs = self.test_step_outputs
        print(outputs)
        total_labels = torch.Tensor().to(self.device)
        total_probs = { }
        for step_dict in outputs:
            total_labels = torch.concat((total_labels, step_dict['labels']))
            probs_dict = step_dict["probs"]
            for k,v in probs_dict.items():
                if k not in total_probs:
                    total_probs[k] = torch.Tensor().to(self.device)
                total_probs[k] = torch.concat((total_probs[k],probs_dict[k]))
        
        #Calcolo delle metriche con torchvision
        auroc = self.auroc(nn.Softmax(dim=1)(total_probs["pCR"])[:,1], total_labels[:,1])
        #print(f"AUC calcolata con torchmetrics: {auroc}")

        # Calcolo delle metriche sulla singola fold
        for output_name, Y_prob in total_probs.items():
            #print(output_name, " : ", Y_prob)
            roc_result_slice = test_predict(total_labels, Y_prob, "slice", output_name, self.folder_time, self.fold_num, self.num_slices)
            self.slice_dict_test[output_name] = roc_result_slice
            Y_val_split, Y_prob_split = get_patient_level(total_labels, Y_prob, self.num_slices)
            roc_result_patient = test_predict(Y_val_split, Y_prob_split, "patient", output_name, self.folder_time, self.fold_num, self.num_slices)
            self.patient_dict_test[output_name] = roc_result_patient
        
        print("Dizionario con le probabilità a livello slice: ")
        print(self.slice_dict_test)
        self.test_step_outputs = []
