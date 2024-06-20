import torch
import lightning.pytorch as pl
import torchmetrics
import torchvision

#from pytorch_lightning import loggers as pl_loggers
import logging
import wandb

from architectures import MultiParametricMRIModel
from utils.computation_utils import compute_loss, get_patient_level, test_predict
from utils.callbacks_utils import get_LR_scheduler

class LightningModel(pl.LightningModule):

    def __init__(self, num_slices = 3, fc_dimension = 128, dropout = 0.5, exp_name="default", backbone="ResNet50", optim = "sgd", lr =0.0001, wd=0.001, secondary_weight=0.2, class_weights=None, folder_time='', fold_num=1):
        super().__init__()
        self.exp_name = exp_name
        self.backbone = backbone
        self.num_slices = num_slices
        self.optimizer = optim
        self.lr= lr
        self.wd = wd
        self.secondary_weight = secondary_weight
        self.folder_time = folder_time
        self.fold_num = fold_num

        if exp_name == "colorization":
            self.model = None
        elif exp_name == "baseline":
            self.model = MultiParametricMRIModel(fc_dimension, dropout, self.backbone, freeze_backbone=True)
        else: #baseline all
            self.model = MultiParametricMRIModel(fc_dimension, dropout, self.backbone, freeze_backbone=False)

        if class_weights is not None: 
            self.weights = class_weights.to(self.device)
        else:
            self.weights = None

        self.loss_fn = compute_loss

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
        probs = self.model(images)
        return probs
    
    def configure_optimizers(self):
        #if self.optimizer == "sgd": #TODO da capire se ci sono più optimizer come fare
        optimizer =  torch.optim.SGD(
            self.parameters(),
            lr = self.lr,
            weight_decay=self.wd,
            momentum=0.9
        )
        scheduler = get_LR_scheduler(optimizer) #lo puoi passare solo se c'è un moniotrer del lr
        
        optimizer_dict = {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                    } 
                }
        return optimizer_dict
    
    def loss_function(self, probs, labels, weights):
        loss = self.loss_fn(probs, labels, self.secondary_weight, type="cross_entropy", weights=weights)
        return loss
    
    def _common_step(self, batch, batch_idx, train=False):
        if train and self.weights is not None:
            weights = self.weights.to(self.device)
        else:
            weights = None
        
        images, labels = batch
        images = images.permute(dims=(1,0, *range(2, images.dim()))).to(self.device)
        labels = labels.squeeze(dim=1).to(self.device)
        colorized_x, probs = self.model(images)
        if not train:
            print(probs)
        loss = self.loss_function(probs, labels, weights)
        return loss, probs, labels

    # Stepwise computations
    def training_step(self, batch, batch_idx):
        x,_ = batch
        loss, probs, labels = self._common_step(batch, batch_idx, train=True)
        accuracy = self.binary_accuracy(probs["pCR"].argmax(axis=-1), labels.argmax(axis=-1)) #TODO aggiunto 29 05 alle 17 33
        auroc = self.auroc(probs["pCR"][:,1], labels[:,1])
        self.log_dict({"train_loss": loss, "train_acc":accuracy, "train_auroc":auroc}, 
                 on_step=False, on_epoch=True, prog_bar=True, logger=True) 
        output = {'loss':loss, 'probs': probs, 'labels': labels}
        self.training_step_outputs.append(output)
        
        #Let's perform some visualization
        if batch_idx == 1:
            x = x[:4][0]
            grid = torchvision.utils.make_grid(x.view(-1,3,224,224))
            self.logger.experiment.log({"MRIs": [wandb.Image(grid, caption = "MRIs")]})

        return output
    
    def validation_step(self, batch, batch_idx):
        loss, probs, labels = self._common_step(batch, batch_idx)
        accuracy = self.binary_accuracy(probs["pCR"].argmax(axis=-1), labels.argmax(axis=-1)) #TODO aggiunto 29 05 alle 17 33
        auroc = self.auroc(probs["pCR"][:,1], labels[:,1])
        self.log_dict({"val_loss": loss, "val_acc":accuracy, "val_auroc":auroc}, 
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)

        output = {'loss':loss, 'probs': probs, 'labels': labels}
        self.validation_step_outputs.append(output)
        return output
    
    def test_step(self, batch, batch_idx):
        loss, probs, labels = self._common_step(batch, batch_idx)
        accuracy = self.binary_accuracy(probs["pCR"].argmax(axis=-1), labels.argmax(axis=-1)) #TODO aggiunto 29 05 alle 17 33
        auroc = self.auroc(probs["pCR"][:,1], labels[:,1])
        self.log_dict({"test_loss": loss, "test_acc":accuracy, "test_auroc":auroc}, 
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)

        output = {'loss':loss, 'probs': probs, 'labels': labels}
        self.test_step_outputs.append(output)
        return output

    # Hooks for on_epoch = True
    def on_train_epoch_end(self):
        #retrieve all the outputs from the epoch steps
        # outputs = self.training_step_outputs
        
        # total_labels = torch.Tensor().to(self.device)
        # total_probs = { }
        # for step_dict in outputs:
        #     total_labels = torch.concat((total_labels, step_dict['labels']))
        #     probs_dict = step_dict["probs"]
        #     for k,v in probs_dict.items():
        #         if k not in total_probs:
        #             total_probs[k] = torch.Tensor().to(self.device)
        #         total_probs[k] = torch.concat((total_probs[k],probs_dict[k]))
        #ora devo mettere le total_probs da qualche parte, se no cosa se ne fa?
        
        #clear the outputs from all the steps
        self.training_step_outputs = []

    def on_validation_epoch_end(self) -> None:
        # outputs = self.validation_step_outputs
        
        # total_labels = torch.Tensor().to(self.device)
        # total_probs = { }
        # for step_dict in outputs:
        #     total_labels = torch.concat((total_labels, step_dict['labels']))
        #     probs_dict = step_dict["probs"]
        #     for k,v in probs_dict.items():
        #         if k not in total_probs:
        #             total_probs[k] = torch.Tensor().to(self.device)
        #         total_probs[k] = torch.concat((total_probs[k],probs_dict[k]))
        
        # #clean the outputs
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
        auroc = self.auroc(total_probs["pCR"], total_labels)
        print(f"AUC calcolata con torchmetrics: {auroc}")

        # Calcolo delle metriche sulla singola fold
        for output_name, Y_prob in total_probs.items():
            print(output_name, " : ", Y_prob)
            roc_result_slice = test_predict(total_labels, Y_prob, "slice", output_name, self.folder_time, self.fold_num, self.num_slices)
            #print(roc_result_slice)
            self.slice_dict_test[output_name] = roc_result_slice
            Y_val_split, Y_prob_split = get_patient_level(total_labels, Y_prob, self.num_slices)
            roc_result_patient = test_predict(Y_val_split, Y_prob_split, "patient", output_name, self.folder_time, self.fold_num, self.num_slices)
            self.patient_dict_test[output_name] = roc_result_patient
        
        print("Dizionario con le probabilità a livello slice: ")
        print(self.slice_dict_test)
        self.test_step_outputs = []
