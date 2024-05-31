from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau

checkpoint_filepath="checkpoint"

def get_callbacks(checkpoint: bool = True, earlystop: bool = True, lr_monitor: bool = True) -> list:
    """
    Returns the list of the callbacks for the Lightning trainer.

    Args:
        checkpoint (bool): whether to use a checkpoint cb. Defaults to True.
        earlystop (bool): whether to use an early stopping cb. Defaults to True.
        lr_monitor (bool): whether to use a lr monitoring cb. Defaults to True.
    Returns:
        list[lightningtorch.Callbacks]: the list with the chosen callbacks.
    """
    cb_list = []
    
    if checkpoint:
        checkpoint_cb = ModelCheckpoint(
            filename = checkpoint_filepath,
            monitor='val_auroc',
            mode='max',
            save_weights_only=True,
            save_top_k=1,
            verbose=True
        )
        cb_list.append(checkpoint_cb)

    if earlystop:
        earlystop_cb = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            min_delta=0.000025,
            verbose=True
        )
        cb_list.append(earlystop_cb)

    if lr_monitor:
        learningrate_cb = LearningRateMonitor(
            logging_interval="epoch"
        )
        cb_list.append(learningrate_cb)

    print(cb_list)
    return cb_list

def get_LR_scheduler(optimizer):
    """
    This method provides a lr scheduler for the model training.
    Args:
        None
    Returns:
        lr_scheduler (torch.LRscheduler)
    """
    lr_scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode = "min",
        factor=0.5,
        patience=5,
        threshold=0.000025,
        verbose=True
    )
    return lr_scheduler