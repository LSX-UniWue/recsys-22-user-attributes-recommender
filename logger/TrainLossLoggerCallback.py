from pytorch_lightning import Callback


# TODO: Right now, PL does not actually provide the outputs of the model per batch to the 'on_train_batch_end' callback.
# Hopefully, this will be implemented in PL 1.1.0 (refer to training_loop.py:568)
class TrainLossLoggerCallback(Callback):

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pass