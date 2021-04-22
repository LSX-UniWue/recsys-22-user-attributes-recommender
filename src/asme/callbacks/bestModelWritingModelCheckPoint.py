import os
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import LightningModule


class BestModelWritingModelCheckpoint(ModelCheckpoint):
    """
        Decorates the ModelCheckpoint class by calling the to_yaml method after each epoch and creating a symlink pointing to the best model.
    """

    def __init__(self, model_checkpoint: ModelCheckpoint, output_file: str):
        # AD: You should store the model_checkpoint object as a property of the decorating class
        #     and then delegate all calls to methods of ModelCheckpoint there, e.g.

        # self.target_object = model_checkpoint
        # then override all methods and just delegate to the underlying object or add functionality -> decorating the method call
        ModelCheckpoint.__init__(self, model_checkpoint.dirpath, model_checkpoint.filename, model_checkpoint.monitor, model_checkpoint.verbose,
                                 model_checkpoint.save_last, model_checkpoint.save_top_k, model_checkpoint.save_weights_only, model_checkpoint.mode,
                                 model_checkpoint.period, model_checkpoint.prefix)
        if output_file is not None:
            self.output_file = output_file
        else:
            self.output_file = self.dirpath

    def on_train_end(self, trainer, pl_module: LightningModule) -> None:
        ModelCheckpoint.to_yaml(self, os.path.join(self.output_file, "best_k_models.yaml"))
        os.symlink(self.best_model_path, os.path.join(self.output_file, "best.ckpt"))

    # Example for delegating:
    #def on_pretrain_routine_start(self, trainer, pl_module):
    #    self.target_object.on_pretrain_routine_start(trainer, pl_module)
    
    # Example for adding functionality
    #def on_train_end(self, trainer, pl_module: LightningModule) -> None:
        # first let the checkpoint model instance handle the call
    #    self.target_object.on_train_end()

        # then add your functionality
    #    self.target_object.to_yaml(self, os.path.join(self.output_file, "best_k_models.yaml"))
    #    os.symlink(self.best_model_path, os.path.join(self.output_file, "best.ckpt"))
