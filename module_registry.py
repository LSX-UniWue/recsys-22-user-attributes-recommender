import pytorch_lightning as pl

from argparse import ArgumentParser
from typing import Type, Dict, Any

from configs.models.model_config import ModelConfig
from configs.training.training_config import TrainingConfig


class ModuleBuilder(object):
    """
    builds a module for the specified training and model config
    """

    def __init__(self,
                 module_cls: Type[pl.LightningModule],
                 training_config_cls: Type[TrainingConfig],
                 model_config_cls: Type[ModelConfig]
                 ):
        super().__init__()

        self.module_cls = module_cls
        self.training_config_cls = training_config_cls
        self.model_config_cls = model_config_cls

    def add_arguments_to_parser(self, parent_parser: ArgumentParser) -> ArgumentParser:
        # first some configs for the trainer
        parser = pl.Trainer.add_argparse_args(parent_parser)
        # parameters for the training
        parser = self.training_config_cls.add_model_specific_args(parser)
        # parameters for the model
        parser = self.model_config_cls.add_model_specific_args(parser)
        return parser

    def build_module(self, argument_parser_dict: Dict[str, Any]) -> pl.LightningModule:
        training_config = self.training_config_cls.from_args(argument_parser_dict)
        model_config = self.model_config_cls.from_args(argument_parser_dict)

        return self.module_cls(training_config, model_config)


def build_module_builder(module_cls: Type[pl.LightningModule],
                         training_config_cls: Type[TrainingConfig],
                         model_config_cls: Type[ModelConfig]
                         ) -> ModuleBuilder:
    return ModuleBuilder(module_cls, training_config_cls, model_config_cls)


class ModuleRegistry(object):

    mapping = {}

    @classmethod
    def register_module(cls,
                        module_id: str
                        ):

        def wrap(module_cls):
            # infer training and model config classes
            training_config_cls = None
            model_config_cls = None
            import inspect
            init_signature = inspect.signature(module_cls.__init__)
            for key, value in init_signature.parameters.items():
                if 'train' in key:
                    training_config_cls = value.annotation
                if 'model' in key:
                    model_config_cls = value.annotation

            if training_config_cls is None or model_config_cls is None:
                raise KeyError("TODO")
                # FIXME: add correct error and error message

            cls.mapping[module_id] = build_module_builder(module_cls, training_config_cls, model_config_cls)
            return module_cls

        return wrap

    @classmethod
    def get_module_builder(cls,
                           module_id: str):
        return cls.mapping[module_id]


module_registry = ModuleRegistry()
