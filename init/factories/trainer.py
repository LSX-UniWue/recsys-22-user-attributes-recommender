from typing import Union, Any, Dict, List, Type

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger, WandbLogger

from init.config import Config
from init.context import Context
from init.factories.common.dependencies_factory import DependenciesFactory
from init.factories.common.union_factory import UnionFactory
from init.factories.util import require_config_keys
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from init.trainer_builder import TrainerBuilder


class KwargsFactory(ObjectFactory):

    def __init__(self, t: Type, key: str, selector_key: str = None, selector_value: str = None):
        super(KwargsFactory, self).__init__()
        self.t = t
        self.key = key
        self.selector_key = selector_key
        self.selector_value = selector_value

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        if self.selector_key and config.get(self.selector_key) != self.selector_value:
            return CanBuildResult(CanBuildResultType.INVALID_CONFIGURATION, f"Can't build for type {config.get(self.selector_key)}")

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        config_keys = config.get_keys()
        if self.selector_key:
            kwargs = {key: config.get(key) for key in config_keys if key != self.selector_key}
        else:
            kwargs = {key: config.get(key) for key in config_keys}

        return self.t(**kwargs)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.key]

    def config_key(self) -> str:
        return self.key


class TensorboardLoggerFactory(ObjectFactory):

    KEY = "logger"
    REQUIRED_KEYS = ["save_dir"]

    def __init__(self):
        super(TensorboardLoggerFactory, self).__init__()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        result = require_config_keys(config, ["save_dir"])
        if result.type != CanBuildResultType.CAN_BUILD:
            return result

        if config.get("type") != "tensorboard":
            return CanBuildResult(CanBuildResultType.INVALID_CONFIGURATION, f"Can't build for type {config.get('type')}")

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        config_keys = config.get_keys()
        kwargs = {key: config.get(key) for key in config_keys if key != "type"}

        if "version" not in kwargs:
            kwargs["version"] = ""

        return TensorBoardLogger(**kwargs)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY


class MLFlowLoggerFactory(KwargsFactory):
    def __init__(self):
        super().__init__(t=MLFlowLogger, key="logger", selector_key="type", selector_value="mlflow")


class WandBLoggerFactory(KwargsFactory):
    def __init__(self):
        super().__init__(t=WandbLogger, key="logger", selector_key="type", selector_value="wandb")


class CheckpointFactory(KwargsFactory):

    def __init__(self):
        super().__init__(t=ModelCheckpoint, key="checkpoint")


class EarlyStoppingCallbackFactory(KwargsFactory):

    def __init__(self):
        super().__init__(t=EarlyStopping, key='early_stopping')


class TrainerBuilderFactory(ObjectFactory):

    KEY = "trainer"

    def __init__(self):
        super().__init__()

        self.dependencies = DependenciesFactory([
            UnionFactory([TensorboardLoggerFactory(),
                          MLFlowLoggerFactory(),
                          WandBLoggerFactory()], "logger", ["logger"]),
            CheckpointFactory(),
            EarlyStoppingCallbackFactory()
        ], optional_based_on_path=True)

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> TrainerBuilder:
        config_keys = config.get_keys()
        dependency_keys = self.dependencies.get_dependency_keys()

        dependencies = self.dependencies.build(config, context)

        trainer_params_names = [x for x in config_keys if x not in dependency_keys]
        trainer_params = {key: config.get(key) for key in trainer_params_names}

        trainer_builder = TrainerBuilder(trainer_parameters=trainer_params)
        if "logger" in dependencies:
            trainer_builder.add_logger(dependencies["logger"])
        trainer_builder.add_callback(dependencies["checkpoint"])

        # add optional early stopping
        early_stopping_callback = dependencies.get('early_stopping', None)
        if early_stopping_callback is not None:
            trainer_builder.add_callback(early_stopping_callback)

        return trainer_builder

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
