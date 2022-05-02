import copy
import inspect
from typing import List, Any, Dict, Union

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.metrics.metrics_container import MetricsContainerFactory
from asme.core.init.factories.modules import MODEL_PARAM_NAME, METRICS_PARAM_NAME, LOSS_FUNCTION_PARAM_NAME
from asme.core.init.factories.modules.util import get_init_parameters, filter_parameters, \
    get_config_required_config_params
from asme.core.init.factories.util import require_config_keys
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class GenericModuleFactory(ObjectFactory):
    """

    this generic factory can build module instances, if the model follows the following conventions:
    1. that the model parameter is named 'model' (if the module does not contain a model, this can be ignored)
    2. that the metrics parameter is named 'metrics'
    3. all tokenizers that are parameters of the module are named x'_tokenizer'
    than the factory will automatically bind the x tokenizer to the 'tokenizers.'x configured tokenizer

    """

    def __init__(self,
                 module_cls,
                 loss_function=None,
                 model_cls=None
                 ):
        super().__init__()

        self._module_csl = module_cls
        # This indicates whether the module we want to build contains a model.
        self.should_build_model = model_cls is not None

        if self.should_build_model:
            self.model_factory = GenericModelFactory(model_cls)
        self.metrics_container_factory = MetricsContainerFactory()
        self.loss_function = loss_function

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        metrics_can_build = self.metrics_container_factory.can_build(
            config.get_config(self.metrics_container_factory.config_path()), context)
        if metrics_can_build.type != CanBuildResultType.CAN_BUILD:
            return metrics_can_build
        # If the module does not contain a model, we short circuit here and don't query the model factory.
        return not self.should_build_model or \
               self.model_factory.can_build(config.get_config(self.model_factory.config_path()), context)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        # collect the parameters from the config
        named_parameters = {}

        # collect the parameters from the config
        parameters = get_init_parameters(self._module_csl)

        # Model is only present for modules that contain a model.
        if self.should_build_model:
            parameters = filter_parameters(parameters,
                                           lambda p: not p.parameter_name == MODEL_PARAM_NAME)

        # We do not want to build the metrics container directly from the config.
        parameters = filter_parameters(parameters, lambda p: not p.parameter_name == METRICS_PARAM_NAME)

        for parameter_info in parameters:
            name = parameter_info.parameter_name
            default_value = parameter_info.default_value
            default_value = None if default_value == inspect._empty else default_value
            named_parameters[name] = copy.deepcopy(config.get_or_default(name, default_value))

        # build the metrics container
        metrics = self.metrics_container_factory.build(config.get_config(self.metrics_container_factory.config_path()),
                                                       context=context)

        # build the model container if a model class was supplied
        if self.should_build_model:
            model = self.model_factory.build(config.get_config(self.model_factory.config_path()), context)
            named_parameters[MODEL_PARAM_NAME] = model

        named_parameters[METRICS_PARAM_NAME] = metrics

        if self.loss_function is not None:
            named_parameters[LOSS_FUNCTION_PARAM_NAME] = self.loss_function

        # create a deep copy to avoid potential config modifications made by the module to leak into asme
        return self._module_csl(**named_parameters)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return self._module_csl.__name__.lower()


class GenericModelFactory(ObjectFactory):
    """
    a generic model factory
    """

    def __init__(self, model_cls):
        super().__init__()
        self._model_cls = model_cls

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        config_parameters = get_config_required_config_params(get_init_parameters(self._model_cls))

        return require_config_keys(config, config_parameters)

    def build(self,
              config: Config,
              context: Context
              ) -> Any:
        named_parameters = {}

        # collect the parameters from the config
        parameters = get_init_parameters(self._model_cls)
        for parameter_info in parameters:
            default_value = parameter_info.default_value
            parameter_name = parameter_info.parameter_name
            default_value = None if default_value == inspect._empty else default_value
            named_parameters[parameter_name] = copy.deepcopy(config.get_or_default(parameter_name, default_value))

        return self._model_cls(**named_parameters)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ['model']

    def config_key(self) -> str:
        return self._model_cls.__name__.lower()


