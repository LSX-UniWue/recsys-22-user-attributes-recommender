import inspect

from typing import List, Any, Dict, Optional, Union, Callable

from init.config import Config
from init.context import Context
from init.factories.metrics.metrics_container import MetricsContainerFactory
from init.factories.tokenizer.tokenizer_factory import get_tokenizer_key_for_voc
from init.factories.util import check_config_keys_exist
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


MODEL_PARAM_NAME = 'model'
METRICS_PARAM_NAME = 'metrics'
TOKENIZER_SUFFIX = '_tokenizer'


class GenericModuleFactory(ObjectFactory):

    """
    this generic factory can build module instances, iff they are following the conventions:
    1. that the model parameter is named 'model'
    2. that the metrics parameter is named 'metrics'
    3. all tokenizers that are a parameter of the module are named x'_tokenizer'

    than the factory will automatically bind the x tokenizer to the 'tokenizers.'x configured tokenizer
    """

    def __init__(self, module_cls, model_cls):
        super().__init__()

        self._module_csl = module_cls

        self.model_factory = GenericModelFactory(model_cls)
        self.metrics_container_factory = MetricsContainerFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        metrics_can_build = self.metrics_container_factory.can_build(config.get_config(self.metrics_container_factory.config_path()), context)
        if metrics_can_build.type != CanBuildResultType.CAN_BUILD:
            return metrics_can_build
        return self.model_factory.can_build(config.get_config(self.model_factory.config_path()), context)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        # collect the parameters from the config
        named_parameters = {}

        # collect the parameters from the config
        parameters = _get_parameters(self._module_csl)
        tokenizer_parameters = _filter_parameters(parameters, lambda param_name: param_name[0].endswith(TOKENIZER_SUFFIX))
        config_parameters = dict([x for x in parameters.items() if x[0] not in tokenizer_parameters])
        config_parameters.pop(MODEL_PARAM_NAME)
        config_parameters.pop(METRICS_PARAM_NAME)

        for parameter, default_value in config_parameters.items():
            default_value = None if default_value == inspect._empty else default_value
            named_parameters[parameter] = config.get_or_default(parameter, default_value)

        # bind the tokenizers
        for tokenizer_parameter in tokenizer_parameters.keys():
            tokenizer_to_use = tokenizer_parameter.replace(TOKENIZER_SUFFIX, '')
            tokenizer = context.get(get_tokenizer_key_for_voc(tokenizer_to_use))
            named_parameters[tokenizer_parameter] = tokenizer

        # build the metrics container
        metrics = self.metrics_container_factory.build(config.get_config(self.metrics_container_factory.config_path()),
                                                       context=context)

        # build the model container
        model = self.model_factory.build(config.get_config(self.model_factory.config_path()), context)

        named_parameters[MODEL_PARAM_NAME] = model
        named_parameters[METRICS_PARAM_NAME] = metrics

        return self._module_csl(**named_parameters)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return self._module_csl.__name__.lower()


def _filter_parameters(parameters: Dict[str, Optional[Any]],
                       filter_func: Callable[[], bool]
                       ) -> Dict[str, Optional[Any]]:
    return dict(filter(filter_func, parameters.items()))


def _get_vocab_parameters(parameters: List[str]):
    return [parameter for parameter in parameters if parameter.endswith('vocab_size')]


def _get_config_required_config_params(parameters: Dict[str, Optional[Any]]) -> List[str]:
    result = []

    for parameter_name, default_value in parameters.items():
        if default_value is inspect._empty and not parameter_name.endswith('_vocab_size'):
            result.append(parameter_name)

    return result


def _get_parameters(cls) -> Dict[str, Optional[Any]]:
    signature = inspect.signature(cls.__init__)

    parameter_info = {}

    for parameter_name, info in signature.parameters.items():
        if parameter_name != 'self':
            parameter_info[parameter_name] = info.default

    return parameter_info


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
        config_parameters = _get_config_required_config_params(_get_parameters(self._model_cls))
        config_keys_exist = check_config_keys_exist(config, config_parameters)
        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION)

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> Any:
        named_parameters = {}

        # collect the parameters from the config
        parameters = _get_parameters(self._model_cls)
        for parameter, default_value in parameters.items():
            default_value = None if default_value == inspect._empty else default_value
            named_parameters[parameter] = config.get_or_default(parameter, default_value)

        for vocab_var in _get_vocab_parameters(list(parameters.keys())):
            tokenizer_to_use = vocab_var.replace('_vocab_size', '')
            tokenizer = context.get(get_tokenizer_key_for_voc(tokenizer_to_use))
            named_parameters[vocab_var] = len(tokenizer)

        return self._model_cls(**named_parameters)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ['model']

    def config_key(self) -> str:
        return self._model_cls.__name__.lower()
