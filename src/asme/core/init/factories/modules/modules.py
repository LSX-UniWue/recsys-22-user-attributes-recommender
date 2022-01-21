import inspect
from dataclasses import dataclass
from typing import List, Any, Dict, Optional, Union, Callable

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.features.tokenizer_factory import get_tokenizer_key_for_voc
from asme.core.init.factories.metrics.metrics_container import MetricsContainerFactory
from asme.core.init.factories.util import require_config_keys
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.utils.inject import InjectTokenizer, InjectTokenizers, InjectVocabularySize, InjectModel

MODEL_PARAM_NAME = 'model'
METRICS_PARAM_NAME = 'metrics'
LOSS_FUNCTION_PARAM_NAME = 'loss_function'
TOKENIZER_SUFFIX = '_tokenizer'
VOCAB_SIZE_SUFFIX = '_vocab_size'
MODEL_SUFFIX = '_' + MODEL_PARAM_NAME


@dataclass
class ParameterInfo:
    """ class to store information about the parameter """
    parameter_type: type
    default_value: Optional[Any] = None


def _get_parameters(cls) -> Dict[str, ParameterInfo]:
    signature = inspect.signature(cls.__init__)

    parameter_infos = {}

    for parameter_name, info in signature.parameters.items():
        if parameter_name != 'self':
            parameter_infos[parameter_name] = ParameterInfo(info.annotation, info.default)

    return parameter_infos


def _filter_parameters(parameters: Dict[str, ParameterInfo],
                       filter_func: Callable[[], bool]
                       ) -> Dict[str, Optional[Any]]:
    return dict(filter(filter_func, parameters.items()))

def _get_tokenizers_from_context(context: Context) -> Dict[str, Tokenizer]:
    tokenizers = {}
    for key, item in context.as_dict().items():
        if isinstance(item, Tokenizer):
            tokenizers[key.replace(TOKENIZER_SUFFIX, '')] = item

    return tokenizers


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
        parameters = _get_parameters(self._module_csl)
        tokenizer_parameters = _filter_parameters(parameters,
                                                  lambda item: isinstance(item[1].parameter_type, InjectTokenizer))
        config_parameters = dict([x for x in parameters.items() if x[0] not in tokenizer_parameters])

        # Model is only present for modules that contain a model.
        if self.should_build_model:
            config_parameters.pop(MODEL_PARAM_NAME)

        config_parameters.pop(METRICS_PARAM_NAME)

        for parameter, parameter_info in config_parameters.items():
            default_value = parameter_info.default_value
            default_value = None if default_value == inspect._empty else default_value
            named_parameters[parameter] = config.get_or_default(parameter, default_value)

        # bind the tokenizers
        for tokenizer_parameter_name, tokenizer_parameter_info in tokenizer_parameters.items():
            inject_instance = tokenizer_parameter_info.parameter_type
            tokenizer_to_use = inject_instance.feature_name
            tokenizer = context.get(get_tokenizer_key_for_voc(tokenizer_to_use))

            if tokenizer is None:
                if tokenizer_parameter_info.default_value == inspect._empty:
                    raise KeyError(f'No tokenizer with id "{tokenizer_to_use}" configured and no default value set.')
                else:
                    tokenizer = tokenizer_parameter_info.default_value
            named_parameters[tokenizer_parameter_name] = tokenizer

        # if requested, bind all tokenizers to a single variable as a dict indexed by the feature keys
        tokenizers_parameters = _filter_parameters(parameters,
                                                   lambda item: isinstance(item[1].parameter_type, InjectTokenizers))
        if len(tokenizers_parameters) > 0:
            tokenizers = _get_tokenizers_from_context(context)
            for name, info in tokenizers_parameters.items():
                named_parameters[name] = tokenizers

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
        config_parameters = _get_config_required_config_params(_get_parameters(self._model_cls))

        return require_config_keys(config, config_parameters)

    def build(self,
              config: Config,
              context: Context
              ) -> Any:
        named_parameters = {}

        # collect the parameters from the config
        parameters = _get_parameters(self._model_cls)
        for parameter, parameter_info in parameters.items():
            default_value = parameter_info.default_value
            default_value = None if default_value == inspect._empty else default_value
            named_parameters[parameter] = config.get_or_default(parameter, default_value)

        vocab_vars = _filter_parameters(parameters,
                                        lambda dict_item: isinstance(dict_item[1].parameter_type, InjectVocabularySize))

        # Collect parameters that are a model themselves
        model_params = _filter_parameters(parameters, lambda param: isinstance(param[1].parameter_type, InjectModel))
        if len(model_params) > 0:
            # Build all parameter models recursively
            for model_param_name, model_param_info in model_params.items():
                inject_instance = model_param_info.parameter_type
                config_section_name = inject_instance.config_section_name \
                    if inject_instance.config_section_name is not None else model_param_name
                if not config.has_path([config_section_name]):
                    if model_param_info.default_value == inspect._empty:
                        raise KeyError(
                            f"Model '{self._model_cls.__name__}' specifies a sub-model '{model_param_name}' of type "
                            f"'{inject_instance.model_cls}' with no default value but no configuration section "
                            f"named '{config_section_name}' was found in the config.")
                else:
                    factory = GenericModelFactory(model_param_info.parameter_type)
                    # We assume that the config entry has the same name as the model parameter
                    model_config = config.get_config([config_section_name])
                    model = factory.build(model_config, context)
                    named_parameters[model_param_name] = model

        for name, info in vocab_vars.items():
            inject_instance = info.parameter_type
            tokenizer_to_use = inject_instance.feature_name
            tokenizer = context.get(get_tokenizer_key_for_voc(tokenizer_to_use))

            vocab_size = 0
            if tokenizer is not None:
                vocab_size = len(tokenizer)

            named_parameters[name] = vocab_size

        return self._model_cls(**named_parameters)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ['model']

    def config_key(self) -> str:
        return self._model_cls.__name__.lower()


def _get_config_required_config_params(parameters: Dict[str, Optional[Any]]) -> List[str]:
    result = []

    for parameter_name, parameter_info in parameters.items():
        default_value = parameter_info.default_value
        if default_value is inspect._empty and not parameter_name.endswith(VOCAB_SIZE_SUFFIX):
            result.append(parameter_name)

    return result
