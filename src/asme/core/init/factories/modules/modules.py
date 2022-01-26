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
from asme.core.utils.inject import InjectTokenizer, InjectTokenizers, InjectVocabularySize, InjectModel, Inject

MODEL_PARAM_NAME = 'model'
METRICS_PARAM_NAME = 'metrics'
LOSS_FUNCTION_PARAM_NAME = 'loss_function'
TOKENIZER_SUFFIX = '_tokenizer'
VOCAB_SIZE_SUFFIX = '_vocab_size'
MODEL_SUFFIX = '_' + MODEL_PARAM_NAME


@dataclass
class ParameterInfo:
    """ class to store information about the parameter """
    parameter_name: str
    parameter_type: type
    default_value: Optional[Any] = None

    def get_inject_instance(self) -> Optional[Inject]:
        return self.parameter_type if self.is_injectable() else None

    def is_injectable(self):
        return isinstance(self.parameter_type, Inject)


def _get_parameters(cls) -> List[ParameterInfo]:
    signature = inspect.signature(cls.__init__)

    parameter_infos = []

    for parameter_name, info in signature.parameters.items():
        if parameter_name != 'self':
            parameter_infos += [ParameterInfo(parameter_name, info.annotation, info.default)]

    return parameter_infos


def _filter_parameters(parameters: List[ParameterInfo],
                       filter_func: Callable[[ParameterInfo], bool]
                       ) -> List[ParameterInfo]:
    return list(filter(filter_func, parameters))


def _get_tokenizers_from_context(context: Context) -> Dict[str, Tokenizer]:
    tokenizers = {}
    for key, item in context.as_dict().items():
        if isinstance(item, Tokenizer):
            tokenizers[key.replace(TOKENIZER_SUFFIX, '')] = item

    return tokenizers


def _handel_injects(injectable_parameters: List[ParameterInfo], context: Context, config: Config,
                    parameter_dict: Dict[str, Any]):
    """
    This function tries to inject a value into the parameter_dict for every parameter given by iunjectable parameters.

    :param injectable_parameters: A list of parameter info that are injectable, i.e. parameter infos
    p where p.is_injectable() == True
    :param context: The context the current factory is working in.
    :param config: The Config object used to build nested models from.
    :param parameter_dict: A dictionary of parameter names and values that should be populated by values gathered from
    the context or the config.
    """
    for injectable_parameter in injectable_parameters:
        inject = injectable_parameter.get_inject_instance()

        if isinstance(inject, InjectTokenizer):
            tokenizer_to_use = inject.feature_name
            tokenizer = context.get(get_tokenizer_key_for_voc(tokenizer_to_use))

            if tokenizer is None:
                if injectable_parameter.default_value == inspect._empty:
                    raise KeyError(f'No tokenizer with id "{tokenizer_to_use}" configured and no default value set.')
                else:
                    tokenizer = injectable_parameter.default_value
            parameter_dict[injectable_parameter.parameter_name] = tokenizer
        elif isinstance(inject, InjectTokenizers):
            tokenizers = _get_tokenizers_from_context(context)
            parameter_dict[injectable_parameter.parameter_name] = tokenizers
        elif isinstance(inject, InjectVocabularySize):
            tokenizer_to_use = inject.feature_name
            tokenizer = context.get(get_tokenizer_key_for_voc(tokenizer_to_use))
            if tokenizer is None:
                raise KeyError(f"No tokenizer with id {tokenizer_to_use} configured. "
                               f"Can not inject vocabulary size into parameter {injectable_parameter.parameter_name}.")
            else:
                parameter_dict[injectable_parameter.parameter_name] = len(tokenizer)
        elif isinstance(inject, InjectModel):
            config_section_name = inject.config_section_name \
                if inject.config_section_name is not None else injectable_parameter.parameter_name
            if not config.has_path([config_section_name]):
                if injectable_parameter.default_value == inspect._empty:
                    raise KeyError(
                        f"A sub-model '{injectable_parameter.parameter_name}' of type "
                        f"'{inject.model_cls}' with no default value was specified but no configuration section "
                        f"named '{config_section_name}' was found in the config.")
            else:
                factory = GenericModelFactory(inject.model_cls)
                # We assume that the config entry has the same name as the model parameter
                model_config = config.get_config([config_section_name])
                model = factory.build(model_config, context)
                parameter_dict[injectable_parameter.parameter_name] = model
        else:
            # We failed ot handle an inject case!
            raise NotImplementedError(f"Inject instances of type {type(inject)} are not handled yet!")


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
        config_parameters = _filter_parameters(parameters, lambda p: not p.is_injectable())

        # Model is only present for modules that contain a model.
        if self.should_build_model:
            config_parameters = _filter_parameters(config_parameters,
                                                   lambda p: not p.parameter_name == MODEL_PARAM_NAME)

        # We do not want to build the metrics container directly from the config.
        config_parameters = _filter_parameters(config_parameters, lambda p: not p.parameter_name == METRICS_PARAM_NAME)

        for parameter_info in config_parameters:
            name = parameter_info.parameter_name
            default_value = parameter_info.default_value
            default_value = None if default_value == inspect._empty else default_value
            named_parameters[name] = config.get_or_default(name, default_value)

        # build the metrics container
        metrics = self.metrics_container_factory.build(config.get_config(self.metrics_container_factory.config_path()),
                                                       context=context)

        # handle Inject directives
        injectable_parameters = _filter_parameters(parameters, lambda p: p.is_injectable())
        _handel_injects(injectable_parameters, context, config, named_parameters)

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
        for parameter_info in parameters:
            default_value = parameter_info.default_value
            parameter_name = parameter_info.parameter_name
            default_value = None if default_value == inspect._empty else default_value
            named_parameters[parameter_name] = config.get_or_default(parameter_name, default_value)

        # handle Inject directives
        injectable_parameters = _filter_parameters(parameters, lambda p: p.is_injectable())
        _handel_injects(injectable_parameters, context, config, named_parameters)

        return self._model_cls(**named_parameters)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ['model']

    def config_key(self) -> str:
        return self._model_cls.__name__.lower()


def _get_config_required_config_params(parameters: List[ParameterInfo]) -> List[str]:
    result = []

    for parameter_info in parameters:
        default_value = parameter_info.default_value
        parameter_name = parameter_info.parameter_name
        if default_value is inspect._empty and not parameter_name.endswith(VOCAB_SIZE_SUFFIX):
            result.append(parameter_name)

    return result
