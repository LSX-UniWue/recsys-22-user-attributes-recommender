import inspect
from dataclasses import dataclass
from typing import Optional, Any, List, Callable, Dict

from asme.core.init.context import Context
from asme.core.init.factories.modules import TOKENIZER_SUFFIX, VOCAB_SIZE_SUFFIX
from asme.core.tokenization.tokenizer import Tokenizer


@dataclass
class ParameterInfo:
    """ class to store information about the parameter """
    parameter_name: str
    parameter_type: type
    default_value: Optional[Any] = None
    inject_instance: Optional["Inject"] = None

    def is_injectable(self):
        return self.inject_instance is not None


def get_init_parameters(cls) -> List[ParameterInfo]:
    signature = inspect.signature(cls.__init__)

    parameter_infos = []

    for parameter_name, info in signature.parameters.items():
        if parameter_name != 'self':
            parameter_infos += [ParameterInfo(parameter_name, info.annotation, info.default)]

    return parameter_infos


def filter_parameters(parameters: List[ParameterInfo],
                      filter_func: Callable[[ParameterInfo], bool]
                      ) -> List[ParameterInfo]:
    return list(filter(filter_func, parameters))


def get_tokenizers_from_context(context: Context) -> Dict[str, Tokenizer]:
    tokenizers = {}
    for key, item in context.as_dict().items():
        if isinstance(item, Tokenizer):
            tokenizers[key.replace(TOKENIZER_SUFFIX, '')] = item

    return tokenizers


def get_config_required_config_params(parameters: List[ParameterInfo]) -> List[str]:
    result = []

    for parameter_info in parameters:
        default_value = parameter_info.default_value
        parameter_name = parameter_info.parameter_name
        if default_value is inspect._empty and not parameter_name.endswith(VOCAB_SIZE_SUFFIX):
            result.append(parameter_name)

    return result
