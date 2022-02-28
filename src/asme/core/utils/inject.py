from dataclasses import dataclass, field
from typing import Optional, Any, Dict

from dataclasses_json import dataclass_json


# All injection anntotations should inherit from this base class to ensure extendability for future use cases.
class Inject:
    pass


class InjectTokenizer(Inject):
    def __init__(self, feature_name: str):
        self.feature_name = feature_name


class InjectTokenizers(Inject):
    pass


class InjectVocabularySize(Inject):
    def __init__(self, feature_name: str):
        self.feature_name = feature_name


class InjectModel(Inject):
    def __init__(self, model_cls, config_section_path: Optional[str] = None):
        self.model_cls = model_cls
        self.config_section_path = config_section_path


class InjectClass(Inject):
    def __init__(self, config_section_path: Optional[str] = None):
        """
        :param config_section_path: The path to the section which contains the data necessary to build the desired
            object.Nested obects in the config can be accessed by using ".".
        """
        self.config_section_path = config_section_path


class InjectList(Inject):
    def __init__(self, config_section_path: Optional[str] = None):
        """
        :param config_section_path: The path to the section which contains the data necessary to build the desired
            object.Nested obects in the config can be accessed by using ".".
        """
        self.config_section_path = config_section_path


class InjectDict(Inject):
    def __init__(self, config_section_path: Optional[str] = None):
        """
        :param config_section_path: The path to the section which contains the data necessary to build the desired
            object.Nested obects in the config can be accessed by using ".".
        """
        self.config_section_path = config_section_path


@dataclass_json
@dataclass
class InjectObjectConfig:
    cls_name: str
    module_name: Optional[str]
    parameters: Dict[str, Any] = field(default_factory=lambda: {})
