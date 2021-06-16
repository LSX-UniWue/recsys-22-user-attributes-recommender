from abc import abstractmethod
from typing import Any, List, Union, Dict

from torch.utils.data import DataLoader

from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.init.object_factory import ObjectFactory, CanBuildResult
from asme.init.templating.datasources.datasources import DatasetSplit
from data.datasets.processors.processor import Processor


class DataSourcesFactory(ObjectFactory):

    def __init__(self):
        super().__init__()
        self._factory = ConditionalFactory("type", {
            "mask": MaskDataSourcesFactory("idk")
        }, "data_sources", ["data_sources"])

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self._factory.can_build(config, context)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        return self._factory.build(config, context)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ["data_sources"]

    def config_key(self) -> str:
        return "data_sources"


class BaseDataSourcesFactory(ObjectFactory):

    def __init__(self, key: str):
        super().__init__()
        self._key = key

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        pass

    def build(self, config: Config, context: Context) -> Dict[str, DataLoader]:
        pass

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self._key]

    def config_key(self) -> str:
        return self._key


class MaskDataSourcesFactory(BaseDataSourcesFactory):
    pass


"""
How do we get from the config of a specific datasource to the final dataloaders?
Current system: Expand the datasource config via a DatasetBuilder during preprocessing such that the factories
                understand it. The actual construction can then be deferred to the factories.
                
What we want:   We do not want to have templating. However, we have to modify the config of the datasource since factories
                do not understand it otherwise. Essentially, we still have to do templating if we want to keep using the 
                factories. Hence, we would need to also ditch the current factories for the dataloaders to get rid of 
                templating?

Bottom Line:    Either we keep templating in some form or we ditch the current dataloader factories. (I think)
"""