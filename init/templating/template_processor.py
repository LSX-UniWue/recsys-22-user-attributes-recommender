from abc import abstractmethod
from typing import Dict, Any


class TemplateProcessor:

    @abstractmethod
    def can_modify(self,
                   config: Dict[str, Any]
                   ) -> bool:
        """
        :param config:
        :return: true iff the template processor can modifiy the config dict
        """
        pass

    @abstractmethod
    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        make all the necessary config modifcations
        :param config:
        :return: the modified config
        """
        pass
