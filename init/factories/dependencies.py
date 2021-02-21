from typing import Dict

from init.config import Config
from init.context import Context
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType

# TODO (AD) rewrite dependencyTrait in a way that it automatically processes all dependencies where they key can be found
# in the configuration
class DependenciesTrait:
    """
    A trait that contains operations that can help manage dependent factories. The trait assumes that dependent
    factories are responsible for building a specific subsection of the configuration and that the subection name
    matches the `config_path()` reported by the dependendent factory.
    """
    def __init__(self):
        super(DependenciesTrait, self).__init__()
        self.dependencies = {}

    def get_dependencies(self) -> Dict[str, ObjectFactory]:
        return self.dependencies

    def add_dependency(self, dependency: ObjectFactory):
        key = dependency.config_key()
        if key in self.dependencies:
            raise Exception(f"Dependency <{key}> is already registered.")
        self.dependencies[key] = dependency

    def can_build_dependencies(self, config: Config, context: Context) -> CanBuildResult:
        for key, factory in self.dependencies.items():
            if not config.has_path(factory.config_path()) and factory.is_required(context):
                return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION, f"missing path <{factory.config_path}>")

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def get_dependency(self, key: str) -> ObjectFactory:
        return self.dependencies[key]
