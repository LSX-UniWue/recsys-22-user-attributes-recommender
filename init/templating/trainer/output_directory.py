from typing import Dict, Any

from init.templating.config_utils import config_entry_exists, get_config_value, set_config_value
from init.templating.template_processor import TemplateProcessor


class OutputDirectoryProcessor(TemplateProcessor):
    """
    Sets all output directories and files based on a common output directory.

    The processor is enabled if the following section exists in the configuration:
    ```
    {
       ...
        output_directory: "a custom path"
       ...
    }
    ```
    Modifies:
    1) the trainer.checkpoint section and sets `dirpath` to `{output_directory}/checkpoints` iff trainer.checkpoint is
       defined and no dirpath is set.

    2) the trainer.logger section and sets `save_dir` to {output_directory} if the logger is of type mlflow or
       tensorboard.
       If no logger is specified a tensorboard logger will be created with `save_dir` set to {output_directory}.
    """
    def can_modify(self, config: Dict[str, Any]) -> bool:
        """
        Checks if the processor can be run. The following conditions must be met:
        1) `output_directory` must be present in the config file
        2) the user did not specify either `trainer.logger.save_dir` or `trainer.checkpoint.dirpath`.

        :param config: the configuration.

        :return: True if the processor can modify the configuration, False otherwise.
        """
        template_present = "output_directory" in config

        # check if user specified output directories on her own
        if config_entry_exists(config, ["trainer", "logger", "save_dir"]):
            raise KeyError('trainer.logger.save_dir is set to acustom value. Can not apply template.')

        if config_entry_exists(config, ["trainer", "checkpoint", "dirpath"]):
            raise KeyError('trainer.checkpoint.dirpath is set to a custom value. Can not apply template.')

        return template_present

    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        output_base_dir = config["output_directory"]

        self._modify_logger(config, output_base_dir)
        self._modify_checkpoint(config, output_base_dir)

        return config

    def _modify_logger(self, config: Dict[str, Any], output_base_dir: str):
        logger_element_exists = config_entry_exists(config, ["trainer", "logger"])
        # TODO refactor logic logger-type wise
        if logger_element_exists:
            logger_type = get_config_value(config, ["trainer", "logger", "type"])

            if logger_type is None:
                print(f"No logger type specified. Configuring a tensorboard logger...")
                set_config_value(config, ["trainer", "logger", "type"], "tensorboard")
                logger_type = "tensorboard"

            if logger_type == "tensorboard" or "mlflow":
                set_config_value(config, ["trainer", "logger", "save_dir"], output_base_dir)
            else:
                print(f"Unknown logger {logger_type} I did not set the output directory!!")

            # for tensorboard set version & name to empty string ("") to avoid subdirectories
            if logger_type == "tensorboard":
                logger_name_path = ["trainer", "logger", "name"]
                if not config_entry_exists(config, logger_name_path):
                    set_config_value(config, logger_name_path, "")

                version_name_path = ["trainer", "logger", "version"]
                if not config_entry_exists(config, version_name_path):
                    set_config_value(config, version_name_path, "")
        else:
            # configure a default tensorboard logger
            set_config_value(config,
                             ["trainer", "logger"],
                             {"type": "tensorboard", "save_dir": output_base_dir, "name": "", "version": ""})

    def _modify_checkpoint(self, config: Dict[str, Any], output_base_dir: str):
        output_checkpoint_dir = f"{output_base_dir}/checkpoints"

        checkpoint_element_exists = config_entry_exists(config, ["trainer", "checkpoint"])

        if checkpoint_element_exists:
            set_config_value(config, ["trainer", "checkpoint", "dirpath"], output_checkpoint_dir)
