from pathlib import Path
from typing import Dict, Any

from optuna import Trial

from init.templating.template_processor import TemplateProcessor


class SearchConfigurationTemplateProcessor(TemplateProcessor):

    def __init__(self, trial: Trial):
        super().__init__()
        self._trial = trial

    def can_modify(self, config: Dict[str, Any]) -> bool:
        return True

    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        study_name = f"study_{self._trial.study.study_name}"
        trial_number = self._trial.number
        sub_path = Path(f"{study_name}") / f"{trial_number}"

        trainer_config = config.get('trainer')
        checkpoint_config = trainer_config.get('checkpoint')
        checkpoint_path = Path(checkpoint_config['dirpath'])
        checkpoint_path_parent = checkpoint_path.parent
        checkpoint_config['dirpath'] = str(Path(checkpoint_path_parent) / sub_path / checkpoint_path.stem)

        logger_config = trainer_config.get('logger', None)
        if logger_config is not None and logger_config.get('type') == 'tensorboard':
            logger_config["version"] = str(trial_number)  # to remove the prefix version_
            logger_config["name"] = study_name

        return config
