from dataclasses import dataclass, field

from configs.base_config import BaseConfig
from configs.training.training_config import TrainingConfig, OptimizerConfig


@dataclass
class BERT4RecTrainingConfig(OptimizerConfig, TrainingConfig):

    mask_probability: float = field(
        default=0.15,
        metadata={
            BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'the probability to mask a step in the sequence'
        }
    )
