from dataclasses import dataclass

from configs.training.training_config import AdamOptimizerConfig, TrainingConfig, OptimizerConfig


@dataclass
class GRUTrainingConfig(OptimizerConfig, AdamOptimizerConfig, TrainingConfig):
    pass