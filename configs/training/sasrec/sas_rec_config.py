from dataclasses import dataclass

from configs.training.training_config import TrainingConfig, AdamOptimizerConfig, OptimizerConfig


@dataclass
class SASRecTrainingConfig(OptimizerConfig, AdamOptimizerConfig, TrainingConfig):
    beta_2 = 0.98
