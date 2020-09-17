from typing import Union, Dict, List

import torch
from pytorch_lightning import EvalResult


class AggregatingMetricTrait:
    def on_step_end(self, prediction: torch.Tensor, target: torch.Tensor, result: EvalResult) -> None:
        raise NotImplementedError()

    def on_epoch_end(self, outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]], result: EvalResult) -> None:
        raise NotImplementedError()