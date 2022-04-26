from abc import abstractmethod
from typing import Dict, Any, List

import torch
from asme.core.evaluation.pred_utils import _extract_sample_metrics, get_positive_item_mask
from asme.core.metrics.container.metrics_container import MetricsContainer
from asme.core.metrics.metric import MetricStorageMode
from asme.core.tokenization.utils.tokenization import remove_special_tokens
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, SAMPLE_IDS, TARGET_ENTRY_NAME, SESSION_IDENTIFIER
from torch import Tensor


class BatchEvaluator:
    """
    Evaluator for batch level evaluation
    """

    @abstractmethod
    def evaluate(self,
                 batch_index: int,
                 batch: Dict[str, Any],
                 logits: Tensor,
                 ) -> List[Any]:
        """
        Execute evaluation on batch and processed logits.

        :param batch_index:
        :param batch:
        :param logits:

        :return:
        """
        pass

    @abstractmethod
    def get_header(self) -> List[str]:
        pass

    @abstractmethod
    def eval_samplewise(self) -> bool:
        pass


class LogInputEvaluator(BatchEvaluator):
    """
    Extract the original input sequence
    """

    def __init__(self, item_tokenizer):
        self.item_tokenizer = item_tokenizer
        self.header = ["input"]

    def get_header(self) -> List[str]:
        return self.header

    def eval_samplewise(self) -> bool:
        return True

    def evaluate(self,
                 batch_index: int,
                 batch: Dict[str, Any],
                 logits: Tensor,
                 ) -> List[Any]:
        input_sequences = []
        for batch_sample in range(logits.shape[0]):
            sequence = batch[ITEM_SEQ_ENTRY_NAME][batch_sample].tolist()
            # remove padding tokens
            sequence = remove_special_tokens(sequence, self.item_tokenizer)
            sequence = self.item_tokenizer.convert_ids_to_tokens(sequence)
            input_sequences.append(sequence)
        return input_sequences


class ExtractSampleIdEvaluator(BatchEvaluator):
    """
    Create Sample ID based on SESSION_IDENTIFIER or sample ID
    """

    def __init__(self, use_session_id: bool = False):
        self.header = ["SID"]
        self.use_session_id = use_session_id

    def get_header(self) -> List[str]:
        return self.header

    def eval_samplewise(self) -> bool:
        return True

    def evaluate(self,
                 batch_index: int,
                 batch: Dict[str, Any],
                 logits: Tensor
                 ) -> List[Any]:

        if self.use_session_id:
            sample_ids = batch[SESSION_IDENTIFIER]
        else:
            sample_ids = batch[SAMPLE_IDS].tolist()
        sequence_position_ids = None
        if 'pos' in batch:
            sequence_position_ids = batch['pos'].tolist()
        id_lambda = lambda x: self._generate_sample_id(sample_ids, sequence_position_ids, x)
        sample_ids = [id_lambda(batch_sample) for batch_sample in range(logits.shape[0])]

        return sample_ids

    def _generate_sample_id(self, sample_ids, sequence_position_ids, sample_index) -> str:
        sample_id = sample_ids[sample_index]
        if sequence_position_ids is None:
            return sample_id
        return f'{sample_id}_{sequence_position_ids[sample_index].item()}'


class TrueTargetEvaluator(BatchEvaluator):
    """
    Extract the true target
    """

    def __init__(self, item_tokenizer):
        self.header = ["target"]
        self.item_tokenizer = item_tokenizer

    def get_header(self) -> List[str]:
        return self.header

    def eval_samplewise(self) -> bool:
        return True

    def evaluate(self,
                 batch_index: int,
                 batch: Dict[str, Any],
                 logits: Tensor,
                 ) -> List[Any]:

        targets = batch[TARGET_ENTRY_NAME]
        sequences = batch[ITEM_SEQ_ENTRY_NAME]
        is_basket_recommendation = len(sequences.size()) == 3

        target_converted = []
        for batch_sample in range(logits.shape[0]):
            true_target = targets[batch_sample]
            if is_basket_recommendation:
                true_target = remove_special_tokens(true_target.tolist(), self.item_tokenizer)
            else:
                true_target = [true_target.item()]

            true_target = self.item_tokenizer.convert_ids_to_tokens(true_target)
            target_converted.append(true_target)
        return target_converted


class ExtractScoresEvaluator(BatchEvaluator):
    """
    Extract the scores
    """

    def __init__(self, item_tokenizer, filter, num_predictions: int):
        self.header = ["score"]
        self.item_tokenizer = item_tokenizer
        self.num_predictions = num_predictions
        self.filter = filter

    def get_header(self) -> List[str]:
        return self.header

    def eval_samplewise(self) -> bool:
        return False

    def evaluate(self,
                 batch_index: int,
                 batch: Dict[str, Any],
                 logits: Tensor,
                 ) -> List[Any]:
        prediction = self.filter(logits)
        softmax = torch.softmax(prediction, dim=-1)
        scores, indices = torch.sort(softmax, dim=-1, descending=True)
        scores = scores[:, :self.num_predictions]
        scores = scores.cpu().numpy().tolist()

        return scores


class ExtractRecommendationEvaluator(BatchEvaluator):
    """
    Extract the recommendation, always needed
    """

    def __init__(self, item_tokenizer, filter, num_predictions: int, selected_items=None):
        self.header = ["recommendation"]
        self.item_tokenizer = item_tokenizer
        self.num_predictions = num_predictions
        self.filter = filter
        self.selected_items = selected_items

    def get_header(self) -> List[str]:
        return self.header

    def eval_samplewise(self) -> bool:
        return False

    def evaluate(self,
                 batch_index: int,
                 batch: Dict[str, Any],
                 logits: Tensor,
                 ) -> List[Any]:

        prediction = self.filter(logits)
        softmax = torch.softmax(prediction, dim=-1)
        scores, indices = torch.sort(softmax, dim=-1, descending=True)
        indices = indices[:, :self.num_predictions]
        indices = indices.cpu().numpy().tolist()

        items = []
        for batch_sample in range(logits.shape[0]):
            item_ids = indices[batch_sample]
            if self.selected_items is not None:
                selected_item_ids = [self.selected_items[i] for i in item_ids]
                item_ids = selected_item_ids
            items.append(self.item_tokenizer.convert_ids_to_tokens(item_ids))

        return items


class PerSampleMetricsEvaluator(BatchEvaluator):
    """
    Calculates the metrics per sample
    """

    def __init__(self, item_tokenizer, filter, module):
        self.item_tokenizer = item_tokenizer
        self.filter = filter
        self.module = module

        metrics_container: MetricsContainer = module.metrics
        for metric in metrics_container.get_metrics():
            metric.set_metrics_storage_mode(MetricStorageMode.PER_SAMPLE)
        self.header = metrics_container.get_metric_names()

    def get_header(self) -> List[str]:
        return self.header

    def eval_samplewise(self) -> bool:
        return True

    def evaluate(self,
                 batch_index: int,
                 batch: Dict[str, Any],
                 logits: Tensor,
                 ) -> List[Any]:

        targets = batch[TARGET_ENTRY_NAME]
        metrics = _extract_sample_metrics(self.module)

        num_classes = logits.size()[1]
        item_mask = get_positive_item_mask(targets, num_classes)
        for name, metric in metrics:
            metric.update(logits, item_mask)

        metric_results = []
        for batch_sample in range(logits.shape[0]):
            metric_values = [value.raw_metric_values()[batch_index].cpu().tolist()[batch_sample] for name, value in
                             metrics]

            metric_results.append(metric_values)

        return metric_results
