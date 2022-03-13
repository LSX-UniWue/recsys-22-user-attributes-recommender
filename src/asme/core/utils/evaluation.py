from abc import abstractmethod
from typing import Dict, Any, List, Tuple

from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, SAMPLE_IDS, TARGET_ENTRY_NAME, SESSION_IDENTIFIER

class BatchEvaluator():

    @abstractmethod
    def evaluate(self,
                batch_index: int,
                batch: Dict[str, Any]
                ) -> Dict[str, Any]:
        pass


class SampleEvaluator():
    """
    Evaluator for sample wise processing
    """

    @abstractmethod
    def evaluate(self,
                 batch_index: int,
                 sample_index: int,
                 batch: Dict[str, Any]
                 ) -> Dict[str, Any]:
        pass


#class DefaultBatchEvaluator(BatchEvaluator):

class SampleIDEvaluator(SampleEvaluator):

    def __init__(self, name: str = "SID", use_session_id: bool = False):
        self.name = name
        self.use_session_id = use_session_id


    def evaluate(self,
                 batch_index : int,
                 sample_index: int,
                 batch: Dict[str, Any]
                 ) -> Tuple[str, Any]:

        if self.use_session_id:
            sample_id = batch[SESSION_IDENTIFIER][sample_index]
        else:
            sample_ids = batch[SAMPLE_IDS]
            sequence_position_ids = None
            if 'pos' in batch:
                sequence_position_ids = batch['pos']
            sample_id = self._generate_sample_id(sample_ids, sequence_position_ids, sample_index)

        return self.name, sample_id

    def _generate_sample_id(self,sample_ids, sequence_position_ids, sample_index) -> str:
        sample_id = sample_ids[sample_index].item()
        if sequence_position_ids is None:
            return sample_id
        return f'{sample_id}_{sequence_position_ids[sample_index].item()}'