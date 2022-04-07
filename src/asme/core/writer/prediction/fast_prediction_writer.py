import csv
from typing import List, Optional, IO, Tuple
import itertools

from asme.core.evaluation.evaluation import BatchEvaluator

def all_equal_in_list(lst):
    return not lst or lst.count(lst[0]) == len(lst)

class EvaluationCSVWriter:

    """
    The EvaluationWriter writes predictions into a file encoded as CSV.

    Each sample is written to the file in the
    following way:

    For each sample the recommendations will be written in a separate line. The positions of the recommendation is added
    as a field in the CSV line to keep the information where in the recommendation list the recommendation item(s)
    was/were.
    """

    def __init__(self,
                 evaluators: List[BatchEvaluator],
                 file_handle: IO[str]):

        self.file_handle = file_handle
        self.csv_writer = csv.writer(file_handle)
        self.evaluators = evaluators

        self.headers = itertools.chain.from_iterable([evaluator.get_header() for evaluator in evaluators])
        self.csv_writer.writerow(self.headers)


    def write_evaluation(self,
                         batch_index,
                         batch,
                         logits):

       # eval_results = [(e.evaluate(batch_index, batch, logits), e.eval_samplewise()) for e in self.evaluators]

        #Per Sample in batch
        for sample in range(logits.shape[0]):
            results = [(evaluator.evaluate(batch_index, batch, logits), evaluator.eval_samplewise(), evaluator.get_header()) for evaluator in self.evaluators]

            num_rows = [len(eval[0]) for eval, samplewise, header in results if not samplewise]
            if not all_equal_in_list(num_rows):
                raise ValueError(f"Evaluators output size must be of same length, but is {num_rows}.")

            rows_to_write = []
            for i in range(num_rows[0]):
                row_to_write = []
                for eval, samplewise, header in results:
                    res = eval[sample] if samplewise else eval[sample][i]
                    if len(header) > 1:
                        row_to_write.extend(res)
                    else:
                        row_to_write.append(res)

                rows_to_write.append(row_to_write)
            self.csv_writer.writerows(rows_to_write)











    def write_values(self, sample_id: str,
                         recommendations: List[str],
                         scores: List[float],
                         targets: List[str],
                         metrics: Optional[List[Tuple[str, float]]],
                         input_sequence: Optional[List[str]]):

        def _extract_metric_names(metrics: List[Tuple[str,float]]) -> List[str]:
            return list(map(lambda metric: metric[0], metrics))

        def _extract_metric_values(metrics: List[Tuple[str,float]]) -> List[float]:
            return list(map(lambda metric: metric[1], metrics))

        if not self._header_written:
            header_to_write = self.HEADER
            if self.log_scores:
                header_to_write += [self.SCORES_HEADER_NAME]
            if self.log_target:
                header_to_write += [self.TARGET_HEADER_NAME]
            if self.log_input:
                header_to_write += [self.INPUT_HEADER_NAME]
            if self.log_per_sample_metrics:
                header_to_write += _extract_metric_names(metrics)
            self.csv_writer.writerow(header_to_write)
            self._header_written = True

        rows_to_write = []
        if self.log_per_sample_metrics:
            metric_row = _extract_metric_values(metrics)
        # loop through the recommendations and scores and build the rows to write
        for recommendation_position in range(0, len(recommendations)):
            row_to_write = [sample_id, recommendation_position + 1, recommendations[recommendation_position]]
            if self.log_scores:
                row_to_write.append(scores[recommendation_position])
            if self.log_target:
                row_to_write.append(targets)
            if self.log_per_sample_metrics:
                row_to_write.append(*metric_row)
            if self.log_input:
                row_to_write.append(input_sequence)
            rows_to_write.append(row_to_write)

        self.csv_writer.writerows(rows_to_write)


