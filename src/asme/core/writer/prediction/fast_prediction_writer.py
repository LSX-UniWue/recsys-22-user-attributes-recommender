import csv
import json
from abc import abstractmethod
from pathlib import Path
from typing import List, Union, Optional, IO, Tuple

class CSVPredictionWriter():

    """
    The CSVPredictionWriter writes predictions into a file encoded as CSV. Each sample is written to the file in the
    following way:

    For each sample the recommendations will be written in a separate line. The positions of the recommendation is added
    as a field in the CSV line to keep the information where in the recommendation list the recommendation item(s)
    was/were.
    """

    HEADER = ["SID", "RECOMMENDATION_POSITION", "RECOMMENDATION"]
    INPUT_HEADER_NAME = "INPUT"
    TARGET_HEADER_NAME = "TARGET"
    SCORES_HEADER_NAME = "PROBABILITY"

    def __init__(self,
                 file_handle: IO[str],
                 log_input: bool,
                 log_target: bool,
                 log_scores: bool,
                 log_per_sample_metrics: bool
                 ):
        self.file_handle = file_handle
        self.log_input = log_input
        self.log_target = log_target
        self.log_scores = log_scores
        self.log_per_sample_metrics = log_per_sample_metrics
        self.csv_writer = csv.writer(file_handle)
        self._header_written = False

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
