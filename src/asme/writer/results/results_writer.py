import csv
import json
from abc import abstractmethod
from pathlib import Path
from typing import IO, Dict


class ResultWriter:
    """
    Writer interface for all writers that serialize predictions
    """

    def __init__(self,
                 file_handle: IO[str]
                 ):
        """
        constructor for prediction writers
        :param file_handle: the handle for the file
        """
        super().__init__()
        self.file_handle = file_handle

    @abstractmethod
    def write_overall_results(self, recommender_name: str, metrics: Dict[str, float]):
        """
        writes the metrics calculated over all data points to the file
        :param recommender_name:
        :param metrics:
        :return:
        """
        pass


class JSONResultWriter(ResultWriter):

    """
    The JSONResultWriter writes the results (metrics) into a file encoded as JSON.

    """

    def __init__(self,
                 file_handle: IO[str]):
        super().__init__(file_handle)

    def write_overall_results(self, recommender_name: str, metrics: Dict[str, float]):
        json_object = {
            'recommender_id': recommender_name,
            'metrics': metrics
        }

        json.dump(json_object, self.file_handle)


class CSVResultWriter(ResultWriter):

    """
    The CSVPredictionWriter writes results into a file encoded as CSV.

    Each metric is written as "metric_name, metric_value"

    the first line of the csv contains the recommender id
    """

    HEADER = ["metric name", "value"]

    def __init__(self,
                 file_handle: IO[str]
                 ):
        super().__init__(file_handle)

        self.csv_writer = csv.writer(file_handle)
        self.csv_writer.writerow(self.HEADER)

    def write_overall_results(self, recommender_name: str, metrics: Dict[str, float]):
        metrics_to_write = [[metric_name, metric_value] for metric_name, metric_value in metrics.items()]
        rows_to_write = metrics_to_write + [["recommender_id", recommender_name]]
        self.csv_writer.writerows(rows_to_write)


def build_result_writer(file_handle: IO[str]
                        ) -> ResultWriter:
    """
    Builds the result writer for the specified file handle based on the extension of the underlying file of the
    file_handle

    Currently the following file extensions are supported:

    - json
    - csv

    :param file_handle: the file handle to use
    :return:
    """
    file_extension = Path(file_handle.name).suffix

    # for lazy loading only point to the class
    supported_prediction_writers = {
        '.csv': CSVResultWriter,
        '.json': JSONResultWriter
    }

    if file_extension not in supported_prediction_writers:
        raise KeyError(f'{file_extension} is not a supported format to write predictions to file')
    return supported_prediction_writers[file_extension](file_handle)
