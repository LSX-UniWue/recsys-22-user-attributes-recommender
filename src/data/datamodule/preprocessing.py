import copy
import math
import random
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Any, Optional, Dict

import pandas as pd

from asme.init.context import Context
from data.base.csv_index_builder import CsvSessionIndexer
from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datamodule.converters import CsvConverter
from data.datamodule.extractors import TargetPositionExtractor, FixedOffsetPositionExtractor
from data.datasets.index_builder import SequencePositionIndexBuilder
from data.datasets.sequence import ItemSessionParser, ItemSequenceDataset, PlainSequenceDataset
from data.utils.csv import read_csv_header, create_indexed_header
from datasets.data_structures.split_names import SplitNames
from datasets.data_structures.train_validation_test_splits_indices import TrainValidationTestSplitIndices
from datasets.vocabulary.create_vocabulary import create_token_vocabulary

EXTRACTED_DIRECTORY_KEY = "raw_file"
MAIN_FILE_KEY = "main_file"
SESSION_INDEX_KEY = "session_index"
OUTPUT_DIR_KEY = "output_dir"
PREFIXES_KEY = "prefixes"
DELIMITER_KEY = "delimiter"
SEED_KEY = "seed"


def format_prefix(prefixes: List[str]) -> str:
    return ".".join(prefixes)


class PreprocessingAction:

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def apply(self, context: Context) -> None:
        pass

    def __call__(self, context: Context) -> None:
        self.apply(context)


class ConvertToCsv(PreprocessingAction):

    def __init__(self, converter: CsvConverter):
        self.converter = converter

    def name(self) -> str:
        return "Converting to CSV"

    def apply(self, context: Context) -> None:
        extracted_dir = context.get(EXTRACTED_DIRECTORY_KEY)
        output_directory = context.get(OUTPUT_DIR_KEY)
        prefix = format_prefix(context.get(PREFIXES_KEY))
        output_file = output_directory / f"{prefix}.csv"
        self.converter(extracted_dir, output_file)
        context.set(MAIN_FILE_KEY, output_file)


class TransformCsv(PreprocessingAction):

    def __init__(self, filter: Callable[[pd.DataFrame], pd.DataFrame]):
        self.filter = filter

    def name(self) -> str:
        return "Filtering CSV"

    def apply(self, context: Context) -> None:
        current_file = context.get(MAIN_FILE_KEY)
        delimiter = context.get(DELIMITER_KEY)
        df = pd.read_csv(current_file, delimiter=delimiter)
        filtered = self.filter(df)
        filtered.to_csv(current_file, sep=delimiter)


@dataclass
class GroupedFilter:
    """
    Objects of this class hold all information necessary to aggregate a grouped dataframe and the filter based on
    aggregated values.

    :param aggregator: An aggregation function provided by pandas either as a string or a function reference, e.g "count".
    :param apply: The actual filter function which determines whether a row should by kept based on the aggregated values.
    :param aggregated_column: The name of the column to which the filter will be applied.
    """
    aggregator: Any
    apply: Callable[[Any], bool]
    aggregated_column: Optional[str] = None


class GroupAndFilter(PreprocessingAction):

    def __init__(self, group_by: str, filter: GroupedFilter):
        def apply_filter(d):
            if filter.aggregated_column is None:
                agg_column = d.columns[0]
            else:
                agg_column = filter.aggregated_column
            agg_value = d[agg_column].aggregate(filter.aggregator)
            return filter.apply(agg_value)

        def filter_fn(df: pd.DataFrame) -> pd.DataFrame:
            aggregated = df.groupby(group_by).filter(apply_filter)
            return aggregated

        self.transform = TransformCsv(filter_fn)

    def name(self) -> str:
        return "Filtering sessions"

    def apply(self, context: Context) -> None:
        self.transform.apply(context)


class CreateSessionIndex(PreprocessingAction):

    def __init__(self, session_key: List[str]):
        self.session_key = session_key

    def name(self) -> str:
        return "Creating session index"

    def apply(self, context: Context) -> None:
        main_file = context.get(MAIN_FILE_KEY)
        output_dir = context.get(OUTPUT_DIR_KEY)
        prefix = format_prefix(context.get(PREFIXES_KEY))
        session_index = output_dir / f"{prefix}.session.idx"
        delimiter = context.get(DELIMITER_KEY)
        csv_index = CsvSessionIndexer(delimiter=delimiter)
        csv_index.create(main_file, session_index, self.session_key)
        context.set(SESSION_INDEX_KEY, session_index, overwrite=True)


class CreateNextItemIndex(PreprocessingAction):

    def __init__(self, column: str, extractor: TargetPositionExtractor):
        self.extractor = extractor
        self.column = column

    def name(self) -> str:
        return "Creating next item index"

    def apply(self, context: Context) -> None:
        output_dir = context.get(OUTPUT_DIR_KEY)
        main_file = context.get(MAIN_FILE_KEY)
        session_index_path = context.get(SESSION_INDEX_KEY)
        delimiter = context.get(DELIMITER_KEY)
        prefix = format_prefix(context.get(PREFIXES_KEY))
        output_file = output_dir / f"{prefix}.nextitem.idx"

        session_index = CsvDatasetIndex(session_index_path)
        reader = CsvDatasetReader(main_file, session_index)
        parser = ItemSessionParser(
            create_indexed_header(read_csv_header(main_file, delimiter)),
            self.column,
            item_separator=delimiter,
            delimiter=delimiter
        )
        dataset = ItemSequenceDataset(PlainSequenceDataset(reader, parser))
        builder = SequencePositionIndexBuilder(target_positions_extractor=self.extractor)
        builder.build(dataset, output_file)


class CreateLeaveOneOutSplit(PreprocessingAction):

    def __init__(self, column: str, training_target_offset=2, validation_target_offset=1, test_target_offset=0):
        """
        This action allows to create a leave-one-out split for a dataset.

        :param column: The name of the column of the CSV file containing the items.
        :param training_target_offset:      The offset of the training target item, i.e. the distance from the end of
                                            the sequence. For instance, setting this to 2 will yield item n-2 as the
                                            target for a sequence of length n.
        :param validation_target_offset:    The offset of the validation target item, i.e. the distance from the end of
                                            the sequence. For instance, setting this to 1 will yield item n-1 as the
                                            target for a sequence of length n.
        :param test_target_offset:          The offset of the test target item, i.e. the distance from the end of the
                                            sequence. For instance, setting this to 0 will yield item n as the target
                                            for a sequence of length n.
        """
        self.column = column
        self.offsets = [training_target_offset, validation_target_offset, test_target_offset]

    def name(self) -> str:
        return "Creating Leave one out split"

    def apply(self, context: Context) -> None:
        output_dir = context.get(OUTPUT_DIR_KEY) / "loo"
        main_file = context.get(MAIN_FILE_KEY)
        session_index_path = context.get(SESSION_INDEX_KEY)
        delimiter = context.get(DELIMITER_KEY)

        output_dir.mkdir(parents=True, exist_ok=True)

        session_index = CsvDatasetIndex(session_index_path)
        reader = CsvDatasetReader(main_file, session_index)
        parser = ItemSessionParser(
            create_indexed_header(read_csv_header(main_file, delimiter)),
            self.column,
            item_separator=delimiter,
            delimiter=delimiter
        )
        dataset = ItemSequenceDataset(PlainSequenceDataset(reader, parser))
        for name, offset in zip([SplitNames.train, SplitNames.validation, SplitNames.test], self.offsets):
            prefix = format_prefix(context.get(PREFIXES_KEY) + [name.name])
            output_file = output_dir / f"{prefix}.loo.idx"
            builder = SequencePositionIndexBuilder(target_positions_extractor=FixedOffsetPositionExtractor(offset))
            builder.build(dataset, output_file)


class CreateVocabulary(PreprocessingAction):

    def __init__(self, columns: List[str],
                 custom_tokens: List[str] = None):
        self.columns = columns
        self.custom_tokens = [] if custom_tokens is None else custom_tokens

    def name(self) -> str:
        return f"Creating Vocabulary for columns: {self.columns}."

    def apply(self, context: Context) -> None:
        main_file = context.get(MAIN_FILE_KEY)
        session_index_path = context.get(SESSION_INDEX_KEY)
        output_dir = context.get(OUTPUT_DIR_KEY)
        prefix = format_prefix(context.get(PREFIXES_KEY))
        delimiter = context.get(DELIMITER_KEY)
        for column in self.columns:
            output_file = output_dir / f"{prefix}.vocabulary.{column}.txt"
            create_token_vocabulary(column, main_file, session_index_path,
                                    output_file, self.custom_tokens, delimiter, None)


class CreateRatioSplit(PreprocessingAction):

    def __init__(self, train_percentage: float, validation_percentage: float, test_percentage: float,
                 inner_actions: List[PreprocessingAction] = None):
        self.test_percentage = test_percentage
        self.validation_percentage = validation_percentage
        self.train_percentage = train_percentage
        self.inner_actions = [] if inner_actions is None else inner_actions

    def name(self) -> str:
        return f"Creating ratio split ({self.train_percentage}/{self.validation_percentage}/{self.test_percentage})"

    def apply(self, context: Context) -> None:
        main_file = context.get(MAIN_FILE_KEY)
        delimiter = context.get(DELIMITER_KEY)
        header = delimiter.join(read_csv_header(main_file, delimiter))
        output_dir = context.get(OUTPUT_DIR_KEY) / f"ratio_split-{self.train_percentage}_{self.validation_percentage}_{self.test_percentage}"
        session_index_path = context.get(SESSION_INDEX_KEY)
        session_index = CsvDatasetIndex(session_index_path)
        reader = CsvDatasetReader(main_file, session_index)

        # Create output dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Split the dataset
        split_indices = self._generate_split_indices(session_index, {
            SplitNames.train: self.train_percentage,
            SplitNames.validation: self.validation_percentage,
            SplitNames.test: self.test_percentage
        })

        # Write individual CSVs and execute inner actions for each split
        for name, indices in split_indices.items():
            self._process_split(context, name.name, header, reader, indices)

    @staticmethod
    def _generate_split_indices(session_index: CsvDatasetIndex, split_ratios: Dict[SplitNames, float]):
        length = len(session_index)
        indices = list(range(length))
        random.shuffle(indices)

        splits = dict()
        remainder = indices
        for split_name, ratio in split_ratios.items():
            remainder_length = len(remainder)
            num_samples_in_split = int(math.ceil(ratio * length))

            # take only what is left for the last split to avoid errors
            num_samples_in_split = min(num_samples_in_split, remainder_length)

            samples = remainder[:num_samples_in_split]
            remainder = remainder[num_samples_in_split:]

            splits[split_name] = samples

        return TrainValidationTestSplitIndices(train_indices=splits[SplitNames.train],
                                               validation_indices=splits[SplitNames.validation],
                                               test_indices=splits[SplitNames.test])

    def _process_split(self, context: Context, name: str, header: str, reader: CsvDatasetReader, indices: List[int]):
        prefix = format_prefix(context.get(PREFIXES_KEY) + [name])
        output_dir = context.get(OUTPUT_DIR_KEY)
        output_file = output_dir / f"{prefix}.csv"

        # Write the CSV file for the current split
        self._write_split(output_file, header, reader, indices)

        # Modify context such that the operations occur in the new output directory using the split file
        cloned = copy.deepcopy(context)
        cloned.set(OUTPUT_DIR_KEY, output_dir, overwrite=True)
        cloned.set(MAIN_FILE_KEY, output_file, overwrite=True)
        current_prefixes = context.get(PREFIXES_KEY)
        cloned.set(PREFIXES_KEY, current_prefixes + [name], overwrite=True)

        # Apply the necessary preprocessing, i.e. session index generation
        for action in self.inner_actions:
            action(cloned)

    @staticmethod
    def _write_split(output_file: Path, header: str, reader: CsvDatasetReader, indices: List[int]):
        with open(output_file, "w") as f:
            f.write(header + "\n")
            for index in indices:
                sample = reader.get_sequence(index)
                f.write(sample.strip() + "\n")

    @staticmethod
    def _get_header(input_file: Path) -> str:
        with open(input_file, "r") as f:
            return f.readline().strip()
