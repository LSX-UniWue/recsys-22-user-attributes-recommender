from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning import Callback

from data.datasets import SAMPLE_IDS, ITEM_SEQ_ENTRY_NAME
from modules.constants import RETURN_KEY_PREDICTIONS, RETURN_KEY_TARGETS
from tokenization.tokenizer import Tokenizer


class PredictionLoggerCallback(Callback):
    """
    Logs all predictions made by the model during the test run into a CSV file of the following format:
    ```
       <sample id>  <prediction pos>    <predicted item (id)>   <probability>   <target item>   <input sequence>
    ```
    """
    HEADER = ["SID", "RANK", "ITEM", "PROBABILITY", "TARGET"]
    INPUT_HEADER_NAME = "INPUT"

    def __init__(self, output_file_path: Path,
                 log_input: bool = False,
                 tokenizer: Optional[Tokenizer] = None,
                 strip_padding_tokens: Optional[bool] = False):

        self.output_file_path = output_file_path
        self.log_input = log_input
        self.tokenizer = tokenizer
        self.strip_padding_tokens = strip_padding_tokens

        if strip_padding_tokens and not tokenizer:
            print("Warning: You requested to strip the padding tokens but no tokenizer was set. Disabling stripping.")
            self.strip_padding_tokens = False

        # will be initialized upon call to `setup`
        self.output_file_handle = None
        self.csv_writer = None

    def setup(self, trainer, pl_module, stage: str):
        import csv
        self.output_file_handle = self.output_file_path.open(mode="w")
        self.csv_writer = csv.writer(self.output_file_handle, delimiter="\t")

    def on_test_start(self, trainer, pl_module):
        header = list(self.HEADER)

        if self.log_input:
            header.append(self.INPUT_HEADER_NAME)

        self.csv_writer.writerow(header)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        probs = torch.softmax(outputs[RETURN_KEY_PREDICTIONS], dim=-1)
        values, indices = probs.topk(k=20)

        num_samples_in_batch = indices.size()[0]
        num_values_in_sample = indices.size()[1]
        sample_ids = batch[SAMPLE_IDS]

        buffer = []
        for in_batch_idx in range(num_samples_in_batch):
            for value_idx in range(num_values_in_sample):
                rank = value_idx + 1
                output = [
                    sample_ids[in_batch_idx].item(),
                    rank,
                    indices[in_batch_idx][value_idx].item(),
                    values[in_batch_idx][value_idx].item(),
                    outputs[RETURN_KEY_TARGETS][in_batch_idx].item()
                ]

                if self.log_input:
                    input_seq = batch[ITEM_SEQ_ENTRY_NAME][in_batch_idx].tolist()

                    if self.strip_padding_tokens:
                        input_seq = list(filter(lambda x: x != self.tokenizer.pad_token_id, input_seq))

                    output.append(input_seq)

                buffer.append(output)

        self.csv_writer.writerows(buffer)

    def teardown(self, trainer, pl_module, stage: str):
        self.csv_writer = None
        self.output_file_handle.close()
        self.output_file_handle = None