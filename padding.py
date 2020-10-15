from functools import partial
from typing import Optional, List

import torch


def padded_session_collate(max_length: int, pad_token_id: int, entries_to_pad: List[str] = ["session"], session_length_entry: str = "session"):
    """
        Pads sequences with a padding token to `max_length`.

    :param max_length: the maximum sequence length.
    :param pad_token_id: the id of the pad token (see Tokenizer).
    :param entries_to_pad: a list of entries in the dictionary that need to be padded.
    :param session_length_entry: the name of the entry that is used to determine individual session length.

    :return: a collate function that can be used to collate session data.
    """
    return partial(_padded_session_collate, max_length, pad_token_id, entries_to_pad, session_length_entry)


def _padded_session_collate(max_length: int, pad_token_id: int, entries_to_pad: List[str], session_length_entry: str, batch):
    from torch.utils.data.dataloader import default_collate

    def pad(x: List[int], pad_token_id: int, padded_length: int):
        length = len(x)
        padded_x = x
        if length < padded_length:
            padded_x = padded_x + ([pad_token_id] * (max_length - length))
        elif length > padded_length:  # truncate if the sequence is longer
            padded_x = padded_x[:padded_length]

        return torch.as_tensor(padded_x)

    padded_batch = []
    for sample in batch:
        padded_sample = dict(sample)
        padded_sample["length"] = len(padded_sample[session_length_entry])

        for entry_name, value in padded_sample.items():
            if entry_name in entries_to_pad:
                padded_sample[entry_name] = pad(value, pad_token_id, max_length)
            else:
                padded_sample[entry_name] = torch.as_tensor(value)

        padded_batch.append(padded_sample)

    collated_batch = default_collate(padded_batch)

    return collated_batch
