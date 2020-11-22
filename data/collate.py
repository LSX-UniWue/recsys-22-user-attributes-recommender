from functools import partial
from typing import List, Callable, Any, Union, Dict

import torch


def padded_session_collate(pad_token_id: int,
                           entries_to_pad: Dict[str, List[int]],
                           session_length_entry: str = "session"
                           ):
    """
        Pads sequences with a padding token to `max_length`.

    :param pad_token_id: the id of the pad token (see Tokenizer).
    :param entries_to_pad: a list of entries in the dictionary that need to be padded.
    :param session_length_entry: the name of the entry that is used to determine individual session length.

    :return: a collate function that can be used to collate session data.
    """
    return partial(_padded_session_collate, pad_token_id, entries_to_pad, session_length_entry)


def _padded_session_collate(pad_token_id: int,
                            entries_to_pad: Dict[str, List[int]],
                            session_length_entry,
                            batch):
    from torch.utils.data.dataloader import default_collate

    def pad(x: List[Any],
            generate_padding: Union[Callable[[int], Any], partial],
            padded_length: int
            ) -> torch.Tensor:
        length = len(x)
        padded_x = x + generate_padding(length)
        # truncate if the sequence is longer
        padded_x = padded_x[:padded_length]

        return padded_x

    def _single_item_pad(length: int,
                         pad_length: int
                         ) -> List[int]:
        return [pad_token_id] * (pad_length - length)

    padded_batch = []
    for sample in batch:
        padded_sample = dict(sample)
        padded_sample["length"] = len(padded_sample[session_length_entry])

        for entry_name, value in padded_sample.items():
            value_to_convert = value
            if isinstance(value, list) and entry_name in entries_to_pad:
                max_values = entries_to_pad[entry_name]
                max_length = max_values[0]
                if isinstance(value[0], list):
                    max_seq_step_length = max_values[1]
                    # first pad entries in the list to the maximum seq step length
                    padded_entries = [
                        pad(value_entry, partial(_single_item_pad, pad_length=max_seq_step_length), max_seq_step_length) for value_entry in value
                    ]

                    value_to_convert = pad(padded_entries, lambda length: [[pad_token_id] * max_seq_step_length] * (max_length - length), max_length)
                else:
                    value_to_convert = pad(value, partial(_single_item_pad, pad_length=max_length), max_length)

            padded_sample[entry_name] = torch.as_tensor(value_to_convert)

        padded_batch.append(padded_sample)

    collated_batch = default_collate(padded_batch)
    return collated_batch
