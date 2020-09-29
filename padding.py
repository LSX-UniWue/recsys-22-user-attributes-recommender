from functools import partial

import torch


def padded_session_collate(max_length: int, pad_token_id: int):
    return partial(_padded_session_collate, max_length, pad_token_id)


def _padded_session_collate(max_length: int, pad_token_id: int, batch):
    from torch.utils.data.dataloader import default_collate

    def pad_session(sample):
        session = sample["session"]
        length = len(session)
        if length < max_length:
            session.extend([pad_token_id] * (max_length - length))

        sample["session"] = torch.as_tensor(session)
        sample["session_length"] = length
        return sample
    updated_batch = [pad_session(sample) for sample in batch]

    return default_collate(updated_batch)
