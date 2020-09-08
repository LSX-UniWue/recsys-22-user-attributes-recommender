from functools import partial

import torch


def padded_session_collate(max_length):
    return partial(_padded_session_collate, max_length)


def _padded_session_collate(max_length, batch):
    from torch.utils.data.dataloader import default_collate

    def pad_session(sample):
        session = sample["session"]
        if len(session) < max_length:
            length = len(session)
            session.extend([0] * (max_length - length))

        sample["session"] = torch.as_tensor(session)
        sample["session_length"] = length
        return sample

    updated_batch = [pad_session(sample) for sample in batch]

    return default_collate(updated_batch)
