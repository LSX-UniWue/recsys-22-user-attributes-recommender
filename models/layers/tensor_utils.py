import torch


def generate_position_ids(shape,
                          device: torch.device = None):
    position_ids = torch.arange(shape[1], dtype=torch.long, device=device)
    return position_ids.unsqueeze(0).repeat(shape[0], 1)


def generate_square_subsequent_mask(size: int):
    """
    returns a mask for only considering the previous items in the
    :param size:
    :return:
    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
