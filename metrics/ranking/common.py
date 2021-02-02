import torch


def reverse_cumsum(input: torch.Tensor,
                   dim: int
                   ) -> torch.Tensor:
    """
    :param input: Tensor to calculate the right to left cumulative sum on
    :param dim: Dimension to use for cumulative sum calculation
    :return: A tensor of same shape as input containing the right to left cumulative sum
    """
    return torch.flip(torch.cumsum(torch.flip(input, dims=[dim]), dim=dim), dims=[dim])


def intersection(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    :param x: Tensor of shape (N,D)
    :param y: Tensor of shape (N,I)
    :return: A Tensor of shape (N,D) containing a 1 at index (i,j) iff x[i][j] is in y[i] and a zero otherwise.
    """
    hits = []
    for i, sample in enumerate(x):
        curr_hits = []
        for val in sample:
            curr_hits.append(1 if val in y[i] else 0)
        hits.append(curr_hits)
    return torch.tensor(data=hits, device=x.device)


def get_tp(predictions: torch.Tensor,
           target: torch.Tensor,
           mask: torch.Tensor,
           k: int
           ) -> torch.Tensor:
    sorted_indices = torch.topk(predictions, k=k).indices
    sorted_indices = torch.repeat_interleave(sorted_indices, mask.size()[-1], dim=0)

    target_expanded = target.view(-1, 1).expand(-1, k)
    tp_mask = torch.repeat_interleave(mask.view(-1, 1), k, dim=1)

    tp = (sorted_indices.eq(target_expanded) * tp_mask).sum(dim=-1).to(dtype=torch.float)
    return tp.view(mask.size()).sum(-1)


def calc_precision(prediction: torch.Tensor,
                   target: torch.Tensor,
                   k: int,
                   mask: torch.Tensor = None
                   ) -> torch.Tensor:
    if len(target.size()) == 1:
        # single dimension, unsqueeze it
        target = torch.unsqueeze(target, dim=1)

    if mask is None:
        mask = torch.ones(target.size(), device=target.device)

    tp = get_tp(predictions=prediction, target=target, mask=mask, k=k)

    return tp / k


def calc_recall(prediction: torch.Tensor,
                target: torch.Tensor,
                k: int,
                mask: torch.Tensor = None
                ) -> torch.Tensor:
    if len(target.size()) == 1:
        # single dimension, unsqueeze it
        target = torch.unsqueeze(target, dim=1)

    if mask is None:
        mask = torch.ones(target.size(), device=target.device)

    tp = get_tp(predictions=prediction, target=target, mask=mask, k=k)
    fn = mask.sum(-1) - tp

    return tp / (tp + fn)


def calc_dcg(prediction: torch.Tensor,
             target: torch.Tensor,
             k: int,
             mask: torch.Tensor = None
             ) -> torch.Tensor:
    """
    Calculates DCG according to https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    :param prediction:
    :param target:
    :param k:
    :param mask:
    :return:
    """

    # ensure target is two-dimensional for the following computations to work properly
    if len(target.size()) == 1:
        target = torch.unsqueeze(target, dim=1)

    # remove masked items if a mask was provided
    masked_target = torch.where(mask > 0, target, -torch.ones_like(target)) if mask is not None else target

    # find the indices of the k items the model believes are the most relevant
    topk = torch.argsort(prediction, descending=True)[:, :k]
    # find the indices of the elements actually matching the targets
    matches = intersection(topk, masked_target)

    # build every possible i till the k position
    positions = torch.arange(2, k + 2, dtype=prediction.dtype)
    # calc every 1 / log2(i+1) and than mask it with the matches
    position_values = 1 / torch.log2(positions) * matches
    return position_values.sum(dim=1)