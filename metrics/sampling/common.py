import torch


def _get_true_positives(prediction: torch.Tensor,
                        positive_item_mask: torch.Tensor,
                        k: int
                        ) -> torch.Tensor:
    # get the indices of the top k predictions
    predicted_id = torch.argsort(prediction, descending=True)
    # limit the prediction to the top k
    predicted_id = predicted_id[:, :k]

    # select the mask for each of the indices, this is than the tp
    return positive_item_mask.gather(1, predicted_id)


def get_true_positive_count(prediction: torch.Tensor,
                            positive_item_mask: torch.Tensor,
                            k: int
                            ) -> torch.Tensor:
    """
    returns the number of true positives for
    :param k: k for the @k metric
    :param prediction: the prediction logits :math`(N, I)`
    :param positive_item_mask: a mask where 1 at position i indicates that the item at index i is relevant :math`(N, I)`
    :return: the number of true positives (N)
    """
    tp = _get_true_positives(prediction, positive_item_mask, k)
    return tp.sum(1)  # get the total number of positive items


def calc_recall(prediction: torch.Tensor,
                positive_item_mask: torch.Tensor,
                k: int
                ) -> torch.Tensor:
    """
    calcs the recall given the predictions and postive item mask
    :param positive_item_mask: a mask, where 1 at index i indicates that item i in predictions is relevant
    :param prediction: the prediction logits :math`(N, I)
    :param k: the k
    :return: the recall :math`(N)`

    where N is the batch size and I the number of items to consider
    """
    tp = get_true_positive_count(prediction, positive_item_mask, k)
    all_relevant_items = positive_item_mask.sum(dim=1)
    recall = tp / all_relevant_items
    # maybe there are no relevant items
    recall[torch.isnan(recall)] = 0
    return recall


def calc_precision(prediction: torch.Tensor,
                   positive_item_mask: torch.Tensor,
                   k: int
                   ) -> torch.Tensor:
    """
    calcs the precision given the predictions and positive item mask
    :param positive_item_mask: a mask, where 1 at index i indicates that item i in predictions is relevant
    :param prediction: the prediction logits :math`(N, I)
    :param k: the k
    :return: the precision :math`(N)`

    where N is the batch size and I the number of items to consider
    """

    tp = get_true_positive_count(prediction, positive_item_mask, k)
    return tp / k


def calc_ndcg(prediction: torch.Tensor,
              positive_item_mask: torch.Tensor,
              k: int
              ) -> torch.Tensor:
    """
    calcs the ndcg given the predictions and positive item mask
    :param positive_item_mask: a mask, where 1 at index i indicates that item i in predictions is relevant
    :param prediction: the prediction logits :math`(N, I)
    :param k: the k
    :return: the ndcg :math`(N)`

    where N is the batch size and I the number of items to consider
    """

    tp = _get_true_positives(prediction, positive_item_mask, k)
    range = torch.arange(2, k + 2).to(dtype=torch.float)

    dcg = 1 / torch.log2(range)
    dcg = dcg.unsqueeze(0).repeat(positive_item_mask.size()[0], 1)
    dcg = dcg * tp

    idcg_all = 1 / torch.log2(range)
    idcg_all = idcg_all.unsqueeze(0).repeat(positive_item_mask.size()[0], 1)

    number_relevant_items = positive_item_mask.sum(dim=1)
    number_relevant_items = torch.where(number_relevant_items <= k, number_relevant_items, k)
    number_relevant_items = number_relevant_items.unsqueeze(1).repeat(1, k)
    relevant_mask = torch.arange(0, k).lt(number_relevant_items)
    idcg = idcg_all * relevant_mask

    return dcg / idcg.sum(dim=1)
