from datasets.dataset_index_splits.RatioSplitStrategy import RatioSplitStrategy


def get_ratio_strategy(train_ratio: float, test_ratio: float, validation_ratio: float, seed: int):
    return RatioSplitStrategy(train_ratio, test_ratio, validation_ratio, seed)
