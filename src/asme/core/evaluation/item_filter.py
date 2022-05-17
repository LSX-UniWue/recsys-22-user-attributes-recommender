from typing import Tuple, List

import torch
from pathlib import Path
import numpy as np
from asme.core.utils.ioutils import load_file_with_item_ids

class FilterPredictionItems:

    def __init__(self, selected_items_file: Path, device):

        self.selected_items = None
        if selected_items_file is not None:
            self.selected_items = load_file_with_item_ids(selected_items_file)
            selected_items_tensor = torch.tensor(self.selected_items, dtype=torch.int32, device=device)

            def _selected_items_filter(sample_predictions):
                return torch.index_select(sample_predictions, 1, selected_items_tensor)

            filter_predictions = _selected_items_filter
        else:
            def _noop_filter(sample_predictions: np.ndarray):
                return sample_predictions

            filter_predictions = _noop_filter

        self.filter_predictions = filter_predictions

    def get_filter(self):
        return self.filter_predictions

    def get_selected_items(self):
        return self.selected_items






