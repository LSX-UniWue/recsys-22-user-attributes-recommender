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

    def get_selected_items(self):
        return self.selected_items






