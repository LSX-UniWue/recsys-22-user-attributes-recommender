from modules.constants import LOG_KEY_VALIDATION_LOSS, LOG_KEY_TEST_LOSS, LOG_KEY_TRAINING_LOSS
from modules.cosrec_module import CosRecModule
from modules.sas_rec_module import SASRecModule
from modules.bert4rec_module import BERT4RecModule
from modules.caser_module import CaserModule

__all__ = [
    "BERT4RecModule",
    "SASRecModule",
    "CaserModule",
    "CosRecModule"
    "LOG_KEY_VALIDATION_LOSS",
    "LOG_KEY_TEST_LOSS",
    "LOG_KEY_TRAINING_LOSS"
]