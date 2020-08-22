from argparse import ArgumentParser


class ModelConfig(object):
    """
    base model config
    """

    MODEL_CONFIG_CONFIG_FILE = 'config_file'
    MODEL_CONFIG_ITEM_VOC_SIZE = 'item_voc_size'
    MODEL_CONFIG_MAX_SEQ_LENGTH = 'max_seq_length'

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--{}'.format(ModelConfig.MODEL_CONFIG_CONFIG_FILE), type=str, default=None,
                            help='path to a config file')
        parser.add_argument('--{}'.format(ModelConfig.MODEL_CONFIG_ITEM_VOC_SIZE), type=int, default=None,
                            help='the item voc size')
        parser.add_argument('--{}'.format(ModelConfig.MODEL_CONFIG_MAX_SEQ_LENGTH), type=int, default=None,
                            help='max length of the sequence')

        return parser

    def __init__(self,
                 item_voc_size: int,
                 max_seq_length: int):
        super().__init__()
        self.item_voc_size = item_voc_size
        self.max_seq_length = max_seq_length
