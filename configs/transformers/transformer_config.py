from argparse import ArgumentParser

from configs.config import ModelConfig


class TransformerConfig(ModelConfig):
    """
    base config for transformer based configuration
    """

    TRANSFORMER_CONFIG_DROPOUT = 'transformer_dropout'
    TRANSFORMER_CONFIG_NUM_LAYERS = 'transformer_num_layers'
    TRANSFORMER_CONFIG_NUM_HEADS = 'transformer_num_heads'
    TRANSFORMER_CONFIG_HIDDEN_SIZE = 'transformer_hidden_size'

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ModelConfig.add_model_specific_args(parent_parser)
        parser.add_argument('--{}'.format(TransformerConfig.TRANSFORMER_CONFIG_HIDDEN_SIZE), type=int, default=None,
                            help='the hidden size of the transformer')
        parser.add_argument('--{}'.format(TransformerConfig.TRANSFORMER_CONFIG_NUM_HEADS), type=int, default=None,
                            help='the number of heads for the transformer encoder')
        parser.add_argument('--{}'.format(TransformerConfig.TRANSFORMER_CONFIG_NUM_LAYERS), type=int, default=None,
                            help='the number of transformer layers')
        parser.add_argument('--{}'.format(TransformerConfig.TRANSFORMER_CONFIG_DROPOUT), type=float, default=None,
                            help='the dropout to use through the whole transformer model')
        return parser

    def __init__(self,
                 item_voc_size: int,
                 max_seq_length: int,
                 d_model: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 transformer_dropout: float):
        super().__init__(item_voc_size=item_voc_size,
                         max_seq_length=max_seq_length)

        self.d_model = d_model
        self.num_transformer_heads = num_transformer_heads
        self.num_transformer_layers = num_transformer_layers
        self.transformer_dropout = transformer_dropout
