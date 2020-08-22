from argparse import ArgumentParser

from configs.config import ModelConfig
from configs.transformers.transformer_config import TransformerConfig
from configs.utils.args_utils import get_config_from_args


class SASRecConfig(TransformerConfig):
    """
    configuration for a SASRecModel
    """

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        return TransformerConfig.add_model_specific_args(parent_parser)

    @classmethod
    def from_args(cls, **kwargs):
        config_file = kwargs.get('config_file', None)
        if config_file is not None:
            return SASRecConfig.from_file(config_file)

        hidden_size = get_config_from_args(kwargs, TransformerConfig.TRANSFORMER_CONFIG_HIDDEN_SIZE, 64)
        num_heads = get_config_from_args(kwargs, TransformerConfig.TRANSFORMER_CONFIG_NUM_HEADS, 4)
        num_layers = get_config_from_args(kwargs, TransformerConfig.TRANSFORMER_CONFIG_NUM_LAYERS, 4)
        dropout = get_config_from_args(kwargs, TransformerConfig.TRANSFORMER_CONFIG_DROPOUT, 0.5)
        max_seq_length = get_config_from_args(kwargs, ModelConfig.MODEL_CONFIG_MAX_SEQ_LENGTH, 64)
        item_voc_size = get_config_from_args(kwargs, ModelConfig.MODEL_CONFIG_ITEM_VOC_SIZE, 32)
        return SASRecConfig(max_seq_length=max_seq_length,
                            item_voc_size=item_voc_size,
                            d_model=hidden_size,
                            num_transformer_heads=num_heads,
                            num_transformer_layers=num_layers,
                            transformer_dropout=dropout)

    @classmethod
    def from_file(cls, config_file: str):
        raise NotImplemented("file loading not implemented")
