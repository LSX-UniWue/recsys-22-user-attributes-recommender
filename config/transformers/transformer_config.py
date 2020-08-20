class TransformerConfig(object):

    def __init__(self,
                 item_voc_size: int,
                 max_seq_length: int,
                 d_model: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 transformer_dropout: float):
        super().__init__()
        self.item_voc_size = item_voc_size
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.num_transformer_heads = num_transformer_heads
        self.num_transformer_layers = num_transformer_layers
        self.transformer_dropout = transformer_dropout
