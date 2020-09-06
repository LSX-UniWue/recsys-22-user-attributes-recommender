from tokenization.tokenizer import Tokenizer, Vocabulary


class BERTTokenizer(Tokenizer):

    def __init__(self,
                 vocabulary: Vocabulary,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 **kwargs):

        super(BERTTokenizer, self).__init__(vocabulary, unk_token=unk_token, sep_token=sep_token,
                                            pad_token=pad_token, cls_token=cls_token,
                                            mask_token=mask_token, **kwargs)
