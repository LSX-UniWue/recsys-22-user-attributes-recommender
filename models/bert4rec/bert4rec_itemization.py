from utils.itemization_utils import PreTrainedItemizer, load_vocab


class BERTItemizer(PreTrainedItemizer):

    def __init__(self,
                 vocab_file,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 **kwargs):

        super(BERTItemizer, self).__init__(unk_token=unk_token, sep_token=sep_token,
                                           pad_token=pad_token, cls_token=cls_token,
                                           mask_token=mask_token, **kwargs)
        self.vocab = load_vocab(vocab_file)

    def _convert_item_to_id(self, item):
        return self.vocab.get(item, self.vocab.get(self.unk_token))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)