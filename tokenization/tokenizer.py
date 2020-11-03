from tokenization import SPECIAL_TOKENS_ATTRIBUTES
from tokenization.vocabulary import Vocabulary

from typing import List, Optional, Union


class Tokenizer:

    def __init__(self,
                 vocabulary: Vocabulary,
                 **kwargs
                 ):
        self.vocabulary = vocabulary

        self.bos_token = None
        self.eos_token = None
        self.unk_token = None
        self.mask_token = None
        self.pad_token = None
        self.sep_token = None
        self.cls_token = None

        for key, value in kwargs.items():
            if key in SPECIAL_TOKENS_ATTRIBUTES:
                assert isinstance(value, str)
                setattr(self, key, value)

    @property
    def pad_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def mask_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def bos_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def sep_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def cls_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.cls_token)

    def get_vocabulary(self) -> Vocabulary:
        return self.vocabulary

    def convert_tokens_to_ids(self,
                              items: Union[str, List[str], List[List[str]]]
                              ) -> Union[Optional[int], List[int]]:
        if items is None:
            return None

        if isinstance(items, str):
            return self._convert_item_to_id(items)

        ids = []
        for item in items:
            ids.append(self._convert_item_to_id(item))
        return ids

    #TODO (AD) this should be a separate class
    def get_special_tokens_mask(self,
                                token_ids: List[int],
                                second_item_ids: List[int] = None,
                                already_has_special_tokens: bool = False
                                ) -> List[bool]:
        if already_has_special_tokens:
            if second_item_ids is not None:
                raise ValueError("You should not supply a second sequence if the provided sequence of "
                                 "ids is already formatted with special tokens for the model.")
            return list(map(lambda x: x in [self.sep_token_id, self.cls_token_id], token_ids))

        if second_item_ids is None:
            return [True] + ([False] * len(token_ids)) + [True]
        return [True] + ([False] * len(token_ids)) + [True, True] + ([False] * len(second_item_ids)) + [True]

    def _convert_item_to_id(self,
                            token: Union[str, List[str]]
                            ) -> Optional[int]:
        """
        Converts the token into its id. If the token is not part of the vocabulary and the unk_token property is set,
        the id for the unk_token will be returned. Otherwise if the token can not be found, None is returned.

        :param token: a token
        :return: the token id if the token is part of the vocabulary, otherwise if possible the id of the unk_token is
         returned else the return value is None
        """
        if token is None:
            return None

        if isinstance(token, str):
            token_id = self.vocabulary.get_id(token)
            if token_id is None:
                if self.unk_token is not None:
                    return self.unk_token_id
            return token_id

        # here we assume it is a list
        ids = []
        for item in token:
            encoded_token = self._convert_item_to_id(item)
            ids.append(encoded_token)
        return ids

    def __len__(self):
        """ Size of the full vocabulary with the added tokens """
        return len(self.vocabulary)
