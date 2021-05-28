from dataclasses import dataclass
from typing import List, Any, Dict, Optional

import torch


@dataclass
class InputSequence:

    sequence: torch.Tensor
    """
    A sequence tensor with item ids. :math:`(N, S)` or :math:`(N, S, BS)`.
    """

    padding_mask: torch.Tensor
    """
    A mask that contains positions of padding tokens. :math:`(N, S)`.
    """

    attributes: Dict[str, Any]
    """
    Attributes that can be used to contextualize the sequence.
    """

    def get_attributes(self) -> List[str]:
        return list(self.attributes.keys())

    def has_attribute(self, name: str) -> bool:
        return name in self.attributes

    def get_attribute(self, name: str) -> Optional[Any]:
        if self.has_attribute(name):
            return self.attributes[name]

        return None


@dataclass
class EmbeddedElementsSequence:

    embedded_sequence: torch.Tensor
    """
    The embedded sequence of the provided items (and maybe more) :math: `(N, S, H)`.
    """

    input_sequence: Optional[InputSequence] = None


@dataclass
class SequenceRepresentation:  # -> Sequence Representation
    encoded_sequence: torch.Tensor
    """
    An encoded sequence representation. :math:`(N, S, R)`.
    """

    embedded_elements_sequence: Optional[EmbeddedElementsSequence] = None


@dataclass
class ModifiedSequenceRepresentation:

    modified_encoded_sequence: torch.Tensor
    """
    A modification of the encoded sequence. :math:`(N, S, T)`
    """

    sequence_representation: Optional[SequenceRepresentation] = None
