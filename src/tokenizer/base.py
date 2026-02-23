from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):
    """Abstract tokenizer interface: encode text -> ids, decode ids -> text."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        pass
